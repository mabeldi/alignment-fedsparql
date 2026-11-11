from typing import List, Dict, Set
import re
from pathlib import Path
from tools.llm_router import run_agent_async
from tools.path_sparql import VarNamer, render_path
from tools.prompt_format import format_edge_graph

from config.sparql_endpoints import SPARQL_ENDPOINTS

# The link that carries a cross-source join. 
SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"

# Path rendering strategy for the prompt.
#   True  = node-grouped edge list (tools/prompt_format.format_edge_graph): every
#           distinct edge stated once, CURIEs under a PREFIX header, class
#           hierarchy included. ~30x fewer tokens on the measured payload.
#   False = original per-path enumeration (format_paths_by_source below), which
#           repeated each edge ~12 times.
# Same switch pattern as USE_CONNECTOR_PATHS in main.py, so the old rendering can
USE_EDGE_GRAPH_PATHS = True

# Whether the model may DERIVE a triple instead of only copying a printed one.
#   False = every triple it writes is a line shown in the schema sections.
#   True  = it may pair the given names into a triple the schema does not print,
#           when the hierarchy makes that pairing sound, and it works the query
#           out in prose before writing it. `_extract_query` reads the query out
#           of that prose, so the reply shape stays the same either way.
AGENT7_REASONING = True

# The parts of OUTPUT that do not depend on AGENT7_REASONING. Held apart so the
# two variants cannot drift on what the query itself must look like.
# No `- ` bullets here: in the reasoning variant `- ` marks the model's plan
# lines, and the query is the first line without it.
_OUTPUT_SHAPE = """Project exactly the fields asked, no others (extra only if a requested computation needs it). WHICH things -> return the things themselves; a text value only for a field whose value is explicitly asked.
Layout: PREFIX lines, `SELECT DISTINCT` line, WHERE, ONE brace pair holding every pattern incl. SERVICE blocks. The query ends at its closing `}` or solution modifier.
Always `SELECT DISTINCT`; never ASK, CONSTRUCT or DESCRIBE."""

# Rule 3 of RULES. Numbered 3 because the rules that used to sit between it and
# rule 5 were restatements of the schema legend and are now only stated there.
_RULE_ONE_STRICT = """3. Copy, never invent: every class, property, prefix and endpoint verbatim from above. If this vocabulary cannot express the question, reply exactly `NO QUERY` (correct; a query with invented names is wrong)."""

_RULE_ONE_REASONING = """3. Only the NAMES given: every class, property, prefix and endpoint verbatim from above; never adapt, shorten or guess a name. One derivation allowed: a relation declared between two classes also holds between any subclass of the first and any subclass of the second (per HIERARCHY), so you may write that triple though it is not printed. Derive nothing else. If this vocabulary cannot express the question, reply exactly `NO QUERY` (correct; a query with invented names is wrong)."""

_OUTPUT_STRICT = f"""Unless rule 3 applies, reply with one complete query only.
{_OUTPUT_SHAPE}
No preamble, explanation, ``` fences or `#` comments."""

_OUTPUT_REASONING = f"""First plan in <=8 short lines, each starting `- `: what is retrieved and which sources hold it; the rule 5 shape; the triples to write, citing for any unprinted triple the hierarchy fact that allows it. The first line not starting `- ` is the query. Write the query, then stop.
{_OUTPUT_SHAPE}
Nothing after the query; no ``` fences or `#` comments in it."""

SKILL_PATH = Path(__file__).resolve().parent / "skills" / "sparql.md"

_skill_cache: str = ""


def load_skill(path: Path = None) -> str:
    global _skill_cache
    if _skill_cache:
        return _skill_cache

    target = path or SKILL_PATH
    text = target.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[text.find("\n", end + 1) + 1:]
    _skill_cache = text.strip()
    return _skill_cache

def format_paths_by_source(paths: List[Dict[str, any]], base_source: str = None) -> str:
    grouped = {}
    for item in paths:
        source = item.get("source")
        class_uri = item.get("class")
        class_name = item.get("label")
        if not source or not class_name:
            continue

        key = (source, class_uri, class_name)
        grouped.setdefault(key, []).extend(item.get("outgoing", []) + item.get("incoming", []))

    # Name variables from the anchor source first so the other sources inherit
    # its naming through the alignment and the UNION branches share columns.
    ordered = sorted(grouped.items(), key=lambda kv: kv[0][0] != base_source)
    namer = VarNamer(set(SPARQL_ENDPOINTS))

    output = []
    for (source, class_uri, class_name), path_list in ordered:
        output.append(f"Source: {source}")
        output.append(f"Class: {class_name}")
        output.append(f"Class Uri: {class_uri}")
        output.append(f"Endpoint: <{SPARQL_ENDPOINTS.get(source, source)}>")
        output.append("Class paths:")
        for path in dict.fromkeys(path_list):
            output.append(f"- {path}")
            triples = render_path(path, source, namer)
            if triples:
                for line in triples.splitlines():
                    output.append(f"      {line}")
        output.append("")  # Spacer between entries

    return "\n".join(output)


def build_messages(question: str, summarized_paths: List[Dict[str, any]], base_source: str = None,
                   alignment_block: str = "", sources: Set[str] = None,
                   extracted: Dict[str, List[str]] = None,
                   provider_plan: List[Dict[str, any]] = None) -> List[Dict[str, str]]:
    """The prompt Agent 7 is called with."""
    return build_messages_skill(question, summarized_paths, base_source,
                                alignment_block, sources, extracted, provider_plan)


def _endpoint_lines(sources: Set[str]) -> str:
    active = {name: url for name, url in SPARQL_ENDPOINTS.items()
              if not sources or name in sources}
    return "\n".join(f"  {name} -> <{url}>" for name, url in active.items())


def build_messages_skill(question: str, summarized_paths: List[Dict[str, any]],
                         base_source: str = None, alignment_block: str = "",
                         sources: Set[str] = None,
                         extracted: Dict[str, List[str]] = None,
                         provider_plan: List[Dict[str, any]] = None) -> List[Dict[str, str]]:
    formatted_paths = (format_edge_graph(summarized_paths, base_source, SPARQL_ENDPOINTS)
                       if USE_EDGE_GRAPH_PATHS
                       else format_paths_by_source(summarized_paths, base_source))

    anchor_line = ""
    if base_source:
        anchor_line = (f"\nAnchor: **{base_source}**. Its block comes first, with no link line; "
                       f"a block JOINED to it opens with a link line to its variable (rule 5).")

    extracted = extracted or {}
    # GTX Give the generator the same resource-versus-literal contract used by
    # GTX retrieval and source selection; the question text alone proved unsafe.
    semantic_shape = (
        "REQUESTED SEMANTIC SHAPE\n"
        f"Resource concepts: {', '.join(extracted.get('classes', [])) or '(none)'}\n"
        f"Resource relationships: "
        f"{', '.join(extracted.get('objectProperties', [])) or '(none)'}\n"
        f"Literal values: "
        f"{', '.join(extracted.get('dataTypeProperties', [])) or '(none)'}"
    )
    provider_plan = provider_plan or []
    provider_lines = ("\n".join(
        f"- {item['relation']}: use <{item['property']}> in source {item['source']}"
        for item in provider_plan
    ) or "- (none; infer from SCHEMA)")

    user_msg = f"""QUESTION
"{question}"

{semantic_shape}

CLAUSE PROVIDER PLAN
{provider_lines}

ENDPOINTS (the only SERVICE URLs allowed; copy verbatim)
{_endpoint_lines(sources)}{anchor_line}

SCHEMA (read the LEGEND first)

{formatted_paths}

ALIGNMENT (cross-source correspondences; same prefixes as SCHEMA)

{alignment_block}

RULES
1. Semantic kind: a Resource concept = a class instance reached by its object-property path; never swap it for a same-named literal attribute. Use a literal only for items under Literal values.
2. CLAUSE PROVIDER PLAN, when present, is authoritative: one joined answer branch using every listed source, each clause with exactly its listed property, sources joined through ALIGNMENT. Never swap a planned property for another source's relation; no unrelated UNION branch.
{_RULE_ONE_REASONING if AGENT7_REASONING else _RULE_ONE_STRICT}
4. Across UNION branches, use one shared projected variable per requested answer field; never split that field into source-specific output columns. Rename only to add your own variable or to give a joined entity one variable per source (rule 5).
5. Shape = two separate decisions.
   ROWS: which sources hold ANSWER CLASS instances? More than one -> UNION, one branch per source (rule 6), all keeping the shared variable name. UNION branches are independent: no link line between them, none in a branch's own block. "Which X are there" is this case: it asks every source's X; a join alone covers one source.
   FIELDS, per branch: take each asked field from the branch's own source if it has it. A field it lacks -> join, after the branch's own block: `OPTIONAL {{ SERVICE <source that has it> {{ ?joined <{SAME_AS}> ?branch_own . ... }} }}`, link line first. The joined entity gets ONE VARIABLE PER SOURCE; project the branch's own variable. Drop OPTIONAL only when the question makes the field a hard condition (entities lacking it must not appear).
   One row source, fields split -> join only. Several row sources, each with all its fields -> UNION only. Both -> UNION with joins inside the branches.
   owl:sameAs appears ONLY as the link line of a join between two SERVICE blocks (same row); a UNION never needs it.
6. Each endpoint holds only its OWN instances. UNION shape: include every source under SOURCES HOLDING THE ANSWER, each in its own block (attribute coverage is not instance coverage; a source supplying no attribute still supplies rows). Join shape: include only sources supplying an asked part; any other block only multiplies or filters rows.
7. ALIGNMENT: check each correspondence one by one; use every relevant one to build the answer patterns across allowed sources, keeping its conditions. A correspondence = the SAME concept at different endpoints, with different instances. Copy its triples as written. A CONDITIONAL one maps a class to a class expression: judge it by the meaning of the WHOLE expression, all its triples together, never by its general class alone. If that meaning matches the answer, the expression IS the answer class in its source, even if the schema shows only the general class: that source holds answer rows and gets its own branch, built from all the expression's triples; dropping any changes the answer. A source appearing only in ALIGNMENT (no SCHEMA section) still belongs in the query; those triples are its only vocabulary.
8. Every source, anchor included, in its own SERVICE block. No bare pattern, no nested SERVICE. OPTIONAL wraps a whole SERVICE from outside: `OPTIONAL {{ SERVICE <url> {{ ... }} }}`, never the reverse.
9. Reuse sources already in the query: if one can supply a missing piece (judge by MEANING, not wording), do not add a SERVICE block for it. Add a new source only when none included can.

OUTPUT
{_OUTPUT_REASONING if AGENT7_REASONING else _OUTPUT_STRICT}"""

    # The closing sentence is the one part AGENT7_REASONING changes: the rest
    # holds either way, and a system message telling the model to return only a
    # query would contradict the OUTPUT section asking it to work one out first.
    reply_shape = ("Plan briefly, then write the query."
                   if AGENT7_REASONING else "Reply with the query only.")
    BASE_SYSTEM = ("You write one complete federated SPARQL SELECT DISTINCT query from the user's question and "
                   "supplied evidence (schema, alignment, endpoints). Use only the supplied "
                   "vocabulary; invent nothing; follow every stated constraint exactly. "
                   f"{reply_shape} If the evidence cannot express the question, reply `NO QUERY`.")
    return [
        {"role": "system", "content": BASE_SYSTEM + "\n\n" + load_skill()},
        {"role": "user", "content": user_msg.strip()},
    ]


# --- output cleanup ---------------------------------------------------------
# A fenced block, with or without a language tag, closed or left open (a model
# that hits its token limit often never writes the closing fence).
_FENCE = re.compile(r"```[ \t]*[A-Za-z]*[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
# Where a SPARQL query legally begins. Everything before the FIRST of these is
# preamble, whatever it says.
_QUERY_START = re.compile(
    r"^[ \t]*(PREFIX|BASE|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.IGNORECASE | re.MULTILINE
)
# The only things allowed to follow the closing brace of a WHERE clause.
_SOLUTION_MODIFIER = re.compile(
    r"^[ \t]*(ORDER[ \t]+BY|GROUP[ \t]+BY|HAVING|LIMIT|OFFSET|VALUES)\b", re.IGNORECASE
)


def _extract_query(content: str) -> str:
    """The SPARQL query inside a model reply, or "" if there is none.
    """
    if not content:
        return ""

    fenced = _FENCE.search(content)
    if fenced:
        content = fenced.group(1)

    start = _QUERY_START.search(content)
    if not start:
        return ""
    content = content[start.start():]

    end = content.rfind("}")
    if end != -1:
        tail = content[end + 1:]
        content = content[:end + 1]
        # Keep the solution modifiers, drop at the first line that is not one.
        # `LIMIT 10` after the brace is part of the query; "This query..." is not.
        for line in tail.splitlines():
            if not line.strip():
                continue
            if not _SOLUTION_MODIFIER.match(line):
                break
            content += "\n" + line.rstrip()

    return content.strip()


# --- cartesian-product detection --------------------------------------------
_SELECT_CLAUSE = re.compile(r"\bSELECT\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)
_VAR = re.compile(r"[?$]([A-Za-z_][\w]*)")
# A triple pattern's three terms, up to the separator that ends it. Deliberately
# crude: it only has to tell which variables SHARE a pattern, and a term it
# misreads as a literal still contributes no variable.
_TRIPLE = re.compile(
    r"([?$][\w]+|<[^>]+>|[\w.-]*:[\w.-]+|\ba\b)\s+"
    r"([?$][\w]+|<[^>]+>|[\w.-]*:[\w.-]+|\ba\b)\s+"
    r"([?$][\w]+|<[^>]+>|[\w.-]*:[\w.-]+|\"[^\"]*\"|[\w.-]+)",
    re.IGNORECASE,
)


def _union_branches(query: str) -> List[str]:
    """The query's top-level UNION alternatives, as raw text.

    Split on UNION at brace depth 1 (inside WHERE, outside any group), because
    each alternative is evaluated on its own — a variable joined in one branch
    is unjoined in the next, and checking the query as a whole would hide that.
    """
    where = query.find("{")
    if where == -1:
        return [query]
    depth, parts, start = 0, [], where + 1
    i = where
    while i < len(query):
        ch = query[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parts.append(query[start:i])
                break
        elif depth == 1 and query[i:i + 5].upper() == "UNION" :
            parts.append(query[start:i])
            start = i + 5
            i += 4
        i += 1
    return parts or [query]


def _disconnected_selected(branch: str, selected: Set[str]) -> List[Set[str]]:
    """Groups of SELECTed variables this branch never links together.

    Two variables are linked when they appear in the same triple pattern, or
    through a chain of such patterns. More than one group means the branch
    multiplies them: 3 people and 4 workshops with no pattern relating them
    returns all 12 pairings, 9 of which were never in the data. It looks like a
    fuller answer, which is what makes it dangerous.

    Returns [] when the branch is fine (0 or 1 group).
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for match in _TRIPLE.finditer(branch):
        vars_here = [v.lstrip("?$") for v in match.groups() if v and v[0] in "?$"]
        for var in vars_here:
            # Registered even when it is the only variable in its pattern.
            # `?person a edas:Person .` links nothing, but it does put ?person in
            # the branch — and a variable that never gets registered looks absent
            # rather than isolated, which is how the plainest cartesian product
            # (two typed variables, no relation between them) went undetected.
            find(var)
        for other in vars_here[1:]:
            union(vars_here[0], other)

    present = {v for v in selected if v in parent}
    groups: Dict[str, Set[str]] = {}
    for var in present:
        groups.setdefault(find(var), set()).add(var)
    return list(groups.values()) if len(groups) > 1 else []


def check_cartesian(query: str) -> List[str]:
    """Warnings for branches that pair unrelated things. Never edits the query.

    Reporting rather than repairing is deliberate: the fix depends on which
    relation is missing, and inventing one — or deleting a block — would replace
    a visible wrong answer with an invisible one. Rule 4 of the prompt asks the
    model for this; nothing verified it until now, and the failure is silent by
    nature, so it needs to be stated out loud somewhere.
    """
    clause = _SELECT_CLAUSE.search(query)
    if not clause:
        return []
    selected = {m.group(1) for m in _VAR.finditer(clause.group(1))}
    if len(selected) < 2:
        return []

    warnings = []
    for i, branch in enumerate(_union_branches(query), start=1):
        groups = _disconnected_selected(branch, selected)
        if groups:
            named = "  vs  ".join(
                "{" + ", ".join("?" + v for v in sorted(g)) + "}" for g in groups
            )
            warnings.append(f"branch {i}: {named} are never linked — every "
                            f"combination will be returned")
    return warnings


# A `PREFIX name: <namespace>` declaration, and a `name:LocalName` use of one.
# The use pattern requires a letter after the colon, so `http://` inside a URI and
# the `name: <ns>` of a declaration are both excluded.
_PREFIX_DECL = re.compile(r"^\s*PREFIX\s+([A-Za-z][\w.-]*)\s*:\s*<([^>]+)>", re.MULTILINE)
_PREFIX_USE = re.compile(r"(?<![<\w.-])([A-Za-z][\w.-]*):[A-Za-z_]")


def _repair_prefixes(query: str, prompt: str) -> str:
    """Make the query's prefixes agree with the prompt's, in both directions.

    ADDED when a prefix is used but never declared. The prompt states every
    prefix twice — the shared header above the schema sections, and again in the
    alignment section. The model reliably copies the first and forgets the
    second, so a source reaching the query only through a correspondence
    produces `<name>:Class` with no declaration: a parse error.

    REWRITTEN when a prefix is declared but points somewhere else. Measured on
    the single-source questions: the model wrote
    `PREFIX cmt: <http://example.org/cmt#>` — a plausible-looking namespace it
    invented rather than copied. Nothing about the query looks wrong, it parses,
    it runs, and every URI in it is a term that does not exist. This is worse
    than the missing case, which at least fails loudly.

    Repaired from the PROMPT's own declarations rather than a namespace table,
    so there is no second copy of the namespaces to drift out of sync.
    """
    known = {m.group(1): m.group(2) for m in _PREFIX_DECL.finditer(prompt)}
    declared = {m.group(1): m.group(2) for m in _PREFIX_DECL.finditer(query)}

    wrong = {p: ns for p, ns in declared.items()
             if p in known and ns != known[p]}
    if wrong:
        for p, ns in sorted(wrong.items()):
            print(f"[agent7] wrong namespace for {p}: <{ns}> -> <{known[p]}>")
        def _fix(m: re.Match) -> str:
            p = m.group(1)
            return (f"PREFIX {p}: <{known[p]}>" if p in wrong else m.group(0))
        query = _PREFIX_DECL.sub(_fix, query)

    missing = {p for p in (m.group(1) for m in _PREFIX_USE.finditer(query))
               if p not in declared and p in known}
    if missing:
        print(f"[agent7] added missing PREFIX declaration(s): {', '.join(sorted(missing))}")
        header = "\n".join(f"PREFIX {p}: <{known[p]}>" for p in sorted(missing))
        query = f"{header}\n{query}"

    return query


async def generate_sparql_query(question: str, summarized_paths: List[Dict[str, any]], base_source: str = None,
                                alignment_block: str = "", sources: Set[str] = None,
                                extracted: Dict[str, List[str]] = None,
                                provider_plan: List[Dict[str, any]] = None) -> str:
    messages = build_messages(question, summarized_paths, base_source,
                              alignment_block, sources, extracted, provider_plan)
    prompt = "\n".join(m["content"] for m in messages)

    # NOTHING TO BUILD FROM: refuse instead of asking.
    #
    # When Agent 6 keeps no path and no correspondence applies, the prompt
    # carries no ontology term at all — the PREFIX header is generated from the
    # terms in the schema sections, so no term means no PREFIX line.
    #
    # Asked anyway, the model does not say it cannot answer. Measured on "what
    # is the title of each paper?" it invented a whole schema — `confof:Paper`,
    # `cmt:title`, five namespaces under `http://example.org/` — and wrapped it
    # in the five correct endpoints. That query names the right ontologies,
    # parses, runs, and returns nothing, because not one term in it exists. It
    # is indistinguishable from a good answer until someone checks every URI.
    #
    # An empty result says the same thing honestly, and costs no model call.
    if not _PREFIX_DECL.search(prompt):
        print("[agent7] no ontology terms in the prompt - refusing to generate "
              "(nothing survived the filters; a query built from here would be invented)")
        return ""

    # The context window is shared by prompt + reasoning + answer. A reasoning
    # model needs a roughly FIXED amount to think, so once the prompt grows past
    # what is left the answer is never written and the agent gets "" — with no
    # sign of why. Printing the size makes that cliff visible before it is hit.
    print(f"[agent7] prompt ~{len(prompt)} chars (~{len(prompt) // 4} tokens)")

    try:
        raw = await run_agent_async("agent7_sparql_generator", messages=messages)
        content = _extract_query(raw)

        if not content:
            # Distinguish the two, because they need different fixes: an empty
            # reply is a model/serving problem (context exhausted, call failed),
            # while a non-empty reply with no query in it is a prompt problem.
            if raw and raw.strip():
                print(f"[agent7] ⚠ reply contained no SPARQL query. "
                      f"Raw: {raw.strip()[:200]!r}")
            else:
                print("[agent7] ⚠ model returned an empty query")
            return ""

        if len(content) < len(raw.strip()):
            print(f"[agent7] trimmed {len(raw.strip()) - len(content)} char(s) of "
                  f"non-query text from the reply")

        for warning in check_cartesian(content):
            print(f"[agent7] ⚠ cartesian product — {warning}")

        return _repair_prefixes(content, prompt)

    except Exception as e:
        # Includes ollama_client.EmptyCompletion, whose message names the cause
        # (context exhausted during reasoning vs. a genuinely empty answer).
        print(f"[agent7] failed to generate SPARQL: {e}")
        return ""

# Example usage:
# asyncio.run(generate_sparql_query(question, summarize_contextual_class_paths))
