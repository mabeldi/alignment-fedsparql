"""
Agent 8 — cross-source selection, ONE call for ALL ontologies.

WHY this shape
--------------
Agent 6 judges one ontology per call, and its prompt tells that call not to be
selective: an ontology covering one part of the question is kept even though it
cannot answer the rest, because the sources are combined afterwards. That is
right for a call that can only see its own source, and it leaves a gap — no call
in the pipeline ever compares two sources against each other.

The gap shows whenever the same concept is modelled in several ontologies, one
of them precisely and the others only as the general thing it is a case of. Each
per-ontology call keeps its own, correctly and in isolation; the query then reads
the general classes as well as the precise one, and answers a wider question than
the one asked.

So this step sees every surviving source at once, in a single prompt, and selects
the ones the query should read from.

IT SELECTS SOURCES, NOT PATHS
-----------------------------
Agent 6 already chose the paths inside each source, with the whole of that
source in front of it. Re-offering them here doubles the prompt and risks undoing
a decision made with more context, so a source is kept or dropped whole and its
blocks pass through untouched.

COMPLETE ANSWER BRANCHES, NOT ISOLATED WORD MATCHES
---------------------------------------------------
Several sources may model the same complete answer branch and hold different
instances, so all of those sources are kept. A source is not useful merely
because it contains one class whose name resembles part of the question: it must
either provide a complete branch for one whole question clause, or a uniquely
needed requested value/condition with a visible way to attach it to such a
branch.

Selection is BY NUMBER, the same contract as Agent 6: sources are numbered in the
prompt and the model returns those numbers. A name echoed back in the model's own
words cannot be matched to a source.

Failure is fail-OPEN throughout: a failed call, unreadable output or an empty
selection keeps every source. Emptying the query is a far worse outcome than not
narrowing it.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Agent 6's renderer and number reader, not second copies of them. The class
# entries and the relation catalog are the same views of a block that agent
# builds, and one number parser tolerating one set of model quirks is easier to
# keep true than two that drift apart.
from agents.agent6_contextual_path_filter import (_relation_numbers, _strip_fences,
                                                  class_entry)
from tools.alignment_store import Mapping, local_name
from tools.llm_router import run_agent_async
from tools.path_catalog import INCLUDE_ROUTES, PathCatalog

USE_CROSS_SOURCE_SELECTION = True

# When True, a non-empty relation provider plan decides the kept sources and
# the LLM vote is skipped. When False, the LLM vote always decides; the plan
# still reaches Agent 7 unchanged.
USE_PROVIDER_PLAN_FOR_SOURCE_SELECTION = False

# A path block as produced by agent5_path_summarizer.summarize_class_paths and
# filtered by agent6_contextual_path_filter.
PathBlock = Dict[str, Any]

# Keys that can hold the selected source numbers. `sources` is the documented
# one; the others are what a model writes when it names the list itself.
_SOURCE_KEYS = ("sources", "ontologies", "keep")

_PROVIDER_STOPWORDS = {
    "a", "an", "are", "at", "by", "for", "from", "has", "have", "in",
    "is", "of", "on", "the", "to", "with",
}


def _provider_tokens(text: str) -> Set[str]:
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return {
        token for token in re.sub(r"[^A-Za-z0-9]+", " ", split).lower().split()
        if token not in _PROVIDER_STOPWORDS
    }


def _candidate_connects_requested_classes(payload: Dict[str, Any],
                                           requested: Sequence[str]) -> bool:
    endpoints = list(payload.get("domains") or []) + list(payload.get("ranges") or [])
    endpoints += [local_name(uri) for uri in (
        list(payload.get("domainUris") or []) + list(payload.get("rangeUris") or []))]
    endpoint_tokens = [_provider_tokens(label) for label in endpoints]
    matched = {
        concept
        for concept in requested
        if (wanted := _provider_tokens(concept))
        and any(wanted <= endpoint for endpoint in endpoint_tokens)
    }
    return len(matched) >= 2


def _provider_sources_connected(sources: Set[str],
                                mappings: Sequence[Mapping]) -> bool:
    if len(sources) < 2:
        return True
    adjacency = {source: set() for source in sources}
    for mapping in mappings:
        left, right = mapping.entity1_source, mapping.target_source
        if left in sources and right in sources and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached = {next(iter(sources))}
    frontier = list(reached)
    while frontier:
        neighbours = adjacency[frontier.pop()] - reached
        reached |= neighbours
        frontier.extend(neighbours)
    return reached == sources


def build_relation_provider_plan(
        extracted: Dict[str, List[str]],
        context_matches: Dict[str, List[Dict[str, Any]]],
        blocks: Sequence[PathBlock],
        mappings: Sequence[Mapping]) -> List[Dict[str, Any]]:
    """Best validated concrete property for every requested resource relation."""
    requested_relations = extracted.get("objectProperties", [])
    if not requested_relations or extracted.get("dataTypeProperties"):
        return []

    sources = {block.get("source") for block in blocks if block.get("source")}
    chains_by_source = {
        source: "\n".join(
            chain
            for block in blocks if block.get("source") == source
            for side in ("outgoing", "incoming")
            for chain in block.get(side) or ()
        )
        for source in sources
    }

    plan: List[Dict[str, Any]] = []
    for relation in requested_relations:
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for match in context_matches.get(relation, []):
            payload = match.get("payload") or match
            if payload.get("type") != "ObjectProperty":
                continue
            source = payload.get("source")
            uri = payload.get("uri") or match.get("uri")
            if not source or source not in sources or not uri:
                continue
            if f"[{local_name(uri)}]" not in chains_by_source.get(source, ""):
                continue
            if not _candidate_connects_requested_classes(
                    payload, extracted.get("classes", [])):
                continue
            item = {
                "relation": relation,
                "source": source,
                "property": uri,
                "label": payload.get("label") or local_name(uri),
                "score": float(match.get("score") or 0.0),
            }
            candidates.append((item["score"], item))
        if not candidates:
            return []
        plan.append(max(candidates, key=lambda candidate: candidate[0])[1])

    provider_sources = {item["source"] for item in plan}
    if not _provider_sources_connected(provider_sources, mappings):
        return []
    return plan


def _group_by_source(blocks: Sequence[PathBlock]) -> Dict[str, List[PathBlock]]:
    """Blocks grouped under their source name, sorted so the numbering is stable."""
    grouped: Dict[str, List[PathBlock]] = {}
    for block in blocks:
        source = block.get("source")
        if source:
            grouped.setdefault(source, []).append(block)
    return {source: grouped[source] for source in sorted(grouped)}


def _render_source(blocks: Sequence[PathBlock]) -> str:
    """One source's classes and relations, indented under its heading.

    The relations come from `PathCatalog`, so they are stated once each and
    grouped by the class pair they join rather than restated in every walk that
    uses them.
    """
    chains = [chain
              for block in blocks
              for side in ("outgoing", "incoming")
              for chain in block.get(side) or []]

    lines: List[str] = []
    if any(block.get("source_complete_answer_branch") for block in blocks):
        lines.append("  complete answer branch: yes")
    else:
        lines.append("  complete answer branch: no")
    entries = [entry for entry in (class_entry(b) for b in blocks) if entry]
    if entries:
        lines.append("  classes:")
        lines += [f"    {entry}" for entry in entries]
    relations = PathCatalog(chains).render()
    if relations:
        lines += [f"  {line}" for line in relations.splitlines()]
    return "\n".join(lines) or "  (nothing)"


# GTX Render only alignments that can actually join two displayed candidates.
def _render_alignment_bridges(by_source: Dict[str, List[PathBlock]],
                              mappings: Sequence[Mapping]) -> str:
    source_numbers = {
        source: number
        for number, source in enumerate(by_source, start=1)
    }
    lines: List[str] = []
    seen: Set[Tuple[str, str, str]] = set()
    for mapping in mappings:
        left = mapping.entity1_source
        right = mapping.target_source
        if left == right or left not in source_numbers or right not in source_numbers:
            continue
        key = (left, right, mapping.render())
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- [{source_numbers[left]}] {left} <-> "
            f"[{source_numbers[right]}] {right}: {mapping.render()}"
        )
    return "\n".join(lines) or "- (no alignment connects the displayed sources)"


def build_source_selection_prompt(question: str,
                                  by_source: Dict[str, List[PathBlock]],
                                  extracted: Optional[Dict[str, List[str]]] = None,
                                  mappings: Sequence[Mapping] = (),
                                  ) -> List[Dict[str, str]]:
    """The single prompt holding every source, numbered.

    Deliberately contains NO ontology vocabulary of its own — the notation is
    explained with placeholder names, so this prompt is identical whichever
    ontologies happen to be loaded.
    """
    sections = "\n".join(
        f"[{i}] {source}\n{_render_source(blocks)}"
        for i, (source, blocks) in enumerate(by_source.items(), start=1)
    )
    extracted = extracted or {}
    resource_concepts = ", ".join(extracted.get("classes", [])) or "(none)"
    relationships = ", ".join(extracted.get("objectProperties", [])) or "(none)"
    literal_values = ", ".join(extracted.get("dataTypeProperties", [])) or "(none)"
    requested_shape = (
        "Requested semantic shape:\n"
        f"- resource concepts: {resource_concepts}\n"
        f"- relationships between resources: {relationships}\n"
        f"- literal values: {literal_values}"
    )
    # GTX Agent 8 must see the bridges before it decides whether a partial
    # GTX source can join a complementary source into one answer branch.
    alignment_bridges = _render_alignment_bridges(by_source, mappings)

    # Only described when the catalog actually renders that block: explaining a
    # section the data does not contain is noise the model has to discard.
    routes_notation = (
        "- A `routes:` line is a walk of several hops. Its heading names the classes\n"
        "  in order, and `1>7` means the first relation then the seventh, joined on\n"
        "  the class they share.\n"
    ) if INCLUDE_ROUTES else ""

    system_prompt = (
        "You are an expert ontology assistant.\n"
        "You are given several ontologies, each already filtered on its own, and "
        "a user's question. They are queried together and their answers are "
        "combined, so keep every ontology that provides a complete answer branch "
        "or a uniquely necessary requested contribution. Do not keep an ontology "
        "for an isolated word or class match. You decide which ones the query "
        "reads from.\n"
        "You answer with a JSON object of selected numbers and nothing else."
    )

    user_prompt = f"""
Question:
"{question}"

{requested_shape}

Notation (placeholder names, not real ones):
- `ClassA — attributes: attrOne, attrTwo` is a class and the values it holds.
- `ClassA>ClassB: [1] relOne  [2] relTwo` numbers two relations that both join
  those same two classes. Relation 1 is the triple (ClassA relOne ClassB).
- `[3] <relThree` on that same line runs the other way: the triple is
  (ClassB relThree ClassA).
{routes_notation}- The numbers inside a section number that section's relations and start again
  in every section. They are not what you select.

Sources, numbered:
{sections}

Cross-source class alignments:
{alignment_bridges}

You select the numbered SOURCES the query should read from.

A COMPLETE ANSWER BRANCH contains the thing or value being requested, every
condition that restricts it in one question clause, and the relations connecting
those concepts. For example, a question asking for abstracts belonging to
invited talks needs an Abstract/InvitedTalk connection; a source containing only
Abstract-to-Reviewer does not provide that branch.

Rules, in order:
0. Respect the requested semantic kind. A requested RESOURCE must be supplied
   by a class instance; a literal attribute cannot substitute for it even when
   both have the same label. A requested LITERAL must be supplied by an
   attribute value. For example, when Abstract is a resource concept,
   `Paper -> Abstract` qualifies while a string-valued `Paper.abstract` does not.
1. Keep every source that provides a complete answer branch for at least one
   whole clause of the question. If several sources provide that same complete
   branch, keep all of them because they may hold different instances.
2. Keep complementary sources when their displayed class alignment joins their
   partial branches into a complete answer. For example, if source A provides
   `Document -> Summary`, source B provides `Document -> Topic`, and their
   Document classes are aligned, keep A and B for a question asking for topics
   of documents that have summaries.
3. Prefer a source that directly expresses a requested relationship between the
   requested endpoint concepts. An indirect route through contextual classes is
   not a substitute. For `Document -> Topic`, `EventPart -> Topic` connected to
   Document only through Event does not replace a direct Document-to-Topic path.
4. A partial source may be kept only when it supplies a UNIQUE value, condition,
   or relationship explicitly requested by the question, another kept source
   does not supply it directly, and a displayed alignment shows how it attaches
   to the same answer entity. All three requirements must hold.
5. Drop a source that offers only an isolated matching class, a broader related
   concept, or relations that do not express the question's requested condition.
   Sharing words with the question is not evidence of a complete branch.
6. Judge by MEANING, not wording. Different names can express the same branch,
   while similar names can express different branches.
7. Select all complete branches, but do not keep a source "just in case".
8. Sources containing equivalent answer classes may hold different instances. Keep every source whose branch satisfies the question, including through the supplied alignments. Equivalent classes with different names are not incomplete or redundant merely because another source expresses the concept more directly.
9. ALIGNMENT: check each correspondence one by one; use every relevant one to build the answer patterns across allowed sources, keeping its conditions. A correspondence = the SAME concept at different endpoints, with different instances. Copy its triples as written. A CONDITIONAL one maps a class to a class expression: judge it by the meaning of the WHOLE expression, all its triples together, never by its general class alone. If that meaning matches the answer, the expression IS the answer class in its source, even if the schema shows only the general class: that source holds answer rows and gets its own branch, built from all the expression's triples; dropping any changes the answer. A source appearing only in ALIGNMENT (no SCHEMA section) still belongs in the query; those triples are its only vocabulary.


Output format:
- A strict JSON object with one array, e.g. {{"sources": [1, 4]}}
- `sources` holds the numbers in square brackets at the source headings.
- The array must not be empty.
- No text, no explanation, no markdown, no code fences.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_selection(raw: str, count: int) -> Optional[Set[int]]:
    """Model output -> the 1-based source numbers it names, or None if unreadable.

    None (unreadable) and an empty set (the model named nothing) both make the
    caller keep every source, but they are kept apart so the log says which
    happened.

    A bare array is accepted as the list itself. That shape carries exactly the
    same information, and rejecting it would drop a usable answer over a missing
    pair of braces.
    """
    text = _strip_fences(raw)
    if not text.lstrip().startswith(("{", "[")):
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        return _relation_numbers(data, count)
    if isinstance(data, dict):
        for key in _SOURCE_KEYS:
            if key in data:
                # Numbers outside 1..count name no source and are discarded here.
                return _relation_numbers(data.get(key), count)
    return None


async def select_sources(question: str, blocks: List[PathBlock],
                         extracted: Optional[Dict[str, List[str]]] = None,
                         mappings: Sequence[Mapping] = ()) -> Set[str]:
    """The sources worth querying, chosen in one call over all of them at once.

    Returns the source names to keep. Every failure mode — no call, a failed
    call, unreadable output, an empty selection — returns every source present in
    `blocks`, so this step can under-select but can never empty the query.
    """
    by_source = _group_by_source(blocks)
    sources = list(by_source)
    if not sources:
        return set()

    everything = set(sources)
    if not USE_CROSS_SOURCE_SELECTION:
        return everything
    # With one source there is nothing to compare it against, and fail-open makes
    # every possible answer keep it, so the call cannot change the result.
    if len(sources) == 1:
        return everything

    messages = build_source_selection_prompt(
        question, by_source, extracted, mappings)
    try:
        raw = await run_agent_async("agent8_source_selector", messages=messages,
                                    json_mode=True)
    except Exception as exc:
        print(f"[agent8] LLM call failed ({exc}) — keeping all {len(sources)} source(s)")
        return everything

    numbers = _parse_selection(raw, len(sources))
    if numbers is None:
        print(f"[agent8] could not parse selection — keeping all {len(sources)} "
              f"source(s). Raw: {raw[:200]!r}")
        return everything
    if not numbers:
        print(f"[agent8] empty selection — keeping all {len(sources)} source(s)")
        return everything

    kept = {sources[n - 1] for n in numbers}
    dropped = everything - kept
    print(f"[agent8] kept {len(kept)}/{len(sources)} source(s): "
          f"{', '.join(sorted(kept))}"
          + (f" — dropped {', '.join(sorted(dropped))}" if dropped else ""))
    return kept


async def apply_source_selection(question: str, blocks: List[PathBlock],
                                 extracted: Optional[Dict[str, List[str]]] = None,
                                 mappings: Sequence[Mapping] = (),
                                 provider_plan: Sequence[Dict[str, Any]] = (),
                                 ) -> Tuple[List[PathBlock], Set[str]]:
    """`blocks` with the unselected sources removed, plus the names of those sources.

    The dropped names are returned because emptying a source is not enough to
    keep it out of the query: the alignment names its classes from the sources
    that stayed, and only a source reported as cut is excluded whatever the
    alignment says.
    """
    if not USE_CROSS_SOURCE_SELECTION or not blocks:
        return blocks, set()

    if provider_plan and USE_PROVIDER_PLAN_FOR_SOURCE_SELECTION:
        # GTX A validated, alignment-connected provider for every requested
        # GTX relation is stronger evidence than another flat LLM source vote.
        kept = {item["source"] for item in provider_plan}
        dropped = {block.get("source") for block in blocks
                   if block.get("source") and block.get("source") not in kept}
        print("[agent8] deterministic relation providers: "
              + ", ".join(f"{item['relation']} -> {item['source']}:{item['label']}"
                          for item in provider_plan))
        return [block for block in blocks if block.get("source") in kept], dropped

    kept = await select_sources(question, blocks, extracted, mappings)
    dropped = {source for block in blocks
               if (source := block.get("source")) and source not in kept}
    if not dropped:
        return blocks, set()
    return [b for b in blocks if b.get("source") in kept], dropped
