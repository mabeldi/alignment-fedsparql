"""
Agent 6 — contextual path filter, ONE call per ONTOLOGY.

WHY this shape
--------------
The previous version made one LLM call per (class, direction) bucket. On the
cmt run that was 8 independent calls, each seeing only its own slice, and the
result was that filtering barely did anything:

    Agent 5: 20 chains / 10 distinct edges
    Agent 6: 13 chains /  8 distinct edges     <- only markConflictOfInterest died

Two causes, both structural:

1. UNION CANCELS DROPS. The same relation appears in several buckets
   (`Author -[submitPaper]-> PaperFullVersion` is in Author[outgoing] AND in
   PaperFullVersion[incoming]). `prompt_format.collect_edges` merges every
   bucket into one deduplicated edge set, so an edge survives if ANY single
   bucket kept it. Author[outgoing] dropped that chain and it still reached
   Agent 7 through the other bucket.
2. NO CALL COULD SEE REDUNDANCY. `hasAuthor` and `writePaper` are the same link
   written in opposite directions, and `Paper`/`PaperFullVersion` repeat the
   same relations because one is a subclass of the other. A bucket holding only
   one side of the pair cannot notice that the other side already covers it.

So the unit of filtering is now the ONTOLOGY: every chain of one source is
shown in a single prompt, the model picks the subset that answers the question,
and an ontology that contributes nothing is dropped whole (its classes then
leave `entry_classes`, the CLASSES block and `plan_federation` too).

Everything selectable is selected BY NUMBER, not echoed back as text. The arrow
notation (`A -[p]-> B` vs `A <-[p]- B`) is direction-carrying and
`tools/path_sparql._parse_chain` drops any chain whose arrows were reworded, so
a model that "helpfully" rewrote a chain used to delete it silently. An integer
cannot be reworded. (The prompt no longer shows that notation at all — see ONE
RELATION, ONE LINE below — but the rule still decides what may be echoed.)

Output shape is unchanged (list of {class, source, label, outgoing, incoming}),
so Agent 7, `plan_federation` and `output_writer` consume it as before.

ONE ONTOLOGY, PART OF A FEDERATION
----------------------------------
Filtering per ontology has a failure mode of its own: the model sees one source
and the whole question, and drops the source entirely when it cannot answer all
of it. Measured on "decisions of papers along with their authors": the source
holding the authorship relation but no decision relation returned [], so its 8
chains died and the query lost that half of the answer.

Nothing was wrong with the source — the prompt just never said the sources are
combined afterwards. It does now, and the empty-array rule is scoped to "nothing
relates to ANY part of the question" instead of "cannot answer the question".

SOME ANSWERS ARE A CLASS, NOT A PATH
------------------------------------
Filtering only chains loses every question whose answer is plain class
membership. Measured on "which papers were accepted?": two sources hold a class
that IS the answer, both were skipped whole because no RELATION in them was
relevant, and the query lost them. The chains were correctly judged useless —
the class was never up for selection.

So the model now returns two lists: the paths worth keeping, and the classes
that answer the question by themselves. A class in the second list survives with
no chains at all, and a source is only skipped when both lists come back empty.

A CLASS IS ALSO ITS ATTRIBUTES
------------------------------
The class list was shown as bare names, so a class whose ATTRIBUTES answer the
question looked worthless. Measured on "what are the surnames of the people who
ran a workshop": the source holding the gold surname property offered `Person`
with no relevant relation, the rule read "membership alone must be the answer",
being a person is not a surname — so it was dropped, and the attribute inside it
died with it. Every other source lost its person class the same way; the one
that survived did so by accident, took the anchor with it, and the query
answered from the wrong ontology.

Retrieval had done its job: the gold property was retrieved and kept by Agent 4.
It was invisible here purely because this prompt never rendered it. It does now,
and a class may be kept for the values it carries as well as for its membership.

CARRYING A VALUE IS NOT BEING THE ANSWER
----------------------------------------
Those two reasons first shared one list, and that list sets `answers_question`.
Every ontology has a person class with a name attribute, so on the surname
question all nine blocks were kept and all nine were flagged as the answer. The
flag is read by the anchor and printed to Agent 7 as "build the query around
this"; pointed at everything it says nothing, and Agent 7 wrote one branch per
ontology. Class F1 fell from 0.286 to 0.200 while recall rose — the class was
kept, and the reason recorded was wrong.

So the two reasons are two lists. CLASSES is the answer and keeps the flag;
CARRIERS survives with the flag off. A carrier is also dropped when a kept
answer class is a narrower kind of it and inherits the same attributes:
`Student` already carries everything `Person` does, so keeping both asks the
question twice, once without the condition that made it the question.

What this does NOT fix: the same value existing in five ontologies at once.
Choosing between them means comparing sources, and this agent judges one source
at a time — no call here can see the other four.

ONE RELATION, ONE LINE
----------------------
The chains were offered as one flattened line each, which restates a relation in
every walk that uses it. Over the 43 saved runs that is 11,482 lines describing
5,947 relations, and the multiplication is a cross product: each hop of a walk
offers several predicates and every combination becomes its own line, so one
class pair in CJ-01 produced 89 equally short chains out of 38 relations.

`tools/path_catalog` states each relation once instead, numbered and grouped by
the class pair it joins, with multi-hop walks written as sequences of those
numbers. The selection unit is therefore a RELATION, not a whole chain — which is
also what `with_prefixes` was working around: 3,761 of those 5,947 relations had
no single-hop chain of their own, so two thirds of them could only be kept with
something else attached. `PathCatalog.resolve` turns the selection back into the
chain strings this agent has always returned, so nothing downstream changes.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tools.alignment_store import local_name
from tools.llm_router import run_agent_async
from tools.path_catalog import INCLUDE_ROUTES, PathCatalog, chain_endpoints

# A path block as produced by agent5_path_summarizer.summarize_class_paths.
PathBlock = Dict[str, Any]

# Where a distinct chain came from: (index of its block, "outgoing" | "incoming").
Origin = Tuple[int, str]

# One ontology's answer: (relation numbers, route tokens, answer classes,
# carrier classes). The first two are read against a `PathCatalog`, which turns
# them back into the chain strings the rest of the pipeline consumes.
Selection = Tuple[Set[int], List[str], Set[int], Set[int]]

# Attributes listed per class. Enough to judge relevance, short enough that a
# class with dozens of them cannot crowd the relation paths out of the prompt.
_MAX_VALUES_SHOWN = 12

# Splits a class entry's name from its attribute list.
_ENTRY_SEP = " — "


def class_entry(block: PathBlock) -> str:
    """One line describing a class for selection: its name and its attributes.

    A bare name gives the model nothing to weigh a class on except membership,
    which is why a class holding the very property the question asks to SEE used
    to read as irrelevant. The attribute names are the ontology's own, same as
    the labels and the chains — nothing about this template is ontology-specific.
    """
    label = block.get("label") or local_name(block.get("class") or "") or ""
    names: List[str] = []
    for edge in block.get("values", ()):
        if not edge or len(edge) < 2:
            continue
        name = local_name(edge[1])
        if name and name not in names:
            names.append(name)

    if not names:
        return label
    shown = ", ".join(names[:_MAX_VALUES_SHOWN])
    if len(names) > _MAX_VALUES_SHOWN:
        shown += f", … (+{len(names) - _MAX_VALUES_SHOWN} more)"
    return f"{label}{_ENTRY_SEP}attributes: {shown}"


def _value_props(block: PathBlock) -> Set[str]:
    """The attribute URIs readable off this block's class."""
    return {e[1] for e in block.get("values", ()) if e and len(e) > 1}


def _covered_by_answer(block: PathBlock, answer_blocks: Sequence[PathBlock]) -> bool:
    """True when a kept ANSWER class already provides everything this class holds.

    A class inherits its parents' attributes, so `Student` carries every value
    `Person` does. Once `Student` is kept as the answer, keeping `Person` too
    adds a second, wider branch that reads the same values off a bigger set of
    things — the query then asks its question twice, once correctly and once
    without the condition that made it the question.

    Both tests must hold: the answer must be a NARROWER kind of this class (so
    it really is the same things, minus some), and it must actually carry the
    attributes (a subclass whose parent's attributes did not reach it proves
    nothing). Rule 9 states this, this enforces it — the pruning has to be
    reliable, and the model is judging by names.
    """
    from core.ontology_loader import get_graph
    from tools.find_class_paths import query_ancestors

    uri, source = block.get("class"), block.get("source")
    if not uri or not source:
        return False
    mine = _value_props(block)
    if not mine:
        return False

    try:
        graph = get_graph().get_context("http://" + source)
    except Exception:
        return False

    for answer in answer_blocks:
        other = answer.get("class")
        if not other or other == uri:
            continue
        if uri in query_ancestors(other, graph) and mine <= _value_props(answer):
            return True
    return False


def build_ontology_filter_prompt(source: str,
                                 question: str,
                                 catalog: PathCatalog,
                                 class_entries: Sequence[str]) -> List[Dict[str, str]]:
    """The single prompt for one ontology: its relations, stated once and numbered.

    Deliberately contains NO ontology vocabulary of its own — the notation is
    explained with placeholder names, so this prompt is identical whichever
    ontologies happen to be loaded.
    """
    numbered_classes = "\n".join(f"[{i}] {c}" for i, c in enumerate(class_entries, start=1)) \
        or "(none)"
    relations = catalog.render() or "(none)"

    # The routes half of the notation, the ROUTES selection rules and the ROUTES
    # output key all appear together or not at all: describing a section the
    # catalog does not render is an invitation to select from it anyway.
    routes_notation = (
        "- A `routes:` line is a walk of several hops. Its heading names the classes\n"
        "  in order, and `1>7` means relation 1 then relation 7, joined on the class\n"
        "  they share.\n"
    ) if INCLUDE_ROUTES else ""

    # Bullets, not numbers: the rules below are referred to by number from the
    # code that enforces them (see `_covered_by_answer`, rule 9), so an optional
    # section must not shift them.
    routes_rules = ("""
ROUTES — the multi-hop walks to keep, written exactly as shown (for example
"1>7"). Rules:
- Keep a route only when the question needs those hops JOINED — when it asks
  about one thing through another. If the hops are useful on their own, keep
  them in RELATIONS instead and leave ROUTES empty.
- A route does not need its relations selected as well; keeping the route keeps
  every hop it names.
""") if INCLUDE_ROUTES else ""

    example = ('{"relations": [2, 5], "routes": ["1>7"], "classes": [1], "carriers": [3]}'
               if INCLUDE_ROUTES else
               '{"relations": [2, 5], "classes": [1], "carriers": [3]}')

    system_prompt = (
        "You are an expert ontology assistant.\n"
        "You are given ONE ontology's classes and relations, and a user's "
        "question. Several ontologies are being queried together, each judged "
        "separately, and their answers are combined afterwards. This one does not "
        "have to answer the question on its own — you select whatever part of the "
        "question it DOES cover, and the smallest set that covers it.\n"
        "You answer with a JSON object of selected numbers and nothing else."
    )

    user_prompt = f"""
Question:
"{question}"

Ontology: {source}

Notation (placeholder names, not real ones):
- `ClassA>ClassB: [1] relOne  [2] relTwo` numbers two relations that both join
  those same two classes. Relation 1 is the triple (ClassA relOne ClassB).
- `[3] <relThree` on that same line runs the other way: the triple is
  (ClassB relThree ClassA). Both directions share one heading because they
  connect the same two classes.
{routes_notation}
Classes found in this ontology (with the attributes each one carries):
{numbered_classes}

Relations found in this ontology, each stated once:
{relations}

The question may have several parts. This ontology is one of several being
queried, so treat each part on its own: cover the parts this ontology has, and
ignore the parts it does not — another ontology supplies those.

You select {"FOUR" if INCLUDE_ROUTES else "THREE"} things.

RELATIONS — the numbered relations to keep. Rules, in order:
1. Keep a relation only if it carries information the question actually asks
   for. A relation that is merely "about the same topic" is not enough. One that
   covers ONE part of the question qualifies; it does not have to cover all.
2. Keep the SMALLEST sufficient set. Anything the query can be built without
   must be left out.
3. Remove duplicates of a connection you already kept:
   - everything on one heading joins the same two classes, so several numbers
     from one heading are one connection said several ways — keep one, unless
     they genuinely mean different things;
   - when a relation uses a specialised class and another gives the same
     relation on its more general class, keep the general one. Keep the
     specialised one only if the question explicitly asks for that subgroup.
{routes_rules}

CLASSES — the classes that ARE the answer. Rules:
4. Keep a class here when simply belonging to it is what the question asks for,
   so no relation is needed to answer. Membership alone must be the answer.
5. Do NOT keep a class here merely because a kept relation touches it; a
   relation already carries the two classes it joins.
6. Prefer the class that states the asked-for condition directly over a broader
   one that would also match things the question excludes.

CARRIERS — the classes that HOLD a value the question needs, without being the
answer themselves. Rules:
7. Keep a class here when one of its attributes is a value the question asks to
   SEE or to test. Asking for someone's name is not asking for people: the
   class is where the value is stored, not the thing being asked for.
8. Judge attributes by MEANING, not by wording. The question and the attribute
   may use completely different words for the same thing.
9. A class inherits every attribute of the classes above it, so a narrower class
   already carries the wider one's attributes. Do NOT keep a class here when a
   class you kept in CLASSES is a narrower kind of it and covers the same
   attribute — the narrower one already provides that value, and keeping both
   asks the same question twice.
10. A class already kept in CLASSES does not belong here as well.

Return EVERY list empty ONLY if nothing here relates to ANY part of the
question. The whole ontology is then skipped, so do not keep anything "just in
case" — but "this ontology cannot answer the whole question" is NOT a reason to
empty it. If it covers even one part, keep what covers that part.

Output format:
- A strict JSON object with these arrays, e.g.
  {example}
- `relations`, `classes` and `carriers` hold the numbers in square brackets.
- Use an empty array for any list where nothing is relevant.
- No text, no explanation, no markdown, no code fences.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences the model may add despite instructions."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def _indices(data: Any, items: Sequence[str]) -> Set[int]:
    """One JSON list -> 0-based indices into `items`. Class entries only.

    Two accepted element shapes, because a small model asked for numbers
    sometimes returns the text anyway:
      * integers (the documented contract), and
      * strings matched EXACTLY against the input — a reworded entry is no
        longer the one we offered, so it is not credited to any index.

    A class entry carries its attributes after `_ENTRY_SEP`, so the bare name is
    accepted too: naming the class is unambiguous, and asking the model to echo
    a long attribute list back verbatim would fail this match for no reason.

    Relations go through `_relation_numbers` instead, which takes numbers only —
    a relation's NAME is not unique (the same property can join two class pairs)
    so matching one back to a single number would be a guess.
    """
    if not isinstance(data, list):
        return set()
    by_text = {c: i for i, c in enumerate(items)}
    for i, c in enumerate(items):
        head = c.split(_ENTRY_SEP)[0].strip()
        if head and head not in by_text:
            by_text[head] = i
    kept: Set[int] = set()
    for item in data:
        if isinstance(item, bool):                 # bool is an int in Python
            continue
        if isinstance(item, int):
            if 1 <= item <= len(items):            # 1-based, as numbered above
                kept.add(item - 1)
            continue
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.isdigit():
                n = int(stripped)
                if 1 <= n <= len(items):
                    kept.add(n - 1)
            elif stripped in by_text:
                kept.add(by_text[stripped])
    return kept


# Keys that have meant "the relations to keep" at some point. `paths` is the old
# contract, when the numbers indexed whole chains; under the new numbering they
# index relations, and a model still writing the old key gets read, not dropped.
_RELATION_KEYS = ("relations", "edges", "paths")


def _relation_numbers(data: Any, count: int) -> Set[int]:
    """One JSON list -> the 1-based relation numbers it names.

    Numbers only, unlike `_indices`. A relation's name is not unique — the same
    property can join two different class pairs, and two headings can carry the
    same predicate — so accepting the text back would have to guess which one was
    meant. `[3]` and `" 3 "` are still the number 3: a model asked for a compact
    id routinely returns it wearing its brackets.
    """
    if not isinstance(data, list):
        return set()
    kept: Set[int] = set()
    for item in data:
        if isinstance(item, bool):                     # bool is an int in Python
            continue
        if isinstance(item, int):
            n = item
        elif isinstance(item, str):
            digits = item.strip().strip("[]").strip()
            if not digits.isdigit():
                continue
            n = int(digits)
        else:
            continue
        if 1 <= n <= count:
            kept.add(n)
    return kept


def _route_tokens(data: Any, catalog: PathCatalog) -> List[str]:
    """One JSON list -> the route tokens it names, unknown ones discarded.

    Two accepted shapes. A string is the token as rendered (`"1>7"`, spacing and
    brackets ignored). A list of numbers is the same walk written the way a model
    that read "arrays of numbers" writes it — cheap to accept, and the
    alternative is silently losing the walk.
    """
    if not isinstance(data, list):
        return []
    kept: List[str] = []
    for item in data:
        if isinstance(item, list):
            item = ">".join(str(x) for x in item if isinstance(x, int)
                            and not isinstance(x, bool))
        if not isinstance(item, str):
            continue
        token = catalog.normalise_token(item)
        if catalog.is_route(token) and token not in kept:
            kept.append(token)
    return kept


def _parse_selection(raw: str,
                     catalog: PathCatalog,
                     class_entries: Sequence[str]) -> Optional[Selection]:
    """Model output -> (relations, routes, answer classes, carriers), or None.

    None (unreadable) and an all-empty selection (explicitly nothing relevant)
    mean different things to the caller: an empty selection deletes a whole
    ontology, so a parse failure must never be mistaken for one.

    A bare array is still accepted and read as relation numbers only. That is the
    old contract, and a model that ignores the object format would otherwise have
    its whole ontology deleted for a formatting slip.
    """
    text = _strip_fences(raw)
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        return _relation_numbers(data, catalog.edge_count), [], set(), set()
    if isinstance(data, dict):
        answers = _indices(data.get("classes"), class_entries)
        # Rule 10 enforced here rather than trusted: a class named in both lists
        # is an answer, and letting it also count as a carrier would make the
        # carrier list look bigger than the pruning below actually left.
        carriers = _indices(data.get("carriers"), class_entries) - answers
        relations: Set[int] = set()
        for key in _RELATION_KEYS:
            relations |= _relation_numbers(data.get(key), catalog.edge_count)
        return relations, _route_tokens(data.get("routes"), catalog), answers, carriers
    return None


def _catalog(blocks: List[PathBlock],
             indices: Sequence[int]) -> Tuple[List[str], Dict[str, List[Origin]]]:
    """One ontology's blocks -> (distinct chains in stable order, where each came from).

    The same chain routinely sits in several buckets, which is exactly what made
    per-bucket filtering ineffective. Here it is offered to the model ONCE, and
    a decision about it applies to every bucket it came from.
    """
    chains: List[str] = []
    origins: Dict[str, List[Origin]] = {}
    for i in indices:
        for direction in ("outgoing", "incoming"):
            for chain in blocks[i].get(direction) or []:
                if chain not in origins:
                    origins[chain] = []
                    chains.append(chain)
                origins[chain].append((i, direction))
    return chains, origins


def _placement(chain: str,
               by_label: Dict[str, int],
               catalog: PathCatalog,
               origins: Dict[str, List[Origin]]) -> List[Origin]:
    """Which class blocks a chain belongs to.

    Needed because most resolved relations have no recorded origin: a relation
    the model kept on its own is rendered as a one-hop chain, and two thirds of
    them (3,761 of 5,947 over the saved runs) never existed as a chain in Agent
    5's output — they only ever appeared inside a longer walk.

    Two rules, in order. First Agent 3's own: a chain is outgoing for the class
    it starts at and incoming for the class it ends at. That needs those classes
    to have a block, which is not guaranteed, so second: the blocks of the walks
    this relation came from. A relation that lands nowhere is a relation dropped,
    and dropping one here would undo the whole point of numbering them.
    """
    ends = chain_endpoints(chain)
    places: List[Origin] = []
    if ends is not None:
        start, end = ends
        if start in by_label:
            places.append((by_label[start], "outgoing"))
        if end in by_label and by_label[end] != by_label.get(start):
            places.append((by_label[end], "incoming"))
    if places:
        return places
    for parent in catalog.containing_chains(chain):
        for place in origins.get(parent, ()):
            if place not in places:
                places.append(place)
    return places


async def filter_source_paths_with_llm(source: str,
                                       question: str,
                                       catalog: PathCatalog,
                                       class_entries: Sequence[str]
                                       ) -> Selection:
    """One ontology's selection, retried once and dropped after two failures.

    Structured JSON mode prevents most formatting errors. A source whose call
    still fails or whose response is still unreadable after the retry is failed
    closed: four empty selections make the caller drop that source instead of
    forwarding every unreviewed path.
    """
    if not catalog.has_content() and not class_entries:
        return set(), [], set(), set()

    messages = build_ontology_filter_prompt(source, question, catalog, class_entries)
    for attempt in range(1, 3):
        try:
            raw = await run_agent_async(
                "agent6_path_filter", messages=messages, json_mode=True
            )
        except Exception as exc:
            if attempt == 1:
                print(f"[agent6] {source}: LLM call failed ({exc}) — retrying once")
            else:
                print(f"[agent6] {source}: LLM call failed again ({exc}) — "
                      "dropping source")
            continue

        kept = _parse_selection(raw, catalog, class_entries)
        if kept is not None:
            return kept

        if attempt == 1:
            print(f"[agent6] {source}: could not parse selection — retrying once. "
                  f"Raw: {raw[:200]!r}")
        else:
            print(f"[agent6] {source}: could not parse selection again — "
                  f"dropping source. Raw: {raw[:200]!r}")

    return set(), [], set(), set()


async def filter_class_paths_with_llm(question: str,
                                      summarized_class_paths: List[PathBlock]) -> List[PathBlock]:
    """Filter the summarized paths, one LLM call per ontology.

    An ontology the model finds irrelevant is removed ENTIRELY — all of its
    blocks go, so its classes disappear from the Agent 7 payload and from
    federation planning rather than lingering as unusable variables.

    Each ontology call requests structured JSON and is retried once. If both
    attempts fail or remain unparseable, that ontology is dropped rather than
    forwarding paths that received no relevance decision.
    """
    by_source: Dict[str, List[int]] = {}
    for i, block in enumerate(summarized_class_paths):
        source = block.get("source")
        if source:
            by_source.setdefault(source, []).append(i)

    sources = list(by_source)
    catalogs = [_catalog(summarized_class_paths, by_source[s]) for s in sources]
    # The flat chain list is still what `origins` is keyed on, so it is built
    # first and the factored view is derived from it — the relations the model is
    # shown are exactly the relations those chains described.
    factored = [PathCatalog(chains) for chains, _ in catalogs]

    selections = await asyncio.gather(*[
        filter_source_paths_with_llm(
            source,
            question,
            catalog,
            [class_entry(summarized_class_paths[i]) for i in by_source[source]],
        )
        for source, catalog in zip(sources, factored)
    ])

    result: List[PathBlock] = []
    for source, (_, origins), catalog, selection in zip(
            sources, catalogs, factored, selections):
        indices = by_source[source]

        kept_relations, kept_routes, kept_classes, kept_carriers = selection
        kept = catalog.resolve(kept_relations, kept_routes)
        if not kept and not kept_classes and not kept_carriers:
            print(f"[agent6] {source}: nothing relevant — ontology skipped "
                  f"({catalog.edge_count} relation(s), {len(indices)} class "
                  f"block(s) dropped)")
            continue

        # Put each kept chain back into every bucket it came from, so the block
        # structure downstream is exactly the one Agent 5 produced. A relation
        # kept on its own has no recorded bucket — it is a chain Agent 5 never
        # wrote — so its two ends say where it goes instead.
        rebuilt: Dict[int, Dict[str, List[str]]] = {
            i: {"outgoing": [], "incoming": []} for i in indices
        }
        by_label: Dict[str, int] = {}
        for i in indices:
            by_label.setdefault(summarized_class_paths[i].get("label") or "", i)
        lost = 0
        for chain in kept:
            places = origins.get(chain) or _placement(chain, by_label, catalog, origins)
            if not places:
                lost += 1
                continue
            for block_index, direction in places:
                rebuilt[block_index][direction].append(chain)
        if lost:
            print(f"[agent6] ⚠ {source}: {lost} kept chain(s) had no class block to "
                  f"file them under and were dropped")

        # Positions in `indices` the model named as answers in their own right,
        # and as holders of a needed value. Only the first sets the flag: a class
        # is not what the question asks for merely because the answer is stored
        # on it, and a flag every block carries tells Agent 7 nothing.
        answers = {indices[p] for p in kept_classes if p < len(indices)}
        carriers = {indices[p] for p in kept_carriers if p < len(indices)}

        # Rule 9, enforced: drop a carrier whose attributes a kept answer class
        # already inherits. Both would read the same values, the wider one
        # without the condition that made the question a question.
        answer_blocks = [summarized_class_paths[i] for i in answers]
        redundant = {i for i in carriers
                     if _covered_by_answer(summarized_class_paths[i], answer_blocks)}
        if redundant:
            print(f"[agent6] {source}: {len(redundant)} carrier(s) dropped — a kept "
                  f"answer class already inherits their attributes")
        carriers -= redundant

        dropped_blocks = 0
        for i in indices:
            block = summarized_class_paths[i]
            outgoing = rebuilt[i]["outgoing"]
            incoming = rebuilt[i]["incoming"]
            # A class left with no path normally cannot be joined to anything, and
            # keeping it just adds an unconstrained variable to Agent 7's payload —
            # that is how a query turns into a cartesian product. Answers and
            # carriers are the exceptions: `?x a C` is already the whole answer,
            # and `?x p ?value` reads the value straight off the class.
            if not outgoing and not incoming and i not in answers and i not in carriers:
                dropped_blocks += 1
                continue
            result.append({
                "class": block["class"],
                "source": block["source"],
                "label": block["label"],
                "outgoing": outgoing,
                "incoming": incoming,
                # NOT filtered, deliberately. This agent's job is to prune the
                # class graph — it decides which walks between classes are worth
                # keeping, and a value property is not a walk. It is also what the
                # question usually asks to SEE ("what is the title of..."), so
                # letting an all-or-nothing per-ontology judgement delete it is
                # how the answer column goes missing while the query still looks
                # well formed. Attributes are cheap: one line each, only on
                # classes that already survived.
                "values": list(block.get("values", [])),
                "answers_question": i in answers,
            })

        print(f"[agent6] {source}: kept {len(kept_relations)}/{catalog.edge_count} "
              f"relation(s)"
              + (f" + {len(kept_routes)}/{catalog.route_count} route(s)"
                 if kept_routes else "")
              + f" = {len(kept)} chain(s), "
              f"{len(indices) - dropped_blocks}/{len(indices)} class block(s)"
              + (f", {len(answers)} answer class(es)" if answers else "")
              + (f", {len(carriers)} carrier(s)" if carriers else ""))

    if not result:
        print("[agent6] ⚠ no ontology kept any path — Agent 7 will receive nothing")
    return result


async def contextual_class_path_filter(question: str,
                                       paths: List[PathBlock]) -> List[PathBlock]:
    return await filter_class_paths_with_llm(question, paths)
