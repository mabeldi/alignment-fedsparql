"""
Agent 4 — contextual relevance filter over the retrieved ontology terms.

WHAT CHANGED AND WHY
--------------------
The matched terms used to be pasted into the prompt as a raw Python `repr` of
the whole dict, and the instructions described `domain` / `range` as fields of a
term when they actually live one level down inside `payload`. On top of that the
property rule read "keep only if their domain or range includes entities that are
DIRECTLY related to the question".

Together those three things systematically deleted correct relations from any
ontology that models things generically. Measured case, question "retrieve the
decisions of papers along with their authors":

    cmt:hasAuthor   domain=Paper   range=Author     -> kept   (words match)
    ekaw:authorOf   domain=Person  range=Document   -> DROPPED

Both are the same relation. ekaw declares it on the general classes, and
`ekaw:Paper rdfs:subClassOf ekaw:Document` / `ekaw:Paper_Author rdfs:subClassOf
ekaw:Person` are facts sitting in the loaded graph that the prompt never showed.
The model was left to infer "a paper is a document" from world knowledge while
being told to be literal — so the filter was really rewarding whichever ontology
happens to name things the way the question does.

Now: terms are rendered as flat lines with domain/range lifted to the top level,
and each property carries the subclass links that connect its ends to the classes
actually retrieved for this question. The MODEL still decides — it just stops
guessing a fact we already have. The graph lookup only ANNOTATES; it never keeps
or drops anything on its own.

ONE CALL PER ONTOLOGY
---------------------
Filtering used to be a single call over every source at once. Two ontologies
naming the same concept produce lines that differ only by the URI and the
`[source]` tag, and the model reads those as duplicates and keeps one — always
the one whose wording is closest to the question. Measured: `confOf:Paper`,
`confOf:Author` and `confOf:writtenBy` all disappeared while their `cmt` twins
survived, at tied similarity scores.

So each source is now filtered in its OWN call, and the calls run concurrently.
The model never sees two ontologies side by side, so it cannot mistake one for a
duplicate of the other. This is structural, not an extra instruction: a small
model follows a prompt rule unreliably, but it cannot compare what it was never
shown.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import json
import re

from rdflib import RDFS, URIRef

from agents.path_scorer import score_terms
from tools.llm_router import run_agent_async

# Vector similarity scores are deliberately NOT shown to the model. They measure
# how much a term's NAME looks like the question's wording, which is exactly the
# lexical bias this agent exists to correct: `ekaw:authorOf` scores lower than
# `cmt:hasAuthor` for reasons that have nothing to do with relevance.
Term = Dict[str, Any]
Matches = Dict[str, List[Term]]


def _local_name(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1] if uri else uri


def _payload(term: Term) -> Dict[str, Any]:
    return term.get("payload") or {}


def _uri_of(term: Term) -> Optional[str]:
    return term.get("uri") or _payload(term).get("uri")


def _type_of(term: Term) -> Optional[str]:
    return _payload(term).get("type") or term.get("type")


def _source_of(term: Term) -> Optional[str]:
    return _payload(term).get("source") or term.get("source")


def _ends(payload: Dict[str, Any], end: str) -> List[str]:
    """Every class URI on one end of a property: `end` is "domain" or "range".

    A domain or range is not always a single class. OWL writes a choice of
    classes as a union, and the indexer stores every member in the plural keys:

        edas:relatesTo  domainUris=[Document]
                        rangeUris =[AcademicEvent, Conference, Person,
                                    ConferenceSession]

    The singular `domainUri`/`rangeUri` hold only the FIRST member, so reading
    them alone hid three of those four classes. `range=AcademicEvent` also made
    the coverage line below answer "none" for a question about a Person — which
    is worse than saying nothing, because the model reads it as proof the
    property is unrelated and drops it.

    Falls back to the singular key so a payload written before the plural keys
    existed (anything indexed earlier, or a term built by hand in a test) still
    renders instead of silently losing its ends.
    """
    plural = payload.get(f"{end}Uris")
    if plural:
        return [u for u in plural if u]
    single = payload.get(f"{end}Uri")
    return [single] if single else []


# --- graph facts ------------------------------------------------------------

def _related_classes(uri: str, candidates: Set[str]) -> List[Tuple[str, str]]:
    """Which `candidates` a class URI covers, and how: [(class uri, relation)].

    Three ways a declared domain/range can cover a class the question asked
    about, in decreasing directness:
      * it IS that class                        -> "same class"
      * that class is a subclass of it          -> "Paper subClassOf Document"
      * it is a subclass of that class          -> "Regular_Paper subClassOf Paper"

    Returns [] when the graph is unavailable rather than raising: this is prompt
    decoration, and losing it must never break the request.
    """
    if not uri or not candidates:
        return []
    if uri in candidates:
        return [(uri, f"{_local_name(uri)} (retrieved for this question)")]

    try:
        from core.ontology_loader import get_graph
        graph = get_graph()
        node = URIRef(uri)
        # transitive_objects walks UP (uri -> its ancestors), transitive_subjects
        # walks DOWN (uri -> its descendants). Both include `uri` itself, which
        # the `uri in candidates` check above already handled.
        ancestors = {str(o) for o in graph.transitive_objects(node, RDFS.subClassOf)
                     if isinstance(o, URIRef)}
        descendants = {str(s) for s in graph.transitive_subjects(RDFS.subClassOf, node)
                       if isinstance(s, URIRef)}
    except Exception:
        return []

    out: List[Tuple[str, str]] = []
    for candidate in sorted(candidates):
        if candidate in descendants:
            out.append((candidate, f"{_local_name(candidate)} subClassOf {_local_name(uri)}"))
        elif candidate in ancestors:
            out.append((candidate, f"{_local_name(uri)} subClassOf {_local_name(candidate)}"))
    return out


def _ancestors(uri: str, source: str) -> Set[str]:
    """Every class `uri` is a kind of, itself included.

    Reads the transitive closure the property index already built, so this costs
    a dict lookup and not a graph walk. Returns `{uri}` when the graph or the
    index is unavailable: a caller then finds no subclass relationship and falls
    back to the declared term, which is the behaviour this replaces.
    """
    if not uri or not source:
        return {uri} if uri else set()
    try:
        from core.ontology_loader import get_graph
        from tools.find_class_paths import query_ancestors
        return query_ancestors(uri, get_graph().get_context("http://" + source))
    except Exception:
        return {uri}


def _most_specific(uris: List[str], source: str) -> List[str]:
    """`uris` minus every member that a NARROWER member of the same list covers.

    `[Person, Paper_Author]` -> `[Paper_Author]`: the two name the same things,
    minus some, and every attribute and relation declared on `Person` is indexed
    for `Paper_Author` too (see build_index), so the wider one adds no reachable
    term — only a second class to pair up and walk from.

    Order is preserved, and a member is dropped only when another is a STRICT
    subclass of it. Some ontologies write equivalence as two subClassOf links in
    opposite directions; without the strictness test each of that pair would
    delete the other and both would disappear.
    """
    if len(uris) < 2:
        return list(uris)
    anc = {u: _ancestors(u, source) for u in uris}
    return [u for u in uris
            if not any(other != u and u in anc[other] and other not in anc[u]
                       for other in uris)]


def _retrieved_classes(matches: Matches) -> Dict[str, Set[str]]:
    """source -> the Class URIs retrieved for this question.

    Kept per source because a subclass link only means something inside one
    ontology; comparing `ekaw:Person` against `cmt:Author` would be nonsense.
    """
    by_source: Dict[str, Set[str]] = {}
    for terms in matches.values():
        for term in terms:
            if _type_of(term) != "Class":
                continue
            uri, source = _uri_of(term), _source_of(term)
            if uri and source:
                by_source.setdefault(source, set()).add(uri)
    return by_source


# --- prompt -----------------------------------------------------------------

def render_terms(matches: Matches) -> str:
    """The term listing shown to the model: one line per term, flat.

    Everything the instructions refer to is at the top level of the line —
    nothing the model needs is nested inside another object, and no term is
    described by a field name that is not actually there.
    """
    classes_by_source = _retrieved_classes(matches)

    lines: List[str] = []
    for keyword, terms in matches.items():
        lines.append(f"## keyword: {keyword}")
        if not terms:
            lines.append("  (no matches)")
            continue
        for term in terms:
            uri = _uri_of(term)
            if not uri:
                continue
            payload = _payload(term)
            ttype = _type_of(term) or "Term"
            source = _source_of(term) or "?"
            label = payload.get("label") or term.get("label") or _local_name(uri)

            line = f"- <{uri}>  {ttype}  {label}  [{source}]"
            if ttype in ("ObjectProperty", "DatatypeProperty"):
                # Union ends are joined with "or" rather than a comma or a pipe:
                # it states the actual meaning — the end is ANY ONE of these —
                # where a bare separator leaves the model to guess whether the
                # list is a choice or a sequence.
                domain = " or ".join(_local_name(u) for u in _ends(payload, "domain"))
                # A datatype range is `xsd:string`, so it is shown by local name:
                # the full URI is 40 characters that say "string" and nothing the
                # model has to reason about. Datatype ranges are never unions.
                range_ = " or ".join(_local_name(u) for u in _ends(payload, "range"))
                line += f"  domain={domain or '?'}  range={range_ or '?'}"
            lines.append(line)

            if ttype not in ("ObjectProperty", "DatatypeProperty"):
                continue

            # The load-bearing addition: state which of the question's own
            # classes each end covers, so "Document" stops looking unrelated to
            # a question about papers.
            candidates = classes_by_source.get(source, set())
            covered: List[str] = []
            # A datatype property has only one end that can cover a class: its
            # range is `xsd:string`, never something the question asked about.
            ends = ("domain",) if ttype == "DatatypeProperty" else ("domain", "range")
            for end in ends:
                # Every member of a union end is checked, not just the first:
                # the connection the question needs is often through a later one.
                for end_uri in _ends(payload, end):
                    for _, how in _related_classes(end_uri, candidates):
                        entry = f"{end}: {how}"
                        # A retrieved class can sit under two members of the same
                        # union and be reported twice, saying nothing new.
                        if entry not in covered:
                            covered.append(entry)
            lines.append(f"  covers retrieved classes: {'; '.join(covered) if covered else 'none'}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are an expert ontology assistant.\n"
    "You are given a user's question and the ontology terms retrieved for it. "
    "You select the terms that are semantically relevant to answering the question.\n"
    "You answer with a JSON array of the selected URIs and nothing else."
)


def build_messages(question: str, matches: Matches) -> List[Dict[str, str]]:
    user_prompt = f"""
Question:
"{question}"

Retrieved ontology terms. Each line starts with the term's URI, then its kind,
its name and the ontology it comes from. A property line also gives its domain
and range, and the line under it says which of the classes retrieved for this
question those ends cover.

There are three kinds of term:
- Class            — a type of thing.
- ObjectProperty   — a relation from one thing to another thing.
- DatatypeProperty — a VALUE carried by a thing: a name, a title, a date, a
                     number. Its range is a datatype, not a class.

{render_terms(matches)}

Select the terms to KEEP.

Classes:
- Keep the classes naming the entity types the question talks about, even when
  the question is phrased around a relationship — a question about how things
  relate is still about the things being related, and those classes are needed
  later.

ObjectProperties:
- Keep a property when its domain or its range covers a concept the question
  asks about. A GENERAL class COUNTS: if a property is declared on a broad class
  and the question asks about a specific kind of that class, the property still
  applies. Read the 'covers retrieved classes' line — it states these subclass
  links explicitly, so you do not have to assume them.
- 'covers retrieved classes: none' means neither end connects to any class the
  question asked about. Decide such a property on its ENDS: keep it only if its
  domain or its range is ITSELF something the question asks about. A name that
  echoes the question's wording is not enough on its own.
- Judge a property by what it MEANS, not by how close its name is to the
  question's wording. A relation declared on generically named ends is just as
  relevant as one whose names echo the question.

DatatypeProperties:
- These carry the VALUES the question asks to be shown. When the question asks
  what something is CALLED or NAMED, WHEN it happened, HOW MANY there are, or
  for any other piece of text or number about a thing, the answer IS a
  DatatypeProperty — dropping it leaves a query that finds the right things and
  prints nothing about them.
- Keep one when its domain is a class the question talks about, or a general
  class that such a class falls under. Read the 'covers retrieved classes' line.
- The same value goes by different names in different ontologies — one may call
  it a title, another a name, another a label. Judge it by what the value IS,
  not by whether the word matches the question.
- Drop one that describes something the question never asks about.

Keep properties that express a requested value even when their domain is an intermediate class not explicitly mentioned in the question. Preserve that domain as a candidate carrier so subsequent path discovery can determine whether it connects to the requested entity
Drop everything that has no bearing on answering the question.

Output format:
- A strict JSON array of the URIs you selected, e.g.
  ["http://example.org/onto#Thing", "http://example.org/onto#relatesTo"]
- Empty array if nothing is relevant: []
- Copy each URI exactly as it appears. No text, no explanation, no markdown.
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --- selection --------------------------------------------------------------

def _parse_selection(raw: str) -> Optional[Set[str]]:
    """Model output -> the set of URIs it kept, or None if unreadable.

    None and set() must stay distinguishable: an empty selection legitimately
    means "nothing is relevant", a parse failure means we learned nothing.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    if not text.lstrip().startswith("["):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None

    kept: Set[str] = set()
    for item in data:
        if isinstance(item, str):
            kept.add(item.strip().strip("<>"))
        elif isinstance(item, dict):
            # Tolerated: the model returned objects instead of bare URIs.
            uri = item.get("uri") or (item.get("payload") or {}).get("uri")
            if isinstance(uri, str):
                kept.add(uri.strip().strip("<>"))
    return kept


def _apply_selection(matches: Matches, kept: Set[str]) -> Matches:
    """Rebuild the ORIGINAL objects for the kept URIs, keyword grouping intact.

    The model's answer is a SELECTION, never data. Earlier versions read back the
    term objects it echoed, which it reshapes — it flattens them and drops
    `payload` / `source`, both of which Agents 3+ require. Selecting by URI also
    means a hallucinated URI simply matches nothing and disappears, and the
    keyword grouping is ours rather than whatever the model chose to call it.
    """
    result: Matches = {}
    for keyword, terms in matches.items():
        seen: Set[str] = set()
        selected = []
        for term in terms:
            uri = _uri_of(term)
            if uri and uri in kept and uri not in seen:
                seen.add(uri)
                selected.append(term)
        result[keyword] = selected
    return result


def _best_end(question: str, source: str, uris: List[str]) -> Optional[str]:
    """The one URI of `uris` whose name is closest to the question.

    None when the comparison cannot be trusted — the question or any single
    member failed to embed — because ranking a scored member against an unscored
    one would pick a winner on missing data. The caller then keeps every member,
    which is the behaviour this selection replaces.
    """
    scores = score_terms(question, [(uri, source) for uri in uris])
    if len(scores) != len(uris):
        return None
    return max(scores, key=scores.get)


def _matched_under(ends: List[str], source: str,
                   matched: Dict[str, Set[str]]) -> List[str]:
    """The matched classes of `source` that ARE one of `ends`, or a kind of one.

    This is what a property endpoint should be for THIS question, as opposed to
    what the ontology declares it as in general. `hasName` is declared on
    `Person`; if the question retrieved `Paper_Author`, then for this question
    the attribute hangs off `Paper_Author` — the index materialises every
    inherited property onto the subclass, so nothing is lost by saying so, and
    `Person` never has to be added as a second class next to the one already
    retrieved.

    Covers the union case with the same test, because `_ancestors` includes the
    class itself: a union member that was itself retrieved is "a kind of" one of
    `ends` and comes back as a hit. That is the case similarity scoring must not
    decide — retrieval already answered it, and a name-similarity comparison
    between the members can and does pick a DIFFERENT member, which then gets
    added as a class nobody asked for while the retrieved one goes unused.

    Several classes can qualify at once ("names of authors and reviewers", both
    under `Person`) and all of them are returned: choosing one would state the
    other as absent from a question that named it. `_most_specific` still runs,
    so a hit list holding both a class and its own subclass keeps the subclass.
    """
    declared = {u for u in ends if u}
    if not declared:
        return []
    hits = [uri for uri in sorted(matched.get(source, ()))
            if _ancestors(uri, source) & declared]
    return _most_specific(hits, source)


def _recover_classes_from_properties(result: Matches, question: str = "",
                                     select_best_end: bool = True) -> Matches:
    """Rebuild Class terms from the domain/range of the properties kept.

    Agent 3 can only build paths from Class terms, so when the filter keeps the
    relations but drops the classes the whole pipeline goes empty. A kept relation
    already names its two classes, so we restore them deterministically instead of
    trusting the model to have kept them.

    BOTH property kinds are recovered, but not symmetrically:

      ObjectProperty   — domain AND range. Both ends are classes and both are
                         walkable, so both are restored.
      DatatypeProperty — the DOMAIN only. Its range is a datatype (`xsd:string`),
                         never a class: adding it as a Class term would put a
                         datatype into the connection search and into the
                         alignment expansion, where nothing can do anything with
                         it but waste a walk.

    Datatype properties used to be skipped outright, which lost the very thing
    the filter had just decided was relevant. Measured case: `confOf:abstract`
    (DatatypeProperty, score 0.954, domain=Contribution) was KEPT by this agent,
    but `confOf#Contribution` was not among the retrieved classes, so it got no
    path block, confof reached no later agent, and the attribute the question
    asked to be shown vanished. Recovering the domain is all it takes: once a
    class is on the list, `value_edges_for` hands over every attribute declared
    on it — `value_edges_for(confOf#Contribution)` returns hasTitle,
    contactEmail, hasKeyword, remark AND abstract — so no attribute needs wiring
    of its own beyond getting its domain class onto the matched list.

    WHICH CLASS AN END RESOLVES TO
    ------------------------------
    An end is resolved against the classes ALREADY MATCHED for this question
    first (`_matched_under`), and only falls through to similarity scoring when
    that finds nothing. A matched class that is the declared end, or a kind of
    it, is used in its place and no new class is created — so a property
    declared on a general class stops dragging that general class in next to the
    specific one the question actually retrieved.

    `select_best_end` therefore describes the FALLBACK only — what to do with an
    end naming several classes when retrieval covered none of them:

      True  — score each member against the question and restore only the best.
              A union is a CHOICE of classes ("this relates a document to an
              event OR a conference OR a person OR a session"), and a question
              asks about one of them; restoring all four states the other three
              as things the question mentioned, and each one costs an Agent 3
              walk, Agent 6 prompt lines, and up to five more classes when the
              alignment expands it.
      False — restore every member, the original behaviour.

    A comparison that cannot be scored falls back to keeping all (see
    `_best_end`). An end naming a single class never reaches the fallback at
    all: either a matched class covers it, or it is restored as declared.

    Whatever the outcome, the classes finally chosen are stamped back onto the
    property term as `selected_ends = {"domain": [...], "range": [...]}`. When
    exactly one class is selected, the payload's singular `domain`/`domainUri`
    or `range`/`rangeUri` fields are updated to that winner while the plural
    fields retain the complete ontology declaration. `selected_ends` is the seed
    main.py turns into `participated_properties` and Agent 3 puts at the
    head of the connector list for the pair, so the edge the retrieval actually
    matched cannot be one of the ones `MAX_CONNECTORS_PER_PAIR` truncates away.
    It is stamped even when the winner was ALREADY a retrieved Class term and no
    new term is appended — that is the case where the class was most obviously
    relevant (a human-independent retriever found it too), and it is exactly the
    case an `existing`-guarded stamp would have skipped.
    """
    existing = {
        _uri_of(t) for terms in result.values() for t in terms
        if _type_of(t) == "Class"
    }
    existing.discard(None)
    # Per SOURCE, because a subclass link only means something inside one
    # ontology: `ekaw:Paper_Author` is not a kind of `cmt:Person`, it is a
    # different ontology's word for a similar idea.
    matched_by_source = _retrieved_classes(result)

    recovered = 0
    for terms in result.values():
        additions = []
        for term in terms:
            payload = _payload(term)
            ptype = payload.get("type")
            if ptype not in ("ObjectProperty", "DatatypeProperty"):
                continue
            source = payload.get("source")
            # A datatype property has one usable end. Its `range` key does hold a
            # URI, but it is `xsd:string` — walking it or restoring it as a class
            # is meaningless, so it is never read here and `selected_ends` reports
            # it as empty rather than absent, so a reader never has to know which
            # kind of property it is holding.
            ends = ("domain",) if ptype == "DatatypeProperty" else ("domain", "range")
            selected: Dict[str, List[str]] = {"domain": [], "range": []}
            # Read from the plural keys, never from `domainUri` / `rangeUri`:
            # those hold member #1, which is document order and nothing else, so
            # reading them alone picked a member at random and hid the rest.
            #
            # The label is derived from each URI rather than read from the
            # payload's `domain` / `range`, which name member #1 only and would
            # otherwise label every later member with the first one's name.
            for end in ends:
                uris = _ends(payload, end)
                # Retrieval first, scoring only as the fallback. A matched class
                # that is the declared end, or a kind of it, is a FACT about this
                # question; `_best_end` is a guess from name similarity. Letting
                # the guess run first was how a union whose member had already
                # been retrieved could still resolve to a different member.
                covered = _matched_under(uris, source, matched_by_source)
                if covered:
                    if set(covered) != set(uris):
                        print(f"[agent4] {_local_name(payload.get('uri'))} {end}: "
                              f"declared {', '.join(_local_name(u) for u in uris)}"
                              f" -> using matched "
                              f"{', '.join(_local_name(u) for u in covered)}")
                    uris = covered
                elif select_best_end and len(uris) > 1:
                    best = _best_end(question, source, uris)
                    if best is None:
                        print(f"[agent4] {_local_name(payload.get('uri'))} {end}: "
                              f"could not score {len(uris)} classes — keeping all")
                    else:
                        others = [_local_name(u) for u in uris if u != best]
                        print(f"[agent4] {_local_name(payload.get('uri'))} {end}: "
                              f"kept {_local_name(best)}, dropped {', '.join(others)}")
                        uris = [best]
                # The winners, recorded BEFORE the `existing` guard below: that
                # guard decides whether a Class term has to be CREATED, which is
                # a different question from which classes this property connects.
                selected[end] = list(uris)
                # The singular fields initially contain member #1 of the OWL
                # union. Once this agent has resolved the union, leaving those
                # fields untouched makes downstream/debug output report a
                # different class from the one actually recovered. Preserve the
                # plural fields as the full declaration and stamp the singular
                # fields only when there is one unambiguous selected member.
                if len(uris) == 1:
                    payload[f"{end}Uri"] = uris[0]
                    payload[end] = _local_name(uris[0])
                for uri in uris:
                    if uri in existing:
                        continue
                    existing.add(uri)
                    label = _local_name(uri)
                    additions.append({
                        "label": label,
                        "uri": uri,
                        "type": "Class",
                        # 0.0, matching federation_planner's alignment-injected
                        # classes: a class nobody retrieved must never win
                        # base-source selection in ontology_ranking_agent.
                        "score": 0.0,
                        "payload": {"type": "Class", "uri": uri, "label": label,
                                    "source": source},
                        # provenance, not a retrieved match
                        "derived_from": payload.get("uri"),
                    })
                    recovered += 1
            term["selected_ends"] = selected
        terms.extend(additions)

    if recovered:
        print(f"[agent4] recovered {recovered} Class term(s) from kept property domain/range")
    return result


def _drop_covered_superclasses(matches: Matches) -> Matches:
    """Drop a matched class that a narrower matched class of the same ontology
    already covers.

    `Paper_Author subClassOf Person`, both matched, is not two things the
    question asked about — it is one thing named twice, at two widths. Keeping
    both costs more than a duplicate line:

      * Agent 3 pairs every matched class with every other and BFSs between
        them. `subClassOf` is not a walkable edge, so the pair is searched as if
        the two were unrelated, and because the index materialises the parent's
        properties onto the child BOTH ends carry the same edge set. Every
        shared neighbour therefore becomes a "connector": measured on ekaw with
        {Paper_Author, Paper, Person}, that pair alone yields 8, all of the form
        "an author and a person who wrote the same document" — vacuous, since
        every Paper_Author IS a Person.
      * `_passes_through` does not prune them. It compares URIs exactly, and the
        intermediates there are `Document` and `Event`, neither of which is a
        matched URI even though `Paper` (which IS matched) sits under Document.
      * The result depends on RETRIEVAL ORDER. `shortest_connectors(a, b)` only
        widens `b` upwards, so the same three classes give 24 path lines when
        the subclass is listed first and 8 when the superclass is. Same
        question, different answer, decided by which term the vector search
        happened to return first.

    Nothing is lost by dropping the parent: the child inherits its attributes
    and its relations in the index, so every term reachable through `Person` is
    reachable through `Paper_Author`. What is lost is the wider EXTENT — the
    people who are not authors — which is why this only fires when the narrower
    class is itself matched. A question about all people retrieves `Person` and
    no subclass of it, and then there is nothing to drop.
    """
    keep_by_source = {source: set(_most_specific(sorted(uris), source))
                      for source, uris in _retrieved_classes(matches).items()}

    dropped: Dict[str, List[str]] = {}
    result: Matches = {}
    for keyword, terms in matches.items():
        kept: List[Term] = []
        for term in terms:
            if _type_of(term) == "Class":
                uri, source = _uri_of(term), _source_of(term)
                keep = keep_by_source.get(source)
                # `keep is None` means the term carries no usable source, so no
                # subclass comparison was possible for it. Untouched rather than
                # dropped: this prunes redundancy, it never deletes on missing
                # information.
                if uri and keep is not None and uri not in keep:
                    dropped.setdefault(source, []).append(_local_name(uri))
                    continue
            kept.append(term)
        result[keyword] = kept

    for source, names in dropped.items():
        print(f"[agent4] {source}: dropped {len(names)} superclass(es) already "
              f"covered by a matched subclass — {', '.join(sorted(set(names)))}")
    return result


def _count_types(matches: Matches) -> Dict[Optional[str], int]:
    counts: Dict[Optional[str], int] = {}
    for terms in matches.values():
        for term in terms:
            ttype = _type_of(term)
            counts[ttype] = counts.get(ttype, 0) + 1
    return counts


def _group_by_source(matches: Matches) -> Dict[str, Matches]:
    """Split into one Matches dict per ontology, keyword grouping preserved."""
    grouped: Dict[str, Matches] = {}
    for keyword, terms in matches.items():
        for term in terms:
            source = _source_of(term) or "?"
            grouped.setdefault(source, {}).setdefault(keyword, []).append(term)
    return grouped


def _merge_by_keyword(matches: Matches, by_source: Dict[str, Matches]) -> Matches:
    """Recombine the per-source results into the original keyword grouping."""
    result: Matches = {}
    for keyword in matches:
        merged: List[Term] = []
        for source_result in by_source.values():
            merged.extend(source_result.get(keyword, []))
        result[keyword] = merged
    return result


async def _filter_source(question: str, source: str, source_matches: Matches) -> Matches:
    """Filter one ontology's terms. Fails OPEN, independently of the others."""
    messages = build_messages(question, source_matches)
    try:
        raw = await run_agent_async("agent4_match_relevance", messages=messages)
    except Exception as exc:
        print(f"[agent4] {source}: LLM call failed ({exc}) — keeping all matches")
        return source_matches

    kept = _parse_selection(raw)
    if kept is None:
        print(f"[agent4] {source}: could not parse selection — keeping all matches. "
              f"Raw: {raw[:200]!r}")
        return source_matches

    return _apply_selection(source_matches, kept)


async def filter_relevant_matches_with_llm(question: str, matches: Matches,
                                           select_best_end: bool = True) -> Matches:
    """Keep the retrieved terms that bear on the question.

    `select_best_end` is passed to `_recover_classes_from_properties`: True
    restores only the best-matching class of a multi-class domain/range, False
    restores every one of them.
    """
    if not matches:
        return matches

    grouped = _group_by_source(matches)
    sources = list(grouped)
    filtered = await asyncio.gather(
        *(_filter_source(question, s, grouped[s]) for s in sources)
    )

    before = _count_types(matches)
    result = _merge_by_keyword(matches, dict(zip(sources, filtered)))
    print(f"[agent4] kept by type: {_count_types(result)}  (from {before})")

    # A keyword that had matches and now has none is the failure mode that used
    # to be invisible: it only showed up as an empty path set much later.
    emptied = [k for k, terms in matches.items() if terms and not result.get(k)]
    if emptied:
        print(f"[agent4] ⚠ keyword(s) left with no term: {', '.join(emptied)}")

    result = _recover_classes_from_properties(result, question, select_best_end)

    # After recovery, not before: recovery resolves each property endpoint down
    # to the matched classes under it, so most parents never get created in the
    # first place. This catches what is left — a parent and a child that
    # RETRIEVAL returned together, and the rarer case of two properties whose
    # ends recover two classes of the same hierarchy.
    result = _drop_covered_superclasses(result)

    if not _count_types(result).get("Class"):
        print("[agent4] ⚠ no Class-typed terms survived — class_paths/alignment will be empty")

    return result
