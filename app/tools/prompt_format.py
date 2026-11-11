"""
Compact, structured rendering of the finalized class paths and the alignment for
the Agent 7 prompt.

WHY this module exists
---------------------
The previous renderer (`format_paths_by_source`) enumerated one block per PATH:
the flattened label chain plus the SPARQL triples for that chain. Measured on 8
matched classes over cmt/confof/ekaw that is 1055 chains and ~438k characters
(~110k tokens) describing only ~210 distinct edges over ~45 classes. A depth-3
walk restates its first hop in every chain that begins with it, so
`Paper -[readByReviewer]-> Reviewer` was emitted dozens of times, each time with
its triples re-rendered underneath. There was no grouping and no deduplication,
so nothing in the payload told the model how the ontology is actually shaped.

What we emit instead: an ADJACENCY LIST. Edges are grouped by subject class and
each distinct edge is stated exactly once, which lets the model traverse the
graph itself instead of us pre-enumerating every walk for it. Three deliberate
choices:

1. CURIEs with a `@prefix`/`PREFIX` header instead of a full `<http://cmt#Paper>`
   inline at every occurrence. Turtle/SPARQL CURIE syntax is heavily represented
   in pretraining, and it costs roughly a third of the tokens of the expanded
   form. The header is emitted as real `PREFIX` lines so the model can copy it
   straight into the query and the CURIEs stay legal SPARQL.

2. Every edge line IS a ready-to-copy SPARQL triple pattern, and every class gets
   exactly one variable (via `VarNamer`, so aligned classes across sources share
   a variable). This preserves the finding recorded in `tools/path_sparql.py`:
   given only label chains the generator rebuilt URIs from guessed namespaces and
   produced queries whose variables were never joined — a silent cartesian
   product. Here the joins come for free: two edges that share a class share the
   variable by construction, so `?paper ... ?reviewer` and `?reviewer ... ?conf`
   chain without the model having to align anything.

3. The class hierarchy is stated explicitly. It was entirely absent before, and
   its absence actively misled: `cmt:Person` declares almost no object property
   of its own while every paper-related relation hangs off its subclasses
   `Author` / `Reviewer`. Seeing `cmt:Author rdfs:subClassOf cmt:Person` is what
   makes that navigable.

The two layers are complementary, not redundant: the grouped edge list is the
MAP the model reasons over, the triple on each line is the TRANSCRIPTION it
copies. Nothing is enumerated twice.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rdflib import RDFS, URIRef
from rdflib.term import BNode

from core.ontology_loader import get_graph
from tools.alignment_store import Mapping, get_alignment_store, local_name, source_of
from tools.find_class_paths import value_edges_for
from tools.path_sparql import VarNamer, path_edges

# A full URI wrapped in angle brackets, as `Mapping.to_sparql_fragment` emits it.
_URI_IN_ANGLES = re.compile(r"<(https?://[^>\s]+)>")

# Edge = (subject class URI, property URI, object class URI)
Edge = Tuple[str, str, str]

# Emit the derived reverse index (which subjects point INTO each class). OFF: it
# restates facts already present, in a second notation, and every agent in this
# pipeline runs on a ~20B local model (see config/agent_models.yaml) where
# redundancy is a liability rather than a convenience.
INCLUDE_REVERSE_INDEX = False

# Stated once, directly above the data. Every consumer of this payload is a small
# local model, so nothing about the notation is left to be inferred:
#   * the sections are named explicitly (no reliance on blank-line grouping),
#   * every fact line is plain Turtle/SPARQL — the format deliberately contains NO
#     custom notation at all. In particular the flattened-chain syntax used
#     upstream (`A -[p]-> B`, and `A <-[p]- B` for an edge traversed against its
#     declared direction) never appears here: `path_edges` has already resolved
#     those into real triples in real direction. A small model that silently
#     normalised `<-[p]-` into a forward arrow would invert the relationship, so
#     the safest handling is not to show it the notation in the first place.
LEGEND = [
    "# ===== LEGEND =====",
    "# Every line is a plain Turtle/SPARQL triple (no arrow notation): copy it character for character.",
    "# SOURCES HOLDING THE ANSWER: each holds its own instances of the answer; one UNION branch each.",
    "# PREFIXES: copy into the query; prefix:Name is invalid without them.",
    "# CLASSES: one variable + type triple per class; copy the type triple whenever you constrain that class.",
    "#   ANSWER CLASS = what the question retrieves: build that block around it. Its parent (usually listed) answers a wider question,",
    "#   unless a CONDITIONAL alignment's whole expression means the answer (rule 7).",
    "# HIERARCHY: 'X rdfs:subClassOf Y' => X has every relation of Y. Relations often sit on the SUBCLASS: check subclasses",
    "#   before deciding a class has none. Reasoning aid only: never write rdfs:subClassOf in the query.",
    "# RELATIONS: one per line, each once, grouped under '# subject-class'. Lines sharing a variable are joined;",
    "#   extend a path by following that variable to the next line mentioning it.",
    "# ATTRIBUTES: values (text/number/date), not things: no type triple on the object variable, no path through it.",
    "#   For WHAT something is called or WHEN it happened, copy the line and SELECT its variable.",
    "# One SOURCE block = one ontology at one endpoint. Never mix two prefixes in one SERVICE block.",
]


# --- prefixes / CURIEs ------------------------------------------------------

def namespace_of(uri: str) -> str:
    """Namespace part of a class/property URI, including the separator.

    'http://cmt#Paper' -> 'http://cmt#'. Derived from the URI rather than from a
    hardcoded table so a fourth ontology needs no change here.
    """
    if "#" in uri:
        return uri.rsplit("#", 1)[0] + "#"
    return uri.rsplit("/", 1)[0] + "/"


# Standard W3C vocabularies keep their conventional prefix. `source_of` reads
# the ontology name out of the URI's host, which is right for `http://cmt#Paper`
# and wrong for `http://www.w3.org/2000/01/rdf-schema#label` — that would render
# as `www.w3.org:label`. These are not ontologies and never get an endpoint.
WELL_KNOWN_PREFIXES: Dict[str, str] = {
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://www.w3.org/2002/07/owl#": "owl",
}


def build_prefix_map(uris: Iterable[str]) -> Dict[str, str]:
    """prefix -> namespace, keyed by ontology short name (cmt / confof / ekaw).

    Note the prefix is the lowercased SOURCE name while the namespace keeps its
    original casing (`confof:` -> `<http://confOf#>`); the source name is what the
    rest of the pipeline and the endpoint table already use. Standard vocabularies
    are the exception and keep their conventional prefix.
    """
    prefixes: Dict[str, str] = {}
    for uri in uris:
        if not uri:
            continue
        namespace = namespace_of(uri)
        known = WELL_KNOWN_PREFIXES.get(namespace)
        if known:
            prefixes.setdefault(known, namespace)
            continue
        src = source_of(uri)
        if src:
            prefixes.setdefault(src, namespace)
    return prefixes


def curie(uri: str, prefixes: Dict[str, str]) -> str:
    """'http://cmt#Paper' -> 'cmt:Paper', or '<uri>' when no prefix covers it.

    Falling back to the angle-bracketed full URI (never to a bare local name)
    keeps the output valid SPARQL in every case.
    """
    if not uri:
        return uri
    for prefix, ns in prefixes.items():
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    return f"<{uri}>"


def curie_text(text: str, prefixes: Dict[str, str]) -> str:
    """Rewrite every `<full URI>` inside an already-rendered fragment as a CURIE.

    Used on `Mapping.to_sparql_fragment()` / `Mapping.render(uris=True)` so the
    alignment side keeps CODE-rendered semantics (the EDOAL translation stays in
    `alignment_store`, which is the only place that knows it) and only its
    surface syntax is shortened.
    """
    return _URI_IN_ANGLES.sub(lambda m: curie(m.group(1), prefixes), text)


# --- graph facts ------------------------------------------------------------

def subclass_edges(source: str, focus: Set[str]) -> List[Tuple[str, str]]:
    """(child, parent) `rdfs:subClassOf` pairs of `source` touching `focus`.

    We keep an edge when EITHER end is a class already in play. Keeping the
    parent side matters (it explains where an inherited relation comes from), but
    keeping the child side matters more: the subclasses of a retrieved class are
    usually where the relations actually live, and they are often not retrieved
    themselves.

    Blank nodes are skipped — a `subClassOf [ owl:Restriction ... ]` is an axiom,
    not a named superclass, and the restriction's content already reached us as an
    edge through the property index.
    """
    graph = get_graph().get_context("http://" + source)
    out: Set[Tuple[str, str]] = set()
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        if isinstance(child, BNode) or isinstance(parent, BNode):
            continue
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        c, p = str(child), str(parent)
        if c.startswith("http://www.w3.org/") or p.startswith("http://www.w3.org/"):
            continue
        if c in focus or p in focus:
            out.add((c, p))
    return sorted(out)


# --- path payload -----------------------------------------------------------

def collect_edges(paths: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Edge]], Dict[str, Set[str]]]:
    """Finalized path blocks -> deduplicated edges per source + the entry classes.

    Edges are derived from the SAME flattened chains the old renderer printed
    (via `path_edges`, which reuses the chain parser), so the reformatting cannot
    invent a relationship and cannot silently drop one: the edge set here is
    exactly the set of edges those chains described.
    """
    edges: Dict[str, List[Edge]] = {}
    seen: Dict[str, Set[Edge]] = {}
    entry_classes: Dict[str, Set[str]] = {}

    for item in paths:
        source = item.get("source")
        if not source:
            continue
        if item.get("class"):
            entry_classes.setdefault(source, set()).add(item["class"])
        bucket = edges.setdefault(source, [])
        seen_here = seen.setdefault(source, set())
        for chain in list(item.get("outgoing", [])) + list(item.get("incoming", [])):
            resolved = path_edges(chain, source)
            if not resolved:
                continue
            for edge in resolved:
                if edge not in seen_here:
                    seen_here.add(edge)
                    bucket.append(edge)
    return edges, entry_classes


ValueEdge = Tuple[str, str, Optional[str]]   # (class URI, property URI, datatype)

# Prefixes stripped when naming a value's variable. `cmt:title`,
# `confof:hasTitle` and `conference:has_title` are the same attribute wearing
# three naming conventions, and each ontology gets its own SERVICE block, so
# giving them one variable name is what makes the UNION branches line up on a
# single column instead of producing three mostly-empty ones.
_VALUE_VAR_PREFIXES = ("has_", "has", "is_", "is", "get_", "get")


def _value_var_name(prop_uri: str) -> str:
    name = local_name(prop_uri)
    lowered = name.lower()
    for prefix in _VALUE_VAR_PREFIXES:
        if lowered.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).lstrip("_")
    if not name:
        return "value"
    return name[0].lower() + name[1:]


def collect_answer_classes(paths: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """source -> classes belonging to a standalone or relational answer.

    Agent 6 works this out (its `answers_question` flag) and, until now, no one
    read it: every class arrived in the prompt as an undifferentiated "class
    retrieved for this question". Measured on "which of the people are still
    studying?", `confOf:Student` and `ekaw:Student` were both flagged as answers
    and both ignored — the query was built on `Person`, which returns everybody.

    Naming them costs one comment line per source and tells the model which
    variable the question is actually about.
    """
    out: Dict[str, Set[str]] = {}
    for item in paths:
        source, cls = item.get("source"), item.get("class")
        if (source and cls and
                (item.get("answers_question")
                 or item.get("complete_answer_branch"))):
            out.setdefault(source, set()).add(cls)
    return out


def collect_values(paths: List[Dict[str, Any]]) -> Dict[str, List[ValueEdge]]:
    """source -> its deduplicated value edges, in a stable order.

    Separate from `collect_edges` because these never went through the chain
    flattener: they arrive as triples and stay triples.
    """
    values: Dict[str, List[ValueEdge]] = {}
    seen: Dict[str, Set[Tuple[str, str]]] = {}
    for item in paths:
        source = item.get("source")
        if not source:
            continue
        bucket = values.setdefault(source, [])
        seen_here = seen.setdefault(source, set())
        for edge in item.get("values", ()):
            if not edge or len(edge) < 2:
                continue
            cls, prop = edge[0], edge[1]
            datatype = edge[2] if len(edge) > 2 else None
            if (cls, prop) in seen_here:
                continue
            seen_here.add((cls, prop))
            bucket.append((cls, prop, datatype))
    return values


def inherited_values(class_uri: str, source: str) -> List[ValueEdge]:
    """The value properties of a class the pipeline never selected.

    `collect_values` above only knows the classes the pipeline chose as path
    ENTRIES. A class that arrives as the TARGET of a relation gets a variable and
    a type line, but no attributes — so the model can reach it and then has
    nothing to read off it.

    Measured on "which workshops belong to a conference, and when is their
    submission deadline?": confOf:ends_on has domain confOf:Event, and
    Submission_event is an Event by two steps of rdfs:subClassOf. Conference and
    Workshop inherit it exactly the same way and BOTH had it printed, because
    both were retrieved. Submission_event was not retrieved, so the one property
    the question asked for was the one property missing — and the query took a
    date from another ontology instead.

    Same lookup the entries used (`by_value` already resolves inheritance), so
    this adds no new notion of what a class carries. It only stops the answer
    depending on which classes retrieval happened to name.

    Goes through `value_edges_for`, the same helper the entries use, so a class
    reached through the hierarchy gets the `rdfs:label` fallback on exactly the
    same terms — otherwise `cmt:Paper` could be named and `cmt:PaperAbstract`,
    two lines below it, could not.
    """
    graph = get_graph().get_context("http://" + source)
    return value_edges_for(class_uri, graph)


# Whether a source must hold the SAME CONCEPT as the anchor to be named a holder
# of the answer. Off, every source with a flagged class is named.
USE_ALIGNED_ANSWER_SOURCES = True


def _same_concept_as(anchor_classes: Set[str]) -> Set[str]:
    """Every class the alignment makes equivalent to one of `anchor_classes`.

    Read symmetrically and followed to the end: a correspondence names the same
    pair whichever side it was written from, and equivalence carries over a chain
    of them. Both matter here — the class that answers in one ontology is often
    tied to the anchor's only through a third.

    Following the conditional shapes as if they were plain equivalences does
    widen this: from a narrow class it reaches the general one, and from there
    every other narrowing of it. That is accepted, because the set is only used
    to tell one concept from an unrelated one, not to tell two narrowings apart.
    """
    store = get_alignment_store()
    reached = set(anchor_classes)
    queue = list(anchor_classes)
    while queue:
        for mapping in store.get(queue.pop()):
            for uri in [mapping.entity1, *mapping.targets]:
                if uri and uri not in reached:
                    reached.add(uri)
                    queue.append(uri)
    return reached


def _aligned_at_all(class_uris: Set[str]) -> bool:
    """Whether the alignment says anything at all about any of these classes."""
    store = get_alignment_store()
    return any(store.get(uri) for uri in class_uris)


def _answer_sources(sources: List[str], answers_by_source: Dict[str, Set[str]],
                    anchor: Optional[str]) -> List[str]:
    """The sources to name as holders of the answer, anchor first.

    A UNION branch is only meaningful for a source holding ITS OWN instances of
    what the anchor's branch retrieves, so a source qualifies on the MEANING of
    its flagged class, not on carrying a flag at all.

    Dropping needs POSITIVE evidence, because the alignment is incomplete and
    silence about a class is not a statement about it. A source is dropped only
    when its flagged classes are aligned to something and none of it is the
    anchor's concept; a class the alignment never mentions is kept, as is every
    source when the anchor's own class is one of those.
    """
    answering = [s for s in sources if answers_by_source.get(s)]
    if not USE_ALIGNED_ANSWER_SOURCES or len(answering) < 2:
        return answering

    reference = anchor if anchor in answering else answering[0]
    own = answers_by_source[reference]
    equivalent = _same_concept_as(own)
    if equivalent <= own:
        return answering

    kept, dropped = [], []
    for source in answering:
        classes = answers_by_source[source]
        if (source == reference or classes & equivalent
                or not _aligned_at_all(classes)):
            kept.append(source)
        else:
            dropped.append(source)

    if dropped:
        print(f"[prompt] aligned elsewhere than to {reference}'s answer class, "
              f"so not holders of the answer: {', '.join(dropped)}")
    return kept


def render_edge_graph(paths: List[Dict[str, Any]],
                      base_source: Optional[str] = None,
                      endpoints: Optional[Dict[str, str]] = None
                      ) -> Tuple[List[str], Dict[str, str]]:
    """(shared PREFIX header lines, {source: its adjacency-list section}).

    Split this way so the per-ontology debug files can hold the EXACT section the
    prompt contains. Rendering each source in its own call would not do: variable
    naming is global (one `VarNamer` seeded from the anchor source, so aligned
    classes in different ontologies get one shared variable), and a per-source
    call would restart the naming and drift from what the model was shown.
    """
    endpoints = endpoints or {}
    edges_by_source, entry_by_source = collect_edges(paths)
    values_by_source = collect_values(paths)
    answers_by_source = collect_answer_classes(paths)
    if not edges_by_source:
        return ([], {})

    # Anchor source first: `VarNamer` propagates a variable to the aligned
    # counterparts of a class it has already named, so naming the anchor's
    # classes first makes the other sources inherit ITS names and the UNION
    # branches line up on the same columns.
    sources = sorted(edges_by_source, key=lambda s: (s != base_source, s))

    # Hierarchy is resolved before rendering because it can introduce classes
    # that have no edge of their own (a subclass where the relations live), and
    # those still need a variable and a prefix.
    hierarchy: Dict[str, List[Tuple[str, str]]] = {}
    for source in sources:
        focus = {c for e in edges_by_source[source] for c in (e[0], e[2])}
        focus |= entry_by_source.get(source, set())
        hierarchy[source] = subclass_edges(source, focus)

    # Fill in the attributes of every class reachable in a block, not only the
    # ones retrieval named. Done HERE, before the prefix map is built, so the
    # properties this adds get a PREFIX line like any other.
    for source in sources:
        described = {cls for cls, _, _ in values_by_source.get(source, ())}
        reachable = {c for e in edges_by_source[source] for c in (e[0], e[2])}
        reachable |= entry_by_source.get(source, set())
        reachable |= {c for pair in hierarchy[source] for c in pair}
        bucket = values_by_source.setdefault(source, [])
        for cls in sorted(reachable - described):
            bucket.extend(inherited_values(cls, source))

    all_uris: List[str] = []
    for source in sources:
        for s, p, o in edges_by_source[source]:
            all_uris += [s, p, o]
        for c, p in hierarchy[source]:
            all_uris += [c, p]
        for cls, prop, _ in values_by_source.get(source, []):
            all_uris += [cls, prop]
        all_uris += sorted(entry_by_source.get(source, set()))
    prefixes = build_prefix_map(all_uris)

    # Standard-vocabulary prefixes are not sources: they have no endpoint and no
    # class of their own, so they must not seed variable naming.
    namer = VarNamer(set(endpoints)
                     or (set(prefixes) - set(WELL_KNOWN_PREFIXES.values())))

    # The legend is emitted ONCE, immediately above the data it describes, rather
    # than per source (it was 14 lines x 3 sources) and rather than only in the
    # system message thousands of tokens away.
    header = list(LEGEND)

    # The set of sources that own instances of what the question retrieves, named
    # once at the top rather than left to be pieced together from the per-source
    # ANSWER CLASS lines below. It is the membership list of the UNION. Emitted
    # only when it holds more than one source: with a single one there is no
    # UNION to decide and the line only adds noise. Order follows `sources`, so
    # the anchor comes first and the list reads in the same order as the blocks.
    # GTX Prefer deterministic source-graph completeness over Agent 6's
    # GTX per-class answer flag when the retained graph supplies that evidence.
    complete_answering = [
        source for source in sources
        if any(item.get("source") == source
               and item.get("source_complete_answer_branch")
               for item in paths)
    ]
    answering = (complete_answering
                 or _answer_sources(sources, answers_by_source, base_source))
    if len(answering) > 1:
        header.append("# ===== SOURCES HOLDING THE ANSWER =====")
        header.append(f"# {', '.join(answering)}")

    header.append("# ===== PREFIXES =====")
    for prefix in sorted(prefixes):
        header.append(f"PREFIX {prefix}: <{prefixes[prefix]}>")
    # The HIERARCHY section writes rdfs:subClassOf whether or not any rdfs term
    # was rendered through `curie`, so the line is guaranteed here — but only if
    # the prefix map did not already produce it.
    if "rdfs" not in prefixes:
        header.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")

    sections: Dict[str, str] = {}
    for source in sources:
        lines: List[str] = []
        edges = edges_by_source[source]
        entries = entry_by_source.get(source, set())

        # Classes to give a variable to: everything reachable in this block.
        classes: Set[str] = {c for e in edges for c in (e[0], e[2])}
        classes |= entries
        classes |= {c for pair in hierarchy[source] for c in pair}

        # Entry classes first, then alphabetical — a stable order so two runs of
        # the same question produce the same prompt.
        ordered_classes = sorted(classes, key=lambda u: (u not in entries, local_name(u)))
        var_of = {uri: namer.var_for(uri) for uri in ordered_classes}

        anchor_tag = " — FEDERATION ANCHOR" if source == base_source else ""
        endpoint = endpoints.get(source, source)
        lines.append(f"# ===== SOURCE {source} — endpoint <{endpoint}>{anchor_tag} =====")

        if entries:
            named = " ".join(curie(u, prefixes) for u in sorted(entries, key=local_name))
            lines.append(f"# classes retrieved for this question: {named}")

        # Stated separately and above the data, because it is the one fact that
        # decides what the query is FOR. A general class is nearly always present
        # alongside the specific one (Person next to Student), and picking the
        # general one silently widens the answer to everybody.
        # Gated on the same list as the header above: a block still marking its
        # class as the answer while the header leaves that source out states the
        # opposite of the header, and rule 4 reads the block.
        answers = (answers_by_source.get(source, set()) & classes
                   if source in answering else set())
        if answers:
            named = " ".join(curie(u, prefixes) for u in sorted(answers, key=local_name))
            lines.append(f"# ANSWER CLASS for this question in {source}: {named} — "
                         f"the query must retrieve THIS, not a more general class "
                         f"it belongs to")

        # No padding and no indentation anywhere below: every fact line has the
        # same shape, which is what a small model follows most reliably, and
        # alignment whitespace is pure token cost in a payload this size.
        lines.append(f"# --- CLASSES ({source}) ---")
        for uri in ordered_classes:
            lines.append(f"?{var_of[uri]} a {curie(uri, prefixes)} .")

        if hierarchy[source]:
            lines.append(f"# --- HIERARCHY ({source}) --- reasoning aid, do NOT query these")
            for child, parent in hierarchy[source]:
                lines.append(f"{curie(child, prefixes)} rdfs:subClassOf "
                             f"{curie(parent, prefixes)} .")

        # Group by subject so the block reads as an adjacency list the model can
        # walk, instead of a flat list of chains we chose for it.
        by_subject: Dict[str, List[Edge]] = {}
        for edge in edges:
            by_subject.setdefault(edge[0], []).append(edge)

        # Header only when there is something under it. A source can now reach the
        # prompt on a class alone (membership is the whole answer), and an empty
        # "copy these lines verbatim" heading is an invitation to invent one.
        if not by_subject:
            lines.append(f"# --- RELATIONS ({source}) --- none needed: the class "
                         f"above is the answer by itself")
        else:
            lines.append(f"# --- RELATIONS ({source}) --- copy these lines verbatim")
        for subject in sorted(by_subject, key=lambda u: (u not in entries, local_name(u))):
            # A comment, not a bare CURIE on its own line: a bare CURIE reads as
            # malformed Turtle, and a small model may try to complete it.
            lines.append(f"# {curie(subject, prefixes)}")
            for _, prop, obj in sorted(by_subject[subject], key=lambda e: local_name(e[1])):
                lines.append(f"?{var_of[subject]} {curie(prop, prefixes)} ?{var_of[obj]} .")

        # A relation is declared once, on the most general classes it applies to,
        # and every subclass inherits it. The block prints it only at the level it
        # was declared, so joining two subclasses directly is a line that exists
        # in the ontology and not in the prompt.
        #
        # Measured on "which workshops belong to a conference, and when is their
        # submission deadline?": ekaw declares partOfEvent as Event -> Event, and
        # Workshop and Conference are both Events. The block offered
        #     ?workshop ekaw:partOfEvent ?event .
        #     ?event ekaw:partOfEvent ?conference .
        # and the legend says two lines sharing a variable walk a two-hop path, so
        # the query chained them — asking for a workshop inside an event inside a
        # conference, one container more than the question has. The single hop
        # `?workshop ekaw:partOfEvent ?conference .` is the answer and was never
        # shown.
        #
        # Restricted to the classes retrieval actually named. Specialising to
        # every class in the block instead turns this one line into seven, most of
        # them about classes the question never mentions. Self-loops are dropped:
        # both ends collapse onto one variable, which asks for a thing inside
        # itself.
        children: Dict[str, Set[str]] = {}
        for child, parent in hierarchy[source]:
            children.setdefault(parent, set()).add(child)

        def below(cls: str) -> Set[str]:
            """`cls` and every class under it, within this block."""
            found, queue = {cls}, [cls]
            while queue:
                for kid in children.get(queue.pop(), ()):
                    if kid not in found:
                        found.add(kid)
                        queue.append(kid)
            return found

        known = set(edges)
        shortcuts: Set[Edge] = set()
        for subj, prop, obj in edges:
            for sub in below(subj) & entries:
                for tgt in below(obj) & entries:
                    if sub != tgt and (sub, prop, tgt) not in known:
                        shortcuts.add((sub, prop, tgt))
        if shortcuts:
            lines.append(f"# --- SHORTCUTS ({source}) --- the same relation stated "
                         f"directly between two classes that inherit it. Use ONE of "
                         f"these instead of chaining through the general class above")
            for subj, prop, obj in sorted(shortcuts,
                                          key=lambda e: (local_name(e[0]), local_name(e[1]),
                                                         local_name(e[2]))):
                lines.append(f"?{var_of[subj]} {curie(prop, prefixes)} ?{var_of[obj]} .")

        # A relation names its target after the class it POINTS AT, and the class
        # the question needs is often a subclass of that one. Those two arrive as
        # separate variables with nothing joining them, so the walk stops.
        #
        # Measured on "which workshops belong to a conference, and when is their
        # submission deadline?": the block offered
        #     ?workshop confof:hasAdministrativeEvent ?administrative_event .
        # and, separately, a variable for the submission event. Constraining the
        # first to the second needs a line that was never printed — the link
        # existed only as an rdfs:subClassOf fact, which the legend forbids
        # querying. So the road ended one step short of the answer.
        #
        # This is not new information: it is the HIERARCHY section restated on a
        # variable the model already holds, in the copyable form everything else
        # in the block uses. Optional by nature — a narrower type is a constraint,
        # and constraining when the question did not ask only loses rows.
        relation_classes = {c for e in edges for c in (e[0], e[2])}
        narrowing = sorted({(parent, child) for child, parent in hierarchy[source]
                            if parent in relation_classes and child in var_of},
                           key=lambda pair: (local_name(pair[0]), local_name(pair[1])))
        if narrowing:
            lines.append(f"# --- NARROWING ({source}) --- optional: restrict a variable "
                         f"you already have to a narrower kind. Use only when the "
                         f"question asks for that narrower kind")
            for parent, child in narrowing:
                lines.append(f"?{var_of[parent]} a {curie(child, prefixes)} .")

        # Attributes last: they are what the question asks to SEE, and a small
        # model reliably copies the section it read most recently. Only classes
        # that already have a variable in THIS block get their values printed —
        # a value line on a class the block never introduces is an orphan
        # variable, which is exactly the unconstrained-variable failure the
        # RELATIONS grouping above is careful to avoid.
        block_values = [(c, p, d) for c, p, d in values_by_source.get(source, [])
                        if c in var_of]
        if block_values:
            lines.append(f"# --- ATTRIBUTES ({source}) --- values, not classes; "
                         f"SELECT these to answer 'what is the ...'")
            # One variable per attribute NAME, shared across the sources so the
            # UNION branches line up. Two different properties that normalise to
            # the same name inside ONE block would silently join, so the second
            # one keeps its own full local name instead.
            taken: Dict[str, str] = {}
            by_class: Dict[str, List[ValueEdge]] = {}
            for edge in block_values:
                by_class.setdefault(edge[0], []).append(edge)
            for cls in sorted(by_class, key=lambda u: (u not in entries, local_name(u))):
                lines.append(f"# {curie(cls, prefixes)}")
                for _, prop, datatype in sorted(by_class[cls], key=lambda e: local_name(e[1])):
                    var = _value_var_name(prop)
                    # A standard property (rdfs:label, rdfs:comment) sits on EVERY
                    # class, so sharing one variable for it would silently demand
                    # that a paper and its review have the same label. Those get
                    # the class's own variable as a prefix; ontology-declared
                    # properties keep the shared name, which is what lines the
                    # UNION branches up across sources.
                    if namespace_of(prop) in WELL_KNOWN_PREFIXES:
                        var = f"{var_of[cls]}_{var}"
                    if taken.setdefault(var, prop) != prop:
                        var = local_name(prop)
                        suffix = 2
                        while taken.setdefault(var, prop) != prop:
                            var, suffix = f"{local_name(prop)}{suffix}", suffix + 1
                    kind = f"  # {local_name(datatype)}" if datatype else ""
                    lines.append(f"?{var_of[cls]} {curie(prop, prefixes)} ?{var} .{kind}")

        if INCLUDE_REVERSE_INDEX:
            # OFF by default. It is derived data (every fact is already above) in a
            # second, differently-shaped notation, and on a 20B local model an
            # extra ~2k characters of redundancy costs more than the lookup
            # convenience buys. Flip on if the model struggles to find the
            # relations pointing INTO a class.
            into: Dict[str, Set[str]] = {}
            for s, _, o in edges:
                into.setdefault(o, set()).add(s)
            if into:
                lines.append(f"# --- INCOMING ({source}) --- subjects with a relation into each class")
                for obj in sorted(into, key=lambda u: (u not in entries, local_name(u))):
                    subs = " ".join(curie(s, prefixes) for s in sorted(into[obj], key=local_name))
                    lines.append(f"# {curie(obj, prefixes)} <- {subs}")

        sections[source] = "\n".join(lines)

    return header, sections


def format_edge_graph(paths: List[Dict[str, Any]],
                      base_source: Optional[str] = None,
                      endpoints: Optional[Dict[str, str]] = None) -> str:
    """The LLM-facing path section: shared prefixes + one adjacency list per
    ontology, anchor source first."""
    header, sections = render_edge_graph(paths, base_source, endpoints)
    if not sections:
        return "No class relations available for these classes."
    return "\n".join(header) + "\n\n" + "\n\n".join(sections.values())


# --- alignment payload ------------------------------------------------------

def format_alignment_grouped(mappings: List[Mapping],
                             endpoints: Optional[Dict[str, str]] = None,
                             base_source: Optional[str] = None) -> str:
    """The LLM-facing alignment section, grouped by ontology PAIR.

    The flat version emitted one `*` bullet per mapping with both sides' full
    URIs and both endpoints repeated inline, so a payload of 11 correspondences
    repeated the same three endpoint URLs eleven times and gave no hint that the
    correspondences cluster into three ontology pairs.

    Here the pair (and its two endpoints) is stated once as a heading, near
    duplicates are collapsed, and each side keeps its ready-made triples — which
    for the conditional shapes (`restriction_*`) is the semantically load-bearing
    part and must survive verbatim: `ekaw:Accepted_Paper` is not `cmt:Paper`, it
    is `cmt:Paper` whose `hasDecision` value is an `Acceptance`.

    Line shape matches the path section: prose and labels live in `#` comments,
    and a triple pattern is always alone on its line. The earlier
    `cmt    : ?x a cmt:Person .` form put a bare `label :` in front of a triple,
    which a small model can easily copy into the query as if it were part of the
    pattern.
    """
    if not mappings:
        return "No cross-ontology class correspondences found for these classes."

    endpoints = endpoints or {}
    uris = [u for m in mappings for u in (m.entity1, *m.targets)]
    uris += [(m.restriction or {}).get("property") for m in mappings]
    uris += [(m.restriction or {}).get("value") for m in mappings]
    prefixes = build_prefix_map(u for u in uris if u)

    # Group on the unordered source pair: which side the alignment file happened
    # to write first is not information the generator can use.
    groups: Dict[Tuple[str, str], List[Mapping]] = {}
    for m in mappings:
        a, b = m.entity1_source, m.target_source
        if not a or not b:
            continue
        groups.setdefault(tuple(sorted((a, b))), []).append(m)

    lines: List[str] = [
        "# ===== ALIGNMENT: the same concept named differently in two ontologies =====",
        "# Grouped by ontology pair. Each entry is one equivalence, then the triple",
        "# pattern for EACH side on its own line. Copy a side's pattern into that",
        "# side's SERVICE block (rename ?x to your variable) — do not re-derive it.",
    ]

    # The PREFIX header above the path sections only covers ontologies that have a
    # schema block. A source reaching the query through a correspondence alone
    # would land here with an undeclared prefix and the model would invent one.
    # Re-declaring a prefix already stated above is harmless: same namespace.
    for prefix in sorted(prefixes):
        lines.append(f"PREFIX {prefix}: <{prefixes[prefix]}>")

    n = 0
    # Pairs involving the federation anchor first: it is the source the query is
    # built around, so its correspondences are the ones the model needs earliest.
    for pair in sorted(groups, key=lambda p: (base_source not in p, p)):
        a, b = pair
        # Both endpoints on the heading, once per pair. The flat version repeated
        # them on every correspondence, which is where most of its bulk was.
        lines.append(f"# --- {a} <-> {b} ---")
        lines.append(f"#   {a} endpoint <{endpoints.get(a, a)}>")
        lines.append(f"#   {b} endpoint <{endpoints.get(b, b)}>")

        # Deduplicate within the group. Two guards, for two distinct duplications
        # in the alignment file: identical renderings, and plain equivalences
        # stated in BOTH directions ("A = B" and "B = A"), which render
        # differently but say the same thing. Direction only carries meaning for
        # the conditional shapes, so only those are kept per-direction.
        seen: Set[str] = set()
        seen_pairs: Set[frozenset] = set()
        for m in sorted(groups[pair], key=lambda m: (m.kind != "simple", m.render())):
            rendered = curie_text(m.render(uris=True), prefixes)
            if rendered in seen:
                continue
            if m.kind == "simple":
                unordered = frozenset((m.entity1, m.targets[0]))
                if unordered in seen_pairs:
                    continue
                seen_pairs.add(unordered)
            seen.add(rendered)
            n += 1

            tag = "" if m.kind == "simple" else f"  [CONDITIONAL {m.kind}: keep the condition]"
            lines.append(f"# [{n}] {rendered}{tag}")

            # Side label as a comment, triple alone on the next line.
            lines.append(f"# side {m.entity1_source}:")
            lines.append(f"?x a {curie(m.entity1, prefixes)} .")
            lines.append(f"# side {m.target_source}:")
            for ln in curie_text(m.to_sparql_fragment(), prefixes).splitlines():
                lines.append(ln)

    return "\n".join(lines)
