from rdflib import Graph, URIRef, RDF, RDFS, OWL
from rdflib.term import Node, BNode
from typing import List, Tuple, Set, Dict


# ---------------------------------------------------------------------------
# Precomputed property index
# ---------------------------------------------------------------------------
# Instead of running an `rdfs:subClassOf*` SPARQL property path on every
# per-class / per-depth lookup, we build ONCE (per ontology context) two maps:
#
#   by_domain[C] -> set of (propertyUri, rangeUri)   (outgoing properties of C)
#   by_range[C]  -> set of (propertyUri, domainUri)  (incoming properties of C)
#
# These reproduce the exact semantics of the old SPARQL: a property declared
# with rdfs:domain D (resp. rdfs:range R) applies to D itself AND every
# subclass (descendant) of D, because `class_uri rdfs:subClassOf* D` is true
# whenever D is an ancestor-or-self of class_uri.
#
# The index is cached by the graph context's stable `.identifier` (NOT
# id(graph), which the interpreter can recycle), so building twice for the
# same ontology is a no-op.
# ---------------------------------------------------------------------------

# LIVE cache, used by build_index() below.
# identifier (str) -> {"by_domain": {cls: [prop dict, ...]}, "by_range": {...}}
_INDEX_CACHE: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}


def _is_relevant_class(node: Node) -> bool:
    """Skip blank nodes and built-in (http://www.w3.org/*) classes, matching
    the existing filtering intent."""
    if isinstance(node, BNode):
        return False
    if isinstance(node, URIRef) and str(node).startswith("http://www.w3.org/"):
        return False
    return isinstance(node, URIRef)


def _expand_class_expression(graph: Graph, node: Node, _seen: Set[Node] = None) -> List[str]:
    """Resolve a class expression used as a domain/range into concrete class IRIs.

    A plain IRI denotes itself. A blank node carrying `owl:unionOf` denotes every
    member of its list, so `rdfs:domain [ owl:unionOf (A B C) ]` yields A, B, C.
    Recursive, so unions nested inside unions flatten too.
    """
    if _seen is None:
        _seen = set()
    if node in _seen:
        return []
    _seen.add(node)

    if _is_relevant_class(node):
        return [str(node)]
    if not isinstance(node, BNode):
        return []

    out: List[str] = []
    for lst in graph.objects(node, OWL.unionOf):
        for member in graph.items(lst):
            out.extend(_expand_class_expression(graph, member, _seen))
    return out


def _restriction_edges(graph: Graph):
    """Yield (domainUri, propertyUri, rangeUri) from existential/universal
    restrictions:

        C rdfs:subClassOf     [ a owl:Restriction ; owl:onProperty P ;
        C owl:equivalentClass   owl:someValuesFrom | owl:allValuesFrom R ]

    Many ontologies (edas, ekaw) declare most of their relations ONLY this way,
    never with rdfs:domain / rdfs:range, so ignoring restrictions drops the
    majority of the schema graph. Deliberately does NOT require P to be typed
    `owl:ObjectProperty` -- restriction-only properties are often left untyped;
    the IRI filter on R already excludes datatype restrictions.
    """
    for pred in (RDFS.subClassOf, OWL.equivalentClass):
        for cls, restr in graph.subject_objects(pred):
            if not _is_relevant_class(cls) or not isinstance(restr, BNode):
                continue
            if (restr, RDF.type, OWL.Restriction) not in graph:
                continue
            for prop in graph.objects(restr, OWL.onProperty):
                if not isinstance(prop, URIRef):
                    continue
                for filler_pred in (OWL.someValuesFrom, OWL.allValuesFrom):
                    for filler in graph.objects(restr, filler_pred):
                        for range_uri in _expand_class_expression(graph, filler):
                            yield str(cls), str(prop), range_uri


def _datatype_edges(graph: Graph):
    """Yield (domainUri, propertyUri, datatypeUri | None) for datatype properties.

    A datatype property is a DEAD END in the class graph: its object is a value
    (a string, a number, a date), not another class. That is exactly why the
    object-property walk above cannot carry it — every rule there is about
    getting from one class to the next.

    Dropping them entirely was wrong though. "What is the title of each paper?"
    is answered by `cmt:title`, and no chain of object properties reaches it: the
    answer is a value hanging off the class, not a class further along.

    Declared two ways, both collected:
      * `P rdfs:domain C` (with `owl:unionOf` domains expanded, as elsewhere)
      * `C rdfs:subClassOf [ owl:onProperty P ; owl:someValuesFrom D ]`, since
        edas/ekaw declare much of their schema only through restrictions.

    The range is kept as-is (`xsd:string`) and may be None: unlike a class range
    it is never traversed, so a missing one costs nothing but a less precise
    label in the prompt.
    """
    datatype_props = {p for p in graph.subjects(RDF.type, OWL.DatatypeProperty)
                      if isinstance(p, URIRef)}

    for prop in datatype_props:
        range_node = graph.value(subject=prop, predicate=RDFS.range)
        range_uri = str(range_node) if isinstance(range_node, URIRef) else None
        for domain_node in graph.objects(prop, RDFS.domain):
            for domain_uri in _expand_class_expression(graph, domain_node):
                yield domain_uri, str(prop), range_uri

    for pred in (RDFS.subClassOf, OWL.equivalentClass):
        for cls, restr in graph.subject_objects(pred):
            if not _is_relevant_class(cls) or not isinstance(restr, BNode):
                continue
            if (restr, RDF.type, OWL.Restriction) not in graph:
                continue
            for prop in graph.objects(restr, OWL.onProperty):
                # Only properties DECLARED datatype: an untyped one in a
                # restriction is assumed to be an object property by
                # _restriction_edges, and claiming it in both places would put
                # the same edge in the class graph and in the value list.
                if prop not in datatype_props:
                    continue
                filler = None
                for filler_pred in (OWL.someValuesFrom, OWL.allValuesFrom):
                    for node in graph.objects(restr, filler_pred):
                        if isinstance(node, URIRef):
                            filler = str(node)
                yield str(cls), str(prop), filler


def _build_descendants(graph: Graph) -> Dict[str, Set[str]]:
    """Build, in pure Python, the DESCENDANTS closure of the class hierarchy.

    `rdfs:subClassOf` triples go child -> parent (child subClassOf parent).
    For domain/range expansion we need the inverse direction: given a class D,
    all classes C such that C subClassOf* D (i.e. every descendant of D).
    We invert the subClassOf edges (parent -> children) and take the
    transitive closure downward.
    """
    # parent -> direct children
    children: Dict[str, Set[str]] = {}
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        if not (_is_relevant_class(child) and _is_relevant_class(parent)):
            continue
        children.setdefault(str(parent), set()).add(str(child))

    descendants: Dict[str, Set[str]] = {}

    def collect(cls: str) -> Set[str]:
        if cls in descendants:
            return descendants[cls]
        # placeholder to guard against cycles in the subClassOf graph
        descendants[cls] = set()
        acc: Set[str] = set()
        for kid in children.get(cls, ()):  # direct children
            acc.add(kid)
            acc |= collect(kid)
        descendants[cls] = acc
        return acc

    for parent in list(children.keys()):
        collect(parent)

    return descendants


# ---------------------------------------------------------------------------
# build_index -- verbatim port of _ensure_index() from the previous project.
# ---------------------------------------------------------------------------
# Logic is unchanged. Only the things that could not work as-is were touched:
#
#   * `self._graph_index` -> module-level `_INDEX_CACHE` (it was a method on a
#     class that no longer exists here).
#   * cache key `id(graph)` -> `str(graph.identifier)`. get_context() hands back
#     a FRESH Graph wrapper on every call, so id() never hit the cache and the
#     index was rebuilt on every single lookup.
#   * returns the index instead of `stats`/`None`, because ensure_index() and
#     the lookups need the maps back. The stats are still computed and printed.
# ---------------------------------------------------------------------------

# Set False to silence the per-ontology build stats.
INDEX_VERBOSE = True


def build_index(graph: Graph) -> Dict[str, Dict[str, List[dict]]]:
    """Build full property index once per graph. All subsequent lookups are O(1).
    Returns the index; prints stats dict on first build, nothing on cache hit."""
    import time
    import sys

    key = str(graph.identifier)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    build_start = time.perf_counter()

    # Build class hierarchy in Python
    all_classes = set()
    for s, _, _ in graph.triples((None, RDF.type, OWL.Class)):
        if not isinstance(s, BNode) and not str(s).startswith("http://www.w3.org/"):
            all_classes.add(str(s))
    for s, _, o in graph.triples((None, RDFS.subClassOf, None)):
        for node in (s, o):
            if not isinstance(node, BNode) and not str(node).startswith("http://www.w3.org/"):
                all_classes.add(str(node))

    supers_map = {}
    for cls in all_classes:
        ancestors = set()
        queue = [cls]
        while queue:
            c = queue.pop()
            if c in ancestors:
                continue
            ancestors.add(c)
            for _, _, parent in graph.triples((URIRef(c), RDFS.subClassOf, None)):
                if not isinstance(parent, BNode):
                    p_str = str(parent)
                    if not p_str.startswith("http://www.w3.org/"):
                        queue.append(p_str)
        supers_map[cls] = ancestors

    # ONE SPARQL query to load ALL properties (direct + unionOf + restriction)
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT ?property ?domainUri ?rangeUri WHERE {
        {
            ?property a owl:ObjectProperty .
            ?property rdfs:domain ?domainUri .
            ?property rdfs:range ?rangeUri .
        }
        UNION
        {
            ?property a owl:ObjectProperty .
            ?property rdfs:domain ?rawDomain .
            ?property rdfs:range ?rangeUri .
            ?rawDomain owl:unionOf ?domainList .
            ?domainList rdf:rest*/rdf:first ?domainUri .
        }
        UNION
        {
            ?property a owl:ObjectProperty .
            ?property rdfs:domain ?domainUri .
            ?property rdfs:range ?rawRange .
            ?rawRange owl:unionOf ?rangeList .
            ?rangeList rdf:rest*/rdf:first ?rangeUri .
        }
        UNION
        {
            ?domainUri rdfs:subClassOf ?restriction .
            ?restriction a owl:Restriction ;
                         owl:onProperty ?property .
            { ?restriction owl:someValuesFrom ?rangeUri . }
            UNION
            { ?restriction owl:allValuesFrom ?rangeUri . }
        }
        FILTER(isIRI(?domainUri) && !STRSTARTS(STR(?domainUri), "http://www.w3.org/"))
        FILTER(isIRI(?rangeUri) && !STRSTARTS(STR(?rangeUri), "http://www.w3.org/"))
    }
    """
    all_props = [
        {"uri": str(row.property), "domainUri": str(row.domainUri), "rangeUri": str(row.rangeUri)}
        for row in graph.query(query)
    ]

    # Index props by declared domain/range
    props_by_domain = {}
    props_by_range = {}
    for p in all_props:
        props_by_domain.setdefault(p["domainUri"], []).append(p)
        props_by_range.setdefault(p["rangeUri"], []).append(p)

    # For each class, collect applicable properties via inheritance
    by_domain = {}
    by_range = {}
    for cls in all_classes:
        ancestors = supers_map.get(cls, {cls})

        seen_out = set()
        outgoing = []
        for anc in ancestors:
            for p in props_by_domain.get(anc, []):
                key_p = (p["uri"], p["domainUri"], p["rangeUri"])
                if key_p not in seen_out:
                    seen_out.add(key_p)
                    outgoing.append(p)
        direct_ranges = {p["rangeUri"] for p in outgoing if p["domainUri"] == cls}
        by_domain[cls] = [p for p in outgoing if p["domainUri"] == cls or p["rangeUri"] not in direct_ranges]

        seen_in = set()
        incoming = []
        for anc in ancestors:
            for p in props_by_range.get(anc, []):
                key_p = (p["uri"], p["domainUri"], p["rangeUri"])
                if key_p not in seen_in:
                    seen_in.add(key_p)
                    incoming.append(p)
        direct_domains = {p["domainUri"] for p in incoming if p["rangeUri"] == cls}
        by_range[cls] = [p for p in incoming if p["rangeUri"] == cls or p["domainUri"] not in direct_domains]

    # --- value (datatype) properties ---------------------------------------
    # Kept in their OWN map rather than mixed into by_domain. Every consumer of
    # by_domain treats an entry's rangeUri as a class to walk to next; a value
    # edge has no such class, so mixing them would send the walk into xsd:string.
    values_by_domain: Dict[str, List[dict]] = {}
    for d_uri, prop_uri, range_uri in _datatype_edges(graph):
        values_by_domain.setdefault(d_uri, []).append(
            {"uri": prop_uri, "domainUri": d_uri, "rangeUri": range_uri}
        )

    by_value: Dict[str, List[dict]] = {}
    # A class can own a value property and no relation at all, in which case the
    # loops above never created a bucket for it.
    for cls in set(all_classes) | set(values_by_domain):
        # Inherited exactly like relations: `cmt:title` declared on Paper is
        # available on every subclass of Paper.
        ancestors = supers_map.get(cls, {cls})
        seen_v: Set[str] = set()
        vals: List[dict] = []
        for anc in sorted(ancestors):
            for p in values_by_domain.get(anc, []):
                # Deduplicated by PROPERTY, not by (property, domain): the same
                # attribute reached through two ancestors is one attribute.
                if p["uri"] in seen_v:
                    continue
                seen_v.add(p["uri"])
                vals.append(p)
        by_value[cls] = vals

    # `supers` is the same transitive closure the value/edge inheritance above
    # already walked. Kept rather than recomputed because callers outside this
    # module need it to tell a subclass from an unrelated class, and rebuilding
    # it per query means re-walking the hierarchy the index was built from.
    _INDEX_CACHE[key] = {"by_domain": by_domain, "by_range": by_range,
                         "by_value": by_value, "supers": supers_map}

    build_time = time.perf_counter() - build_start

    # Memory estimation: size of the index dicts
    index_size = sys.getsizeof(by_domain) + sys.getsizeof(by_range)
    for entries in by_domain.values():
        index_size += sys.getsizeof(entries)
        for p in entries:
            index_size += sys.getsizeof(p)
    for entries in by_range.values():
        index_size += sys.getsizeof(entries)
        for p in entries:
            index_size += sys.getsizeof(p)

    stats = {
        "graph": key,
        "build_time_s": round(build_time, 6),
        "num_classes": len(all_classes),
        "num_properties": len(all_props),
        "num_hierarchy_edges": sum(len(v) for v in supers_map.values()) - len(all_classes),
        "index_entries_outgoing": sum(len(v) for v in by_domain.values()),
        "index_entries_incoming": sum(len(v) for v in by_range.values()),
        "index_memory_bytes": index_size,
        "graph_triples": len(graph),
    }
    if INDEX_VERBOSE:
        print(f"🧩 build_index {stats}")

    return _INDEX_CACHE[key]


def ensure_index(graph: Graph) -> Dict[str, Dict[str, List[dict]]]:
    """Return the cached index for `graph`, building it on first use."""
    return build_index(graph)


# graph identifier -> the property URIs the index was able to give ends to
_USABLE_PROPS_CACHE: Dict[str, Set[str]] = {}


def usable_property_uris(graph: Graph) -> Set[str]:
    """Every property of this ontology that can actually appear in a path.

    build_index files a property under a class only once it has BOTH ends,
    from any of the three sources it reads: a declared rdfs:domain/rdfs:range,
    an owl:unionOf unfolded into its members, or an owl:Restriction. A property
    the index never filed has no ends at all, so no walk can reach it and no
    generated query can use it.

    They are not rare and they are not low-scoring: ekaw declares `partOf` and
    `hasPart` with neither domain nor range, and on "which abstract belongs to
    each invited talk?" those two were the 1st and 4th hit for "belongs to" —
    two of five slots spent on terms incapable of producing a single triple.

    Datatype properties come from `by_value`, which needs a domain only.
    """
    key = str(graph.identifier)
    cached = _USABLE_PROPS_CACHE.get(key)
    if cached is not None:
        return cached

    index = ensure_index(graph)
    usable: Set[str] = {
        entry["uri"]
        for bucket in ("by_domain", "by_range", "by_value")
        for entries in index.get(bucket, {}).values()
        for entry in entries
    }
    _USABLE_PROPS_CACHE[key] = usable
    return usable


# local name (lower) -> full URI, per ontology context
_LABEL_CACHE: Dict[str, Dict[str, str]] = {}


def build_label_index(graph: Graph) -> Dict[str, str]:
    """Map every class/property LOCAL NAME in this ontology back to its full URI.

    Path summaries are flattened to bare labels for the LLM filter, which throws
    the URIs away. This lets us resolve them again so the final SPARQL is built
    from real URIs instead of the generator guessing a namespace.
    """
    key = str(graph.identifier)
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return cached

    index = ensure_index(graph)
    labels: Dict[str, str] = {}

    def add(uri: str):
        name = uri.split("#")[-1].split("/")[-1].lower()
        # sorted iteration below keeps "first wins" deterministic across runs
        labels.setdefault(name, uri)

    for direction in ("by_domain", "by_range"):
        for cls in sorted(index[direction]):
            add(cls)
            for p in sorted(index[direction][cls], key=lambda d: (d["uri"], d["domainUri"], d["rangeUri"])):
                add(p["uri"])
                add(p["domainUri"])
                add(p["rangeUri"])

    # Value properties too: without them a datatype property named in a path
    # summary resolves to nothing and the attribute silently disappears again.
    # The rangeUri is deliberately NOT added — it is `xsd:string`, and mapping
    # the local name "string" to a URI would let it be read back as a class.
    for cls in sorted(index.get("by_value", {})):
        add(cls)
        for p in sorted(index["by_value"][cls], key=lambda d: d["uri"]):
            add(p["uri"])
            add(p["domainUri"])

    _LABEL_CACHE[key] = labels
    return labels


def index_stats(graph: Graph) -> Tuple[int, int]:
    """(number of classes with outgoing props, number with incoming props)."""
    index = _INDEX_CACHE.get(str(graph.identifier))
    if index is None:
        return (0, 0)
    return (
        sum(1 for v in index["by_domain"].values() if v),
        sum(1 for v in index["by_range"].values() if v),
    )


# ---------------------------------------------------------------------------
# O(1) lookups (same names / signatures / return shape as before)
# ---------------------------------------------------------------------------

def query_outgoing_object_properties(class_uri: str, graph: Graph):
    """[{uri, domainUri, rangeUri}] -- properties usable FROM `class_uri`.
    `domainUri` is the class the property was DECLARED on, which may be an
    ancestor of `class_uri` when the property is inherited."""
    index = ensure_index(graph)
    # copy: the entries are shared with the cache
    return list(index["by_domain"].get(str(class_uri), ()))


def query_incoming_object_properties(class_uri: str, graph: Graph):
    """[{uri, domainUri, rangeUri}] -- properties pointing INTO `class_uri`.
    `rangeUri` may be an ancestor of `class_uri` when the property is inherited."""
    index = ensure_index(graph)
    return list(index["by_range"].get(str(class_uri), ()))


def query_datatype_properties(class_uri: str, graph: Graph):
    """[{uri, domainUri, rangeUri}] -- VALUE properties readable off `class_uri`.

    `rangeUri` is a datatype (`xsd:string`) or None, never a class. These are
    terminal by nature: nothing continues past them, so they are looked up once
    per class instead of being walked."""
    index = ensure_index(graph)
    return list(index.get("by_value", {}).get(str(class_uri), ()))


def value_edges_for(class_uri: str, graph: Graph) -> List[Tuple[str, str, str]]:
    """[(class, property, datatype)] — the values readable off `class_uri`.

    Only what the ontology DECLARES. Nothing is synthesised here.

    This used to append `rdfs:label` whenever a class declared no name-like
    property of its own. The reasoning was sound on its face: 224 of the 302
    classes across the five ontologies declare no way of being named, worst in
    ekaw (0 of 73) and edas (6 of 103), while the loaded data carries
    `rdfs:label` on nearly every instance (ekaw 27,698, edas 25,986,
    conference 20,724, cmt 13,816, confof 4,473). So "what is this called" was
    unanswerable for those classes, and adding the edge made it answerable.

    It also made it UNAVOIDABLE, which is why it is gone. Because the offer sat
    on 224 classes, the generator met it constantly and took it constantly: for
    a question like "which subject does each paper deal with" it returned the
    subject's label rather than the subject, since a label was on offer and
    reads better. Agent 7 copies what it is handed, so an edge handed to it on
    three quarters of all classes is not an option, it is a default. Measured
    on the benchmark: 9 of 42 questions project something whose value is an
    identifier, and the label edge was what made those answers unmatchable.

    Two questions genuinely needed it — a paper title in the three sources that
    declare no title property, and the place a conference is held, whose class
    declares no datatype property at all. Both were rewritten to stop depending
    on it (see eval/questions/benchmark_audit.md), so no question now requires
    an edge the ontology does not declare.
    """
    return [(class_uri, prop["uri"], prop.get("rangeUri"))
            for prop in query_datatype_properties(class_uri, graph)]


def local_name(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1] if uri else ""


def query_ancestors(class_uri: str, graph: Graph) -> Set[str]:
    """Every class `class_uri` is a kind of, itself included.

    Reads the closure the index already built. Returns {class_uri} for a class
    the index does not know, so a caller comparing two classes finds no
    relationship rather than raising on an unindexed URI.
    """
    index = ensure_index(graph)
    return set(index.get("supers", {}).get(str(class_uri), {str(class_uri)}))
