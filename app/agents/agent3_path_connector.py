"""
Agent 3 — CONNECTOR variant (experimental, parallel to agent3_class_path_extractor).

Instead of walking the graph blindly from each matched class up to a fixed depth
and emitting every chain (see build_class_paths_for_matches), this finds the paths
that actually CONNECT the matched classes to each other.

Difference in one line:
  - old: termination = depth        -> exhaustive, noisy, Agent 6 must prune hard
  - new: termination = target       -> minimal connecting subgraph, relevant by construction

For each pair of matched classes in the SAME ontology we BFS for the SHORTEST
connectors (there can be several of equal length — e.g. Paper is linked to Author
by hasAuthor AND writePaper AND submitPaper; we keep all of them). Longer detours
are dropped. A pair whose shortest connector merely passes THROUGH a third matched
class is skipped, because the two sub-connectors are found on their own — so the
result is a minimal connected subgraph, not a redundant all-pairs mesh.

Output shape is identical to build_class_paths_for_matches, so summarizer / Agent 6
/ Agent 7 consume it unchanged and the pipeline can switch back with one flag. Each
connector is filed under BOTH classes it joins, since every class block is read on
its own downstream. Unlike the exhaustive walk a connector may traverse an edge
against its declared direction, which Agent 5 renders as `A <-[p]- B`.

This version is INTRA-ONTOLOGY only. Cross-ontology connection (via alignment
edges = SERVICE boundaries) is a deliberate follow-up.
"""

from collections import deque
from itertools import combinations
from typing import Dict, List, Tuple, Any, Set

from rdflib import OWL, RDFS, URIRef

from tools.find_class_paths import (
    query_outgoing_object_properties,
    query_incoming_object_properties,
)
from core.ontology_loader import get_graph
from agents.agent3_class_path_extractor import traverse_outgoing, traverse_incoming, value_edges

# Guard against pathological walks; NOT the termination rule (target is).
MAX_HOPS = 3
# When two classes are only weakly related, BFS returns many equal-length bridges
# (no semantics to prefer one). Cap how many we forward per pair so a distant pair
# cannot flood the prompt; Agent 6 still ranks the survivors. 0 = unlimited.
MAX_CONNECTORS_PER_PAIR = 8
# Depth of the per-class fallback walk (see build_connector_paths_for_matches).
# 0 disables it: traverse_* returns [] as soon as depth >= max_depth, so the class
# keeps only its `values` (datatype properties) and no object properties at all.
#
# The bet being tested: if a question needs a second class, retrieval matched that
# class too, so the connector search already covers the relationship — and a class
# left with no connector is a class the question does not relate to anything.
#
# Measured cost of the alternatives, chains emitted per fallback class:
#   depth 1 -> 2.3 (confof) .. 9.1 (ekaw)
#   depth 2 -> 4.7 (confof) .. 49.5 (ekaw), worst single class 206 (cmt)
# Known risk at 0: ekaw declares 33 object properties and ZERO datatype ones, so a
# lone ekaw class arrives with nothing at all. Raise to 1 if that shows up.
FALLBACK_MAX_DEPTH = 1

Triple = Tuple[str, str, str]      # (subject, predicate, object) — correctly oriented
Path = List[Triple]


def _neighbours(node: str, graph):
    """Yield (property_uri, other_class, direction) for every object-property edge
    incident to `node`. 'out' => node is the subject, 'in' => node is the object."""
    for p in query_outgoing_object_properties(node, graph):   # node -prop-> range
        yield p["uri"], p["rangeUri"], "out"
    for p in query_incoming_object_properties(node, graph):    # domain -prop-> node
        yield p["uri"], p["domainUri"], "in"


def _triple(node: str, prop: str, other: str, direction: str) -> Triple:
    """Return the edge as a correctly oriented (s, p, o) triple, so downstream
    rendering never has to guess direction."""
    return (node, prop, other) if direction == "out" else (other, prop, node)


# Top-of-hierarchy classes. Every class is one, so accepting them as a landing
# point would make every pair of classes "connected" through any property at all.
_GENERIC_CLASSES = {str(OWL.Thing), str(RDFS.Resource)}

# graph identifier -> {class URI: its DIRECT named superclasses}
_PARENTS_CACHE: Dict[str, Dict[str, Set[str]]] = {}


def _parents(graph) -> Dict[str, Set[str]]:
    """Direct rdfs:subClassOf links of one ontology, built once and reused.

    Blank-node superclasses (restrictions, unions) are skipped: they name no
    class, so they cannot be a landing point or a target.
    """
    key = str(graph.identifier)
    table = _PARENTS_CACHE.get(key)
    if table is None:
        table = {}
        for s, _, o in graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                table.setdefault(str(s), set()).add(str(o))
        _PARENTS_CACHE[key] = table
    return table


def _target_classes(b: str, graph) -> Set[str]:
    """Landing on any of these counts as reaching `b`.

    A property declares its range as the GENERAL class, while the question asks
    about a SPECIFIC one: `organises` has range Event, and Workshop is an Event.
    Comparing URIs as text reports no path between Person and Workshop even
    though the ontology plainly connects them, so `b` and every class ABOVE it
    are accepted, and the landing class is then rewritten back down to `b` (the
    query narrows it with a `?x a b .` line, which is what makes that sound).

    Only this direction. The reverse — landing on a class BELOW `b` — is also
    true in logic (every Reviewer is a Person) but useless in practice: it makes
    every specialisation of a hub class an answer and floods the result.
    """
    parents = _parents(graph)
    out = {b}
    queue = deque(parents.get(b, ()))
    while queue:
        p = queue.popleft()
        if p in out:
            continue
        out.add(p)
        queue.extend(parents.get(p, ()))
    return out - _GENERIC_CLASSES


def shortest_connectors(a: str, b: str, graph, max_hops: int = MAX_HOPS) -> List[Path]:
    """All SHORTEST oriented-triple paths from class `a` to class `b`.

    Returns [] if they are not connected within max_hops. Each returned path is a
    list of (s, p, o) triples; the traversal may mix edge directions, but every
    triple is emitted in its true subject->object order.

    Landing on a class that `b` sits under counts as reaching `b` (see
    `_target_classes`), and the closing triple is written with `b` in place of
    that general class — so every returned path really does end at `b`.
    """
    # queue items: (current_node, path_of_triples, chain_of_nodes_incl_current)
    queue = deque([(a, [], [a])])
    best_len = None
    results: List[Path] = []
    seen_paths: Set[Tuple[Triple, ...]] = set()
    targets = _target_classes(b, graph)

    while queue:
        node, path, chain = queue.popleft()
        # BFS is level-ordered; once we know the shortest length, stop growing past it.
        if best_len is not None and len(path) >= best_len:
            continue
        for prop, other, direction in _neighbours(node, graph):
            if other in chain:                       # no cycles within a single path
                continue
            if other in targets:
                # Rewrite the general landing class down to the class actually
                # asked for; two properties can land on two different ancestors
                # of `b` and collapse onto the same triple, hence the dedupe.
                triple = _triple(node, prop, b, direction)
                if best_len is None:
                    best_len = len(path) + 1
                if len(path) + 1 == best_len:
                    candidate = tuple(path + [triple])
                    if candidate not in seen_paths:
                        seen_paths.add(candidate)
                        results.append(list(candidate))
                continue                             # never walk past the target
            if len(path) + 1 < max_hops:
                triple = _triple(node, prop, other, direction)
                queue.append((other, path + [triple], chain + [other]))
    return results


def _orient_from(path: Path, cls: str) -> Path:
    """Re-order the triple list so the chain reads starting at `cls`.

    BFS returns each connector as walked from the pair's FIRST class, so the second
    class needs the same edges in reverse order — otherwise the flattener starts the
    chain at the wrong end. The triples themselves are never rewritten, only their
    order; each one keeps its true subject->object orientation.
    """
    if not path or path[0][0] == cls or path[0][2] == cls:
        return path
    return list(reversed(path))


def _bucket_for(path: Path, cls: str) -> str:
    """Which block the (already oriented) path belongs in for `cls`: 'outgoing'
    when the first edge leaves `cls` as its subject, 'incoming' when it points in."""
    return "outgoing" if path[0][0] == cls else "incoming"


def _passes_through(path: Path, a: str, b: str, matched: Set[str]) -> bool:
    """True if any INTERMEDIATE class on the path (not the endpoints) is itself a
    matched class — then the pair is redundant (covered by its sub-connectors)."""
    nodes = {s for s, _, _ in path} | {o for _, _, o in path}
    intermediates = nodes - {a, b}
    return bool(intermediates & matched)


def build_connector_paths_for_matches(matches: Dict[str, List[Dict[str, Any]]],
                                      participated_classes: Set[Any],
                                      participated_properties: List[Dict[str, Any]] = None,
                                      max_per_pair: int = MAX_CONNECTORS_PER_PAIR
                                      ) -> List[Dict[str, Any]]:
    """Connector-based replacement for build_class_paths_for_matches.

    The matched Class set used as connection targets is derived here from the matches
    themselves (we must not treat ObjectProperty URIs as targets); the wider
    `participated_classes` set is passed to the fallback walk, whose early-stop rule
    is the exhaustive builder's.

    `participated_properties` (see main.collect_participated_properties) SEEDS the
    search with the relations retrieval actually matched. Everything above rebuilds
    relations from the graph and never reads the matched properties, so whether a
    matched relation survives into the prompt was decided by BFS discovery order
    and the `max_per_pair` cut applied to it: Conference -> Administrator in cmt
    yields 14 equal-length connectors, 8 are forwarded, and the matched one can be
    among the 6 truncated. Seeding puts it FIRST, so the cut can never reach it.

    Note what does NOT change: the cap stays where it is. The guarantee comes from
    ORDERING, not from a bigger budget, so a 14-connector pair still cannot flood
    the prompt — the seeded edge simply takes the first of the 8 slots.

    None or an empty list restores the previous behaviour exactly.
    """
    # source -> ordered unique Class URIs matched in that ontology
    by_source: Dict[str, List[str]] = {}
    seen: Set[Tuple[str, str]] = set()
    for candidates in matches.values():
        for c in candidates:
            payload = c.get("payload") or {}
            if payload.get("type") != "Class":
                continue
            uri, source = payload.get("uri"), payload.get("source")
            if not uri or not source or (uri, source) in seen:
                continue
            seen.add((uri, source))
            by_source.setdefault(source, []).append(uri)

    # (source, {class a, class b}) -> the matched edges joining that pair.
    # Keyed on a frozenset because a pair is unordered here: the loop below draws
    # (a, b) from `combinations`, which follows retrieval order, while the edge is
    # stored in its true domain -> range direction.
    #
    # Only `kind == "object"` rows are seeded. A value row's range is a datatype,
    # so it joins no pair and there is nowhere to file it; its contribution is
    # already made upstream, by its domain being on the matched class list, which
    # is what makes `value_edges` below pick the attribute up.
    seeds: Dict[Tuple[str, frozenset], List[Triple]] = {}
    matched_by_source = {s: set(uris) for s, uris in by_source.items()}
    for prop in participated_properties or []:
        if prop.get("kind") != "object":
            continue
        source, uri = prop.get("source"), prop.get("uri")
        domain, range_ = prop.get("domain"), prop.get("range")
        if not (source and uri and domain and range_):
            continue
        # Both ends must be matched classes of that ontology, or there is no pair
        # in `combinations` to file the edge under and it would sit unread. Skipped
        # silently: it is a normal outcome, not a fault — Agent 4 can keep a
        # relation whose other end no filter or recovery step ever restored.
        matched_here = matched_by_source.get(source, set())
        if domain not in matched_here or range_ not in matched_here:
            continue
        bucket = seeds.setdefault((source, frozenset((domain, range_))), [])
        triple = (domain, uri, range_)
        if triple not in bucket:
            bucket.append(triple)

    class_paths: List[Dict[str, Any]] = []

    for source, class_uris in by_source.items():
        graph = get_graph().get_context("http://" + source)
        matched = set(class_uris)

        # One entry per matched class.
        # `values` is filled here and never touched by the connector search: a
        # value property joins nothing, so it can neither help nor hinder finding
        # the shortest connector between two classes. It rides along so the class
        # arrives at Agent 7 with its attributes attached.
        entries: Dict[str, Dict[str, Any]] = {
            uri: {"class": uri, "source": source, "outgoing": [], "incoming": [],
                  "values": value_edges(uri, source)}
            for uri in class_uris
        }

        for a, b in combinations(class_uris, 2):
            connectors = [
                path for path in shortest_connectors(a, b, graph)
                if not _passes_through(path, a, b, matched)
            ]
            # Seeded edges go in FRONT of the BFS result, before the cut. Order is
            # the whole mechanism: everything after this point reads the list
            # positionally, so first place is the only position `max_per_pair`
            # cannot take away.
            #
            # `_passes_through` is deliberately not applied to them. A seed is one
            # triple between the two endpoints, so it has no intermediate node to
            # test and the pruning could not fire — but a seed must reach the
            # prompt whatever happens, so it must not be arranged so that a filter
            # even gets the chance.
            seeded = seeds.get((source, frozenset((a, b))))
            if seeded:
                # A seeded edge is MOVED to the front, not merely inserted when
                # missing. The BFS almost always finds it too — but finding it at
                # position 9 of 14 is exactly the situation the seed exists to fix,
                # so "skip it, it is already there" would hand the cut the edge
                # back. Its BFS copy is dropped in the same step, on the exact
                # triple, so no edge is stated twice and no slot is spent twice.
                hoisted = {(t,) for t in seeded}
                connectors = ([[t] for t in seeded]
                              + [path for path in connectors if tuple(path) not in hoisted])
            if max_per_pair:
                connectors = connectors[:max_per_pair]
            # File every connector under BOTH endpoints, oriented and bucketed from
            # each. A connector describes the relationship of both classes it joins,
            # and Agent 5/6/7 read each class block on its own — nothing concatenates
            # them — so filing it only under the pair's first class left the other
            # class (and always the last one in class_uris) looking relationless.
            for path in connectors:
                for cls in (a, b):
                    oriented = _orient_from(path, cls)
                    entries[cls][_bucket_for(oriented, cls)].append(oriented)

        # A class can still end up with nothing: it is the only match in its ontology
        # (no pair to connect it to — common for alignment-injected classes), or every
        # connector it took part in was pruned as passing through a third matched
        # class. Fall back to a bounded exhaustive walk so Agent 7 never has to type a
        # class it knows no relationship for.
        for uri, entry in entries.items():
            if entry["outgoing"] or entry["incoming"]:
                continue
            entry["outgoing"] = traverse_outgoing(
                uri, source, participated_classes, max_depth=FALLBACK_MAX_DEPTH)
            entry["incoming"] = traverse_incoming(
                uri, source, participated_classes, max_depth=FALLBACK_MAX_DEPTH)

        class_paths.extend(entries.values())

    return class_paths
