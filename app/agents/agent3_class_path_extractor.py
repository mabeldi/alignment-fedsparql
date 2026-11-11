from typing import List, Dict, Tuple, Any
from tools.find_class_paths import (
    query_outgoing_object_properties,
    query_incoming_object_properties,
    value_edges_for,
)
from core.ontology_loader import get_graph
# Max depth for traversal to avoid infinite loops
MAX_DEPTH = 3

ClassPath = List[Tuple[str, str, str]]  # (subject, predicate, object)

# (class URI, property URI, datatype URI | None). Same shape as a path triple so
# downstream code can treat it uniformly, but the third slot is a DATATYPE, not a
# class — nothing may traverse it.
ValueEdge = Tuple[str, str, Any]


def value_edges(class_uri: str, ontology_id: str) -> List[ValueEdge]:
    """The value properties readable straight off `class_uri`.

    These are not walked and not chained: a value property ends the path by
    definition, so there is one flat list per class rather than a tree. Depth,
    cycles and `participated_classes` are all irrelevant here — that machinery
    exists to decide where a walk goes NEXT, and there is no next.
    """
    graph = get_graph().get_context("http://" + ontology_id)
    return value_edges_for(class_uri, graph)


def traverse_outgoing(class_uri: str, ontology_id: str, participated_classes, depth=0, visited=None, max_depth=MAX_DEPTH) -> List[ClassPath]:
    if visited is None:
        visited = set()
    if depth >= max_depth or class_uri in visited:
        return []

    visited.add(class_uri)
    outgoing_paths = []
    graph = get_graph().get_context('http://'+ontology_id)
    direct_props = query_outgoing_object_properties(class_uri, graph)
    for prop in direct_props:
        next_class = prop["rangeUri"]
        path = [(class_uri, prop["uri"], next_class)]
        outgoing_paths.append(path)

        if(next_class not in participated_classes):
            deeper_paths = traverse_outgoing(next_class, ontology_id, participated_classes, depth + 1, visited.copy(), max_depth)
            for dp in deeper_paths:
                outgoing_paths.append(path + dp)

    return outgoing_paths


def traverse_incoming(class_uri: str, ontology_id: str, participated_classes, depth=0, visited=None, max_depth=MAX_DEPTH) -> List[ClassPath]:
    if visited is None:
        visited = set()
    if depth >= max_depth or class_uri in visited:
        return []
    

    visited.add(class_uri)
     #print("for class ", class_uri)
    incoming_paths = []
    graph = get_graph().get_context('http://'+ontology_id)
    direct_props = query_incoming_object_properties(class_uri, graph)

    for prop in direct_props:
        prev_class = prop["domainUri"]
        path = [(prev_class, prop["uri"], class_uri)]
        #print("adding: ",[(prev_class, prop["uri"], class_uri)])
        incoming_paths.append(path)
        if(prev_class not in participated_classes):
            deeper_paths = traverse_incoming(prev_class, ontology_id, participated_classes, depth + 1, visited.copy(), max_depth)
            for dp in deeper_paths:
                incoming_paths.append(dp + path)
    return incoming_paths


def build_class_paths_for_matches(matches: Dict[str, List[Dict[str, Any]]], participated_classes: set[any],
                                  participated_properties: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    # `participated_properties` is accepted and IGNORED, so main.py can keep one
    # call shape for both builders and USE_CONNECTOR_PATHS stays a one-line flip.
    # Seeding the matched relations is connector-only: this builder terminates on
    # DEPTH rather than on a target, so it already emits every edge incident to a
    # matched class and has nothing a seed could add.
    class_paths = []
    # The same class can be returned under several keywords (Agent 2 searches per
    # keyword), so traverse each class only once — otherwise we pay for the walk
    # twice and emit duplicate path blocks downstream.
    seen_classes = set()
    for label, candidates in matches.items():
        for candidate in candidates:
            payload = candidate["payload"]
            if payload["type"] != "Class":
                continue

            class_uri = payload["uri"]
            source = payload["source"]
            if (class_uri, source) in seen_classes:
                continue
            seen_classes.add((class_uri, source))


            outgoing = traverse_outgoing(class_uri, source, participated_classes)
            incoming = traverse_incoming(class_uri, source, participated_classes)
            class_paths.append({
                "class": class_uri,
                "source": source,
                "outgoing": outgoing,
                "incoming": incoming,
                "values": value_edges(class_uri, source),
            })
    
    return class_paths
