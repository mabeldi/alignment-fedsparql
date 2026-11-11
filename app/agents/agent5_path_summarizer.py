

from typing import List, Tuple, Dict, Any, Set

from agents.agent3_class_path_extractor import value_edges

ClassPath = List[Tuple[str, str, str]]

def extract_label(uri: str) -> str:
    """Extract readable label from a URI."""
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri

def _chain_start(path: ClassPath) -> str:
    """The node the chain should be read from.

    Connector paths (Agent 3 connector mode) may traverse an edge against its
    declared direction, so the first triple's subject is not necessarily where the
    walk began. Pick the endpoint of the first triple that the second one does not
    touch; single-edge paths keep the declared subject-first reading.
    """
    s, _, o = path[0]
    if len(path) == 1:
        return s
    nxt = {path[1][0], path[1][2]}
    if s in nxt and o not in nxt:
        return o
    return s


def flatten_path(path: ClassPath) -> str:
    """Flatten full path into readable string like: A -[p1]-> B <-[p2]- C

    `<-[p]-` marks an edge traversed backwards, i.e. the triple is (C, p, B). A
    purely forward rendering cannot express that: on a mixed-direction path it
    printed the same class twice and silently dropped the other endpoint.

    Returns "" if the triples do not form a connected chain, so an unreadable
    path is dropped instead of being rendered into a wrong one.
    """
    if not path:
        return ""

    node = _chain_start(path)
    parts = [f"{extract_label(node)}"]
    for s, p, o in path:
        if s == node:                                   # traversed forward
            parts.append(f"-[{extract_label(p)}]-> {extract_label(o)}")
            node = o
        elif o == node:                                 # traversed backwards
            parts.append(f"<-[{extract_label(p)}]- {extract_label(s)}")
            node = s
        else:
            return ""
    return " ".join(parts)

def filter_deepest_paths(paths: List[ClassPath]) -> List[ClassPath]:
    """
    Keep only the deepest (longest) paths that are not strict prefixes of others.
    """
    def is_prefix(shorter, longer):
        if len(shorter) >= len(longer):
            return False
        return shorter == longer[:len(shorter)]
    
    result = []
    for i, p in enumerate(paths):
        if not any(is_prefix(p, other) for j, other in enumerate(paths) if j != i):
            result.append(p)
    return result

def with_prefixes(paths: List[ClassPath]) -> List[ClassPath]:
    """Every path, preceded by each shorter walk that begins it.

    `filter_deepest_paths` drops a path that is a strict prefix of a longer one,
    on the reasoning that the longer path already contains it. It does — as
    TEXT. It does not as a CHOICE: the next stage picks whole chains, so a head
    hop that only ever appears with a tail attached can only be kept together
    with that tail.

    Measured on "which workshops belong to a conference, and when is their
    submission deadline?". confOf reaches the deadline through
    `hasAdministrativeEvent`, and the filter offered it only as:

        Workshop -[hasAdministrativeEvent]-> Administrative_event -[follows]-> ...
        Workshop -[hasAdministrativeEvent]-> Administrative_event -[parallel_with]-> ...

    Nothing in that question asks which event follows which, and Agent 6 is told
    to keep only paths carrying information the question asks for and to keep the
    smallest sufficient set. So keeping the relation it needed meant breaking its
    own two rules, and whether it did was a coin flip — it kept them on two runs
    and dropped all six on a third, which cost that run the entire confOf half of
    the answer. Ontologies whose paths came with usable one-hop forms (ekaw,
    conference) never wavered.

    Restoring the prefixes costs little because the flattened strings deduplicate:
    the same first hop shared by eleven chains is one extra line. Measured over
    the five ontologies of that question: 265 -> 319 chains, and cmt's 179 grew by
    20 despite having the longest list.

    No new EDGE is introduced — every prefix is made of triples the longer path
    already carried — so Agent 7's vocabulary cannot widen. Only what Agent 6 is
    able to select does.

    Shorter walks come first, so the most reusable form is the one read first and
    the one that survives if a caller ever caps the list.
    """
    prefixes = [path[:k] for path in paths for k in range(1, len(path))]
    return sorted(prefixes, key=len) + list(paths)


def summarize_paths(paths: List[ClassPath], max_per_class: int = 3) -> List[str]:
    """Summarize the deepest non-overlapping paths and their beginnings, flattened.

    Deduplicates before capping: the connector agent files each connector under
    both classes it joins, so the same chain can reach one bucket twice, and a
    prefix restored by `with_prefixes` is often already a path in its own right.
    """
    deepest = filter_deepest_paths(paths)
    flattened = (flatten_path(path) for path in with_prefixes(deepest))
    unique = list(dict.fromkeys(f for f in flattened if f))
    return unique[:max_per_class]

def _touched_classes(paths: List[ClassPath]) -> Set[str]:
    """Every class URI these raw (unflattened) paths pass through, either end."""
    classes: Set[str] = set()
    for path in paths:
        for s, _, o in path:
            classes.add(s)
            classes.add(o)
    return classes


def summarize_class_paths(class_paths: List[Dict[str, Any]], max_per_class: int = 3) -> List[Dict[str, Any]]:
    summaries = []
    for cp in class_paths:
        class_name = extract_label(cp["class"])
        source = cp.get("source")
        summarized = {
            "label": class_name,
            "class": cp["class"],
            "source": source,
            "outgoing": summarize_paths(cp.get("outgoing", []), max_per_class),
            "incoming": summarize_paths(cp.get("incoming", []), max_per_class),
            # Passed through as TRIPLES, not flattened into `A -[p]-> B` chains
            # like the two lists above. Flattening exists so a multi-hop walk can
            # be read as one line and parsed back by `path_edges`; a value edge is
            # a single triple whose object is a datatype, and pushing it through
            # that round trip would only give the parser a chance to read
            # `xsd:string` back as a class.
            "values": list(cp.get("values", [])),
        }
        summaries.append(summarized)

    # A class that only ever shows up as a path's DESTINATION never got a block
    # above (retrieval never named it directly), so Agent 6 saw a hop like
    # `Workshop -[hasAdministrativeEvent]-> Administrative_event` with nothing
    # to say whether Administrative_event carries anything the question asked
    # for. Give every such class the same value-only block a retrieved class
    # gets, so Agent 6 can read its attributes right next to the chain that
    # reaches it instead of guessing from the relation's name alone.
    known = {(cp.get("source"), cp["class"]) for cp in class_paths if cp.get("class")}
    added: Set[Tuple[str, str]] = set()
    for cp in class_paths:
        source = cp.get("source")
        if not source:
            continue
        reachable = _touched_classes(cp.get("outgoing", [])) | _touched_classes(cp.get("incoming", []))
        reachable.discard(cp.get("class"))
        for uri in reachable:
            key = (source, uri)
            if key in known or key in added:
                continue
            added.add(key)
            summaries.append({
                "label": extract_label(uri),
                "class": uri,
                "source": source,
                "outgoing": [],
                "incoming": [],
                "values": value_edges(uri, source),
            })
    return summaries