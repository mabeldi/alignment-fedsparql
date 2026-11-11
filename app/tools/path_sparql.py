"""
Deterministic rendering of class paths into SPARQL triple patterns.

Agent 5 flattens paths to readable label chains ("A -[p]-> B") so the LLM path
filter can reason about them cheaply — but that strips every URI. Handing those
chains to the query generator forces it to reinvent full URIs from bare labels
and a guessed namespace; in practice it skips the paths entirely and only copies
the patterns that arrive ready-made, producing queries whose variables are never
joined (a silent cartesian product).

So we resolve the labels back to URIs here and emit the triples ourselves. Same
trust boundary as the alignment fragments: code renders, the LLM only assembles.

Variable naming is alignment-aware — classes that the alignment declares
equivalent share one variable, so the UNION branches of a federated query line up
on the same column instead of each projecting its own.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from core.ontology_loader import get_graph
from tools.find_class_paths import build_label_index
from tools.alignment_store import get_alignment_store, local_name, source_of

# One hop of a flattened chain: "-[p]->" (forward) or "<-[p]-" (backwards, i.e.
# the triple runs from the NEXT class into the current one). Agent 5 emits the
# backwards form for connector paths that traverse an edge against its direction.
_STEP = re.compile(r"\s*(<-\[|-\[)\s*([^\]]+?)\s*(\]->|\]-)\s*")


def _parse_chain(path_str: str) -> Optional[Tuple[List[str], List[str], List[bool]]]:
    """"A -[p]-> B <-[q]- C" -> (['A','B','C'], ['p','q'], [True, False]).

    Returns (nodes, props, forwards) or None when the string is not a well-formed
    chain — a mismatched arrow ("-[p]-") means the filter mangled the notation and
    we must not guess which way the triple runs.
    """
    s = path_str.strip()
    steps = list(_STEP.finditer(s))
    if not steps:
        return None

    nodes = [s[:steps[0].start()].strip()]
    props: List[str] = []
    forwards: List[bool] = []

    for i, m in enumerate(steps):
        opener, prop, closer = m.group(1), m.group(2).strip(), m.group(3)
        if opener == "-[" and closer == "]->":
            forwards.append(True)
        elif opener == "<-[" and closer == "]-":
            forwards.append(False)
        else:
            return None
        props.append(prop)
        end = steps[i + 1].start() if i + 1 < len(steps) else len(s)
        nodes.append(s[m.end():end].strip())

    if not all(nodes) or not all(props):
        return None
    return nodes, props, forwards


def _labels_for(source: str) -> Dict[str, str]:
    return build_label_index(get_graph().get_context("http://" + source))


class VarNamer:
    """Assigns one SPARQL variable per concept, shared across aligned classes.

    A UNION over several sources only produces ONE list of answers when every
    branch binds the same variable. So the variable is chosen per GROUP of
    aligned classes, not per class.

    WHY THE GROUP IS BUILT FIRST
    ----------------------------
    The earlier version reused a counterpart's name only if that counterpart had
    already been named, and took the first such match. That is order-dependent
    and not transitive: A = C and B = C did not put A and B together, because
    whichever of them came first was named before C existed.

    Measured on "list all the papers", where the alignment makes edas:Paper
    equivalent to the Paper class of all four other ontologies — one group, and
    the correct answer is one variable in five UNION branches. What came out was:

        ?paper            a conference:Paper .
        ?paper2           a cmt:Paper .
        ?paperfullversion a confOf:Paper .
        ?paper2           a edas:Paper .
        ?paper2           a ekaw:Paper .

    conference:Paper was named first, when nothing it linked to had a name yet;
    edas:Paper, the one class that links to all the others, was reached fifth and
    could only take one of the three names already in use. The query that
    followed selected three columns, each mostly empty, instead of one list. It
    also named confOf:Paper after a class in another ontology, which reads as a
    bug even though the code did exactly what it said.

    So the group is resolved by walking the correspondences transitively BEFORE
    the name is chosen. Only `simple` equivalences merge: a `restriction_*`
    correspondence holds under an extra condition, so its two sides are not the
    same set of things and must not share a variable.
    """

    def __init__(self, loaded_sources: Set[str]):
        self.loaded = loaded_sources
        self.store = get_alignment_store()
        self._by_class: Dict[str, str] = {}
        self._used: Set[str] = set()
        self._group_of: Dict[str, str] = {}      # class URI -> group key
        self._group_var: Dict[str, str] = {}     # group key -> its variable
        self._claimed: Set[Tuple[str, str]] = set()   # (group key, source)

    def _group(self, class_uri: str) -> str:
        """Key of the group `class_uri` belongs to: every class reachable from it
        through `simple` correspondences, itself included.

        Cached for every member at once — the walk from any member finds the same
        set, so doing it once per group is enough.
        """
        if class_uri in self._group_of:
            return self._group_of[class_uri]

        members = {class_uri}
        queue = [class_uri]
        while queue:
            for uri, mapping in self.store.counterparts(queue.pop(), self.loaded):
                if mapping.kind == "simple" and uri not in members:
                    members.add(uri)
                    queue.append(uri)

        # Lowest URI as the key: any member must yield the same key, and the walk
        # can start anywhere, so the choice has to depend on the set alone.
        key = min(members)
        for uri in members:
            self._group_of[uri] = key
        return key

    def var_for(self, class_uri: str) -> str:
        if class_uri in self._by_class:
            return self._by_class[class_uri]

        key = self._group(class_uri)
        claim = (key, source_of(class_uri))
        if claim in self._claimed:
            # This ontology already spent the group's variable on another of its
            # classes. Sharing it again would put two type lines on one variable
            # inside a single SERVICE block, demanding one instance be both
            # classes at once — a silent, empty result.
            var = self._fresh(local_name(class_uri))
        else:
            var = self._group_var.get(key) or self._fresh(local_name(class_uri))
            self._group_var[key] = var
            self._claimed.add(claim)

        self._by_class[class_uri] = var
        return var

    def _fresh(self, label: str) -> str:
        base = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower() or "x"
        var, n = base, 2
        while var in self._used:
            var, n = f"{base}{n}", n + 1
        self._used.add(var)
        return var


def render_path(path_str: str, source: str, namer: VarNamer) -> Optional[str]:
    """One flattened path -> SPARQL triples, or None if it cannot be resolved.

    Returning None is deliberate: a label that does not exist in this ontology
    means the path filter invented it, and an unresolvable name must never reach
    the query as a guessed URI.
    """
    parsed = _parse_chain(path_str)
    if parsed is None:
        return None
    nodes, props, forwards = parsed

    labels = _labels_for(source)
    node_uris = [labels.get(n.lower()) for n in nodes]
    prop_uris = [labels.get(p.lower()) for p in props]
    if any(u is None for u in node_uris) or any(u is None for u in prop_uris):
        return None

    lines: List[str] = []
    subj = namer.var_for(node_uris[0])
    lines.append(f"?{subj} a <{node_uris[0]}> .")
    for i, prop in enumerate(prop_uris):
        obj_uri = node_uris[i + 1]
        obj = namer.var_for(obj_uri)
        # A backwards hop is the same triple with the roles swapped, so the join
        # variable is preserved and the chain still connects.
        if forwards[i]:
            lines.append(f"?{subj} <{prop}> ?{obj} .")
        else:
            lines.append(f"?{obj} <{prop}> ?{subj} .")
        lines.append(f"?{obj} a <{obj_uri}> .")
        subj = obj
    # dedup while preserving order (chains revisit the same class triples)
    return "\n".join(dict.fromkeys(lines))


def path_edges(path_str: str, source: str) -> Optional[List[Tuple[str, str, str]]]:
    """One flattened chain -> its resolved (subject, property, object) URI edges.

    Same label resolution and direction handling as `render_path` — a backwards
    hop comes back with the roles already swapped, so every tuple is in real
    triple order — but it returns the EDGES instead of rendering variables.

    That distinction is the whole point: the ~1000 chains Agent 6 keeps describe
    only ~200 distinct edges, because a depth-3 walk re-states its first hop in
    every chain that starts with it. Callers that want an adjacency list (each
    relation stated once) need the edges, not the chains.

    Returns None under exactly the same condition as `render_path` — an
    unresolvable label means the path filter invented a name, and an invented
    name must never become an edge in the prompt.
    """
    parsed = _parse_chain(path_str)
    if parsed is None:
        return None
    nodes, props, forwards = parsed

    labels = _labels_for(source)
    node_uris = [labels.get(n.lower()) for n in nodes]
    prop_uris = [labels.get(p.lower()) for p in props]
    if any(u is None for u in node_uris) or any(u is None for u in prop_uris):
        return None

    edges: List[Tuple[str, str, str]] = []
    for i, prop in enumerate(prop_uris):
        a, b = node_uris[i], node_uris[i + 1]
        edges.append((a, prop, b) if forwards[i] else (b, prop, a))
    return edges


