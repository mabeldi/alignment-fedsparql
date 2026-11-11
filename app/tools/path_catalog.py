"""
Factored rendering of one ontology's chains for the Agent 6 prompt.

WHY this module exists
----------------------
Agent 5 hands Agent 6 a flat list of flattened chains, one line each. The same
relation is restated in every chain that uses it, so the list grows with the
NUMBER OF WALKS while the thing being described — the relations — grows far more
slowly. Measured over the 43 saved runs in `eval/harness/runs*`, per ontology:

    11,482 chain lines  describing  5,947 distinct relations

and the multiplication is a cross product: one class pair in CJ-01 produced 89
equally short chains built from 38 relations, because each hop offers several
predicates and every combination becomes its own line.

So state each relation ONCE, number it, and describe a multi-hop walk as the
sequence of numbers it uses:

    edges:
      ClassA>ClassB: [1] relOne  [2] relTwo  [3] <relThree
    routes:
      ClassA>ClassB>ClassC: 1>7  2>7

Three choices, each paying for itself:

1. GROUPED BY CLASS PAIR. All relations joining the same two classes share one
   heading, so the two class names are written once instead of once per
   predicate. This is also the grouping rule 3 of the prompt asks the model to
   apply ("the same two classes linked by a different property that means the
   same thing counts as one connection") — it can now see that set as a set
   instead of reconstructing it from a dozen scattered lines.

2. INVERSE DIRECTIONS MERGED. `ClassA -[p]-> ClassB` and `ClassB -[q]-> ClassA`
   are the same connection travelled both ways, and rule 3 says so. They used to
   be two unrelated lines; now they are one heading, and `<` marks the predicate
   whose triple runs against it.

3. SINGLE-HOP WALKS ARE NOT ROUTES. A one-edge chain says exactly what the edges
   block already said. Emitting it again is the whole reason the earlier
   prototype of this format came out LARGER than the flat list on small inputs
   (CJ-05: 34 chains, all single-hop, 1,411 -> 1,784 characters). Routes here are
   multi-hop only, so a catalog with no multi-hop walk renders as the edges block
   alone and is strictly smaller than the chain list it replaces.

WHAT THE MODEL SELECTS
----------------------
Numbers, same as before — the arrow notation is direction-carrying and a model
that rewords it silently deletes the path (see `tools/path_sparql._parse_chain`),
so nothing selectable is ever echoed back as prose. But the unit changed: it used
to be a whole chain, and it is now a single relation, plus optionally a route
token like `1>7` for a walk that must stay joined.

That is a real gain in reachable choices, not only in size. `with_prefixes`
(agent5) exists because a hop that only ever appeared with a tail attached could
not be kept without the tail; it restores head hops, but a hop in the MIDDLE of a
walk still had no standalone form. Measured over the same 43 runs, 3,761 of the
5,947 relations had no single-hop chain of their own — nearly two thirds were
only selectable together with something else. Numbering the relations makes every
one of them individually selectable.

ROUND TRIP
----------
`resolve()` maps a selection back to chain strings, which is the contract
everything downstream of Agent 6 consumes. A selected relation resolves to its
own one-edge chain (`flatten_path` always renders a single-edge path
subject-first, so this is character-identical to the chain Agent 5 would have
produced); a route token resolves to the exact chain string it was built from.
`selection_for()` is the inverse and exists so the round trip can be asserted:
every chain Agent 6 kept on a saved run maps to a selection that resolves back to
that same chain, with no chain added and none lost.
"""

import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# The chain parser, not a second copy of it. A private import on purpose: two
# readers of the `A -[p]-> B` notation that can drift apart is exactly the bug
# that notation keeps producing, and `path_edges` downstream uses this one.
from tools.path_sparql import _parse_chain

# (subject label, property label, object label), already in true triple order.
Edge = Tuple[str, str, str]

# Emit the `routes` block. ON = the format described above. OFF = the edges block
# alone, and the model selects only relations.
#
# The case for OFF: a route is DERIVED data. Every relation it names is already
# in the edges block with both its class names, so "these two hops join" is
# readable off those class names, and Agent 7 never sees a route at all —
# `prompt_format.collect_edges` breaks every kept chain back into single edges
# before the query generator is shown anything. It is the same argument
# `prompt_format.INCLUDE_REVERSE_INDEX` is switched off for: restating derived
# facts in a second notation costs a 20B local model more than the lookup saves.
#
# Measured over the 43 saved runs, summed across every per-ontology catalog:
#     flat chain list   831,170 chars / 11,482 lines
#     edges + routes    495,911 chars /  9,091 lines   (60% / 79%)
#     edges only        207,931 chars /  3,206 lines   (25% / 28%)
# i.e. the routes block is roughly 58% of the compacted payload. Left ON because
# the routes are what Agent 3 actually computed — they are the walks that connect
# two RETRIEVED classes, and dropping them drops that curation even though it
# drops no relation. Flip to False to spend the rest.
INCLUDE_ROUTES = True

# `1 > 7`, `[1]>[7]`, `1>7` all mean the same token. A model asked for a compact
# id will space it out or bracket it; none of that changes which walk it names.
_TOKEN_NOISE = re.compile(r"[\s\[\]]+")


def _chain_edges(chain: str) -> Optional[List[Edge]]:
    """One flattened chain -> its edges, in walk order and true triple order.

    Label-level, not URI-level, on purpose: this runs inside Agent 6, which never
    had URIs and does not need the ontology graph loaded. `path_edges` does the
    same resolution against the graph later, for Agent 7.
    """
    parsed = _parse_chain(chain)
    if parsed is None:
        return None
    nodes, props, forwards = parsed
    return [(nodes[i], props[i], nodes[i + 1]) if forwards[i]
            else (nodes[i + 1], props[i], nodes[i])
            for i in range(len(props))]


def chain_endpoints(chain: str) -> Optional[Tuple[str, str]]:
    """(first class, last class) of a chain, read in walk order.

    Used to file a chain under a class block. A chain the catalog synthesised for
    a single relation was never in Agent 5's output, so it has no recorded origin
    and its two ends are the only thing that says where it belongs.
    """
    parsed = _parse_chain(chain)
    if parsed is None:
        return None
    nodes = parsed[0]
    return nodes[0], nodes[-1]


def one_hop_chain(edge: Edge) -> str:
    """An edge as the chain string Agent 5 would have written for it.

    Forward form always: `flatten_path` reads a single-edge path from its
    declared subject, so a one-hop chain never carries a `<-[p]-` arrow. Verified
    on the saved runs — 2,186 single-hop chains, none of them backwards — which
    is what makes a resolved relation character-identical to the catalog entry
    when the catalog happened to hold one.
    """
    return f"{edge[0]} -[{edge[1]}]-> {edge[2]}"


class PathCatalog:
    """One ontology's distinct chains, factored into numbered relations + routes.

    Built from the SAME chain list the flat prompt used, so the relation set here
    is exactly the set those chains described — the reformatting cannot invent a
    relation and cannot drop one.
    """

    def __init__(self, chains: Sequence[str]) -> None:
        self.chains: List[str] = list(chains)

        self._number: Dict[Edge, int] = {}
        self._edges: List[Edge] = []                       # index = number - 1
        self._pairs: "OrderedDict[Tuple[str, str], List[Edge]]" = OrderedDict()
        self._routes: "OrderedDict[Tuple[str, ...], List[str]]" = OrderedDict()
        self._chain_of_token: Dict[str, str] = {}
        self._token_of_chain: Dict[str, str] = {}
        self._users: Dict[Edge, List[str]] = {}            # edge -> chains using it
        self.unparsed: List[str] = []

        for chain in self.chains:
            edges = _chain_edges(chain)
            if edges is None:
                # Agent 5 wrote it, so this should not happen; dropping it
                # silently would remove a relation from the catalog, so keep it
                # visible instead of guessing which way its triples run.
                self.unparsed.append(chain)
                continue
            for edge in edges:
                self._register(edge)
                self._users.setdefault(edge, []).append(chain)
            # Gated at registration, not at rendering: a route that is not shown
            # is not selectable, so it must not be resolvable either. Otherwise
            # the switch looks harmless in a replay and changes behaviour live.
            if len(edges) > 1 and INCLUDE_ROUTES:
                self._register_route(chain, edges)

    # --- construction -------------------------------------------------------

    def _register(self, edge: Edge) -> None:
        if edge in self._number:
            return
        self._edges.append(edge)
        self._number[edge] = len(self._edges)
        # First orientation seen owns the heading; the reverse one joins it and
        # gets a `<`. Which of the two came first is arbitrary and does not
        # matter — what matters is that they end up on one line.
        key = (edge[0], edge[2])
        if key not in self._pairs and (edge[2], edge[0]) in self._pairs:
            key = (edge[2], edge[0])
        self._pairs.setdefault(key, []).append(edge)

    def _register_route(self, chain: str, edges: Sequence[Edge]) -> None:
        base = ">".join(str(self._number[e]) for e in edges)
        # Two different walks over the same numbers in the same order would be
        # the same walk, so this suffix has never fired on real data (0 of 9,296
        # multi-hop chains). It is here because a token that names two chains
        # would make `resolve` return the wrong one, silently.
        token, n = base, 2
        while self._chain_of_token.setdefault(token, chain) != chain:
            token, n = f"{base}#{n}", n + 1
        self._token_of_chain[chain] = token
        self._routes.setdefault(tuple(_parse_chain(chain)[0]), []).append(token)

    # --- rendering ----------------------------------------------------------

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def route_count(self) -> int:
        return len(self._chain_of_token)

    def has_content(self) -> bool:
        return bool(self._edges)

    def render(self) -> str:
        """The block the prompt shows. Empty string when there is nothing to show."""
        if not self._edges:
            return ""
        lines = ["edges:"]
        for (a, b), edges in self._pairs.items():
            shown = []
            for edge in edges:
                back = "" if (edge[0], edge[2]) == (a, b) else "<"
                shown.append(f"[{self._number[edge]}] {back}{edge[1]}")
            lines.append(f"  {a}>{b}: " + "  ".join(shown))
        if INCLUDE_ROUTES and self._routes:
            lines.append("routes:")
            for nodes, tokens in self._routes.items():
                lines.append("  " + ">".join(nodes) + ": " + "  ".join(tokens))
        return "\n".join(lines)

    # --- selection ----------------------------------------------------------

    def normalise_token(self, token: str) -> str:
        return _TOKEN_NOISE.sub("", token)

    def is_route(self, token: str) -> bool:
        return self.normalise_token(token) in self._chain_of_token

    def resolve(self, edge_numbers: Iterable[int],
                route_tokens: Iterable[str]) -> List[str]:
        """A selection -> the chain strings it stands for.

        Relations first and routes after, so the shortest form of a connection is
        the one a downstream reader meets first — the same ordering rule
        `with_prefixes` follows.
        """
        out: List[str] = []
        for n in sorted(set(edge_numbers)):
            if 1 <= n <= len(self._edges):
                out.append(one_hop_chain(self._edges[n - 1]))
        for token in route_tokens:
            chain = self._chain_of_token.get(self.normalise_token(token))
            if chain:
                out.append(chain)
        return list(dict.fromkeys(out))

    def containing_chains(self, chain: str) -> List[str]:
        """The catalog chains that carry every relation of `chain`.

        The last resort for filing a resolved relation under a class block. The
        obvious rule — file it under the block of the class it starts at — needs
        that class to HAVE a block, and it does not always: Agent 5 only started
        adding a block per reachable class later, and on the older saved runs a
        source can hold two blocks while its chains walk through a dozen classes
        (CC-01/cmt: 86 relations, 2 blocks, 41 relations with neither end
        blocked). Falling back to the chains the relation came from puts it
        exactly where Agent 5 filed the walk that contained it, which is where
        the old whole-chain selection would have put it too.
        """
        edges = _chain_edges(chain)
        if not edges:
            return []
        found: Optional[Set[str]] = None
        for edge in edges:
            users = set(self._users.get(edge, ()))
            found = users if found is None else (found & users)
            if not found:
                return []
        # Catalog order, so the placement of a given relation does not depend on
        # set iteration order and two runs of one question stay comparable.
        return [c for c in self.chains if c in found]

    def selection_for(self, chains: Iterable[str]) -> Tuple[Set[int], List[str]]:
        """The inverse of `resolve`: chains -> (relation numbers, route tokens).

        Not used in the pipeline — it is how the round trip is asserted, and how
        a saved run's kept chains are replayed against this format.
        """
        numbers: Set[int] = set()
        tokens: List[str] = []
        for chain in chains:
            token = self._token_of_chain.get(chain)
            if token is not None:
                tokens.append(token)
                continue
            edges = _chain_edges(chain)
            if not edges:
                continue
            if len(edges) == 1 and edges[0] in self._number:
                numbers.add(self._number[edges[0]])
            else:
                # A multi-hop chain the catalog never saw: fall back to its
                # relations, which is the most this format can say about it.
                numbers.update(self._number[e] for e in edges if e in self._number)
        return numbers, tokens
