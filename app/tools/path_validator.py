"""Drop class paths that no instance in the triple store actually follows.

Each surviving chain is turned into a single ASK against its ontology's endpoint;
a chain the store answers `false` for is removed. Every lookup failure keeps the
path.
"""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import requests

from config.sparql_endpoints import SPARQL_ENDPOINTS
from tools.path_sparql import _parse_chain, path_edges

ASK_TIMEOUT_S = 5
USE_ASK_VALIDATION = True
MAX_PARALLEL_ASKS = 8

# One hop of a chain: its predicate URI and whether the triple runs forward.
Hop = Tuple[str, bool]

# Answers never change during a run, so every distinct check is issued once.
_hop_cache: Dict[Tuple[str, str], Optional[bool]] = {}
_chain_cache: Dict[Tuple[str, Tuple[Hop, ...]], Optional[bool]] = {}
_class_cache: Dict[Tuple[str, str], Optional[bool]] = {}

_SESSION = requests.Session()
_SESSION.mount("http://", requests.adapters.HTTPAdapter(
    pool_connections=MAX_PARALLEL_ASKS, pool_maxsize=MAX_PARALLEL_ASKS))


# --- asking the endpoint ----------------------------------------------------

def _ask(source: str, where: str) -> Optional[bool]:
    """Run `ASK { where }` against `source`. None when the endpoint gave no answer."""
    url = SPARQL_ENDPOINTS.get(source)
    if not url:
        print(f"[validator] {source}: unknown source, paths kept unchecked")
        return None
    try:
        response = _SESSION.post(
            url, data={"query": "ASK { " + where + " }"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=ASK_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        print(f"[validator] {source}: {type(exc).__name__} on ASK, path kept — {exc}")
        return None
    if response.status_code != 200:
        print(f"[validator] {source}: HTTP {response.status_code} on ASK, path kept")
        return None
    try:
        return bool(response.json()["boolean"])
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[validator] {source}: unreadable ASK response, path kept — {exc}")
        return None


# A queued check: the cache to fill, its key, and the question to ask.
_Check = Tuple[Dict, Tuple, str, str]


def _want(cache: Dict, key: Tuple, source: str, where: str,
          pending: Dict[Tuple, _Check]) -> None:
    """Queue a check unless it is already cached or already queued."""
    if key in cache or (id(cache), key) in pending:
        return
    pending[(id(cache), key)] = (cache, key, source, where)


def _run_checks(pending: Dict[Tuple, _Check]) -> None:
    """Issue every queued check concurrently and store the answers."""
    checks = list(pending.values())
    if not checks:
        return
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_ASKS, len(checks))) as pool:
        answers = list(pool.map(lambda c: _ask(c[2], c[3]), checks))
    for (cache, key, _, _), answer in zip(checks, answers):
        cache[key] = answer


# --- turning a chain into one ASK -------------------------------------------

def _chain_hops(chain: str, source: str) -> Optional[List[Hop]]:
    """Chain string -> one (predicate_uri, runs_forward) per hop, or None if unresolvable."""
    edges = path_edges(chain, source)
    parsed = _parse_chain(chain)
    if not edges or not parsed or len(parsed[2]) != len(edges):
        return None
    return [(edge[1], forward) for edge, forward in zip(edges, parsed[2])]


def _chain_where(hops: List[Hop]) -> str:
    """The body of the ASK: every hop joined end to end on shared variables."""
    lines = []
    for i, (predicate, forward) in enumerate(hops):
        near, far = f"?n{i}", f"?n{i + 1}"
        # A backwards hop is the same triple with subject and object swapped, so
        # the next hop still meets this one on ?n{i+1}.
        lines.append(f"{near} <{predicate}> {far} ." if forward
                     else f"{far} <{predicate}> {near} .")
    return " ".join(lines)


def _entry_for(source: str, hops: List[Hop]) -> Tuple[Dict, Tuple]:
    """The cache and key holding this chain's answer."""
    if len(hops) == 1:
        return _hop_cache, (source, hops[0][0])
    return _chain_cache, (source, tuple(hops))


def _first_value_prop(block: Dict) -> Optional[str]:
    """The URI of the block's first attribute, if it has one."""
    for edge in block.get("values") or ():
        if edge and len(edge) > 1 and edge[1]:
            return edge[1]
    return None


def _chains_of(block: Dict) -> List[str]:
    return list(block.get("outgoing") or []) + list(block.get("incoming") or [])


def _class_has_data(block: Dict, source: str) -> bool:
    """True unless the class is known empty AND its first attribute is too."""
    if _class_cache.get((source, block.get("class"))) is not False:
        return True
    prop = _first_value_prop(block)
    return bool(prop) and _hop_cache.get((source, prop)) is not False


# --- entry points -----------------------------------------------------------

def validate_paths(blocks: List[Dict]) -> List[Dict]:
    """Return the blocks with dead chains removed, and dead chainless blocks dropped."""
    if not USE_ASK_VALIDATION or not blocks:
        return blocks

    live_sources = {b.get("source") for b in blocks} & set(SPARQL_ENDPOINTS)

    # Round 1: every chain, plus a class check for the blocks that have none.
    hops_of: Dict[Tuple[str, str], Optional[List[Hop]]] = {}
    pending: Dict[Tuple, _Check] = {}
    for block in blocks:
        source = block.get("source")
        if source not in live_sources:
            continue
        chains = _chains_of(block)
        if not chains:
            class_uri = block.get("class")
            if class_uri:
                _want(_class_cache, (source, class_uri), source,
                      f"?x a <{class_uri}> .", pending)
            continue
        for chain in chains:
            if (source, chain) not in hops_of:
                hops_of[(source, chain)] = _chain_hops(chain, source)
            hops = hops_of[(source, chain)]
            if hops:
                cache, key = _entry_for(source, hops)
                _want(cache, key, source, _chain_where(hops), pending)
    _run_checks(pending)

    # Round 2: an attribute lookup for the chainless blocks whose class is empty.
    # It cannot be queued earlier because it only runs when the class fails.
    pending = {}
    for block in blocks:
        source = block.get("source")
        if source not in live_sources or _chains_of(block):
            continue
        if _class_cache.get((source, block.get("class"))) is not False:
            continue
        prop = _first_value_prop(block)
        if prop:
            _want(_hop_cache, (source, prop), source, f"?x <{prop}> ?v .", pending)
    _run_checks(pending)

    checked: Counter = Counter()
    dead_chains: Counter = Counter()
    dead_blocks: Counter = Counter()
    kept_per_source: Counter = Counter()

    kept: List[Dict] = []
    for block in blocks:
        source = block.get("source")
        if source not in live_sources:
            kept.append(block)
            continue

        surviving: Dict[str, List[str]] = {}
        for side in ("outgoing", "incoming"):
            surviving[side] = []
            for chain in block.get(side) or []:
                hops = hops_of.get((source, chain))
                if not hops:
                    surviving[side].append(chain)      # unresolvable: keep it
                    continue
                checked[source] += 1
                cache, key = _entry_for(source, hops)
                if cache.get(key) is False:
                    dead_chains[source] += 1
                else:
                    surviving[side].append(chain)

        if surviving["outgoing"] or surviving["incoming"]:
            kept.append({**block, "outgoing": surviving["outgoing"],
                         "incoming": surviving["incoming"]})
            kept_per_source[source] += 1
            continue

        # Nothing left to join on. A block that arrived without chains is an
        # answer or carrier class, so it stands on its own class instead.
        if not _chains_of(block):
            if _class_has_data(block, source):
                kept.append(block)
                kept_per_source[source] += 1
            else:
                dead_blocks[source] += 1
        elif block.get("answers_question"):
            kept.append({**block, "outgoing": [], "incoming": [],
                         "complete_answer_branch": False})
            kept_per_source[source] += 1
        else:
            dead_blocks[source] += 1

    for source in sorted(live_sources):
        print(f"[validator] {source}: {checked[source]} chain(s) checked, "
              f"{dead_chains[source]} dropped, {dead_blocks[source]} block(s) removed")
        if not kept_per_source[source]:
            print(f"[validator] ⚠ {source}: every path is empty in the store — "
                  f"that is a data gap, not a bug")

    return kept


async def validate_paths_async(blocks: List[Dict]) -> List[Dict]:
    """`validate_paths` run off the event loop, for the async pipeline."""
    return await asyncio.to_thread(validate_paths, blocks)
