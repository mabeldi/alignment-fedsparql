"""Relevance scoring of class paths against the user's question.

Scoring only: nothing here filters, cuts or reorders the pipeline's data.
"""

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdflib import RDFS, URIRef

from agents.agent2_vector_lookup import embed_text
from core.ontology_loader import get_graph
from tools.find_class_paths import build_label_index, local_name
from tools.path_sparql import _parse_chain

# (full path, classes, relations) — fixed.
SCORE_WEIGHTS: Tuple[float, float, float] = (0.5, 0.3, 0.2)

# How the per-class similarities of one chain are combined:
#   "all"       -> mean over every class in the chain
#   "terminals" -> mean of the first and the last class only
CLASS_REL_MODE = "all"

# Value given to any component whose embedding could not be produced.
NEUTRAL_SCORE = 0.0

# text -> embedding, for the process lifetime. The key is the term's URI when the
# name resolves to one, else "text:<readable form>" for chain texts and for names
# no ontology claims.
_VECTORS: Dict[str, np.ndarray] = {}

# (source, raw name) -> (uri or None, readable text)
_TERMS: Dict[Tuple[str, str], Tuple[Optional[str], str]] = {}

# (source, term uri) -> readable text
_LABELS: Dict[Tuple[str, str], str] = {}


@dataclass
class PathScore:
    source: str
    chain: str
    full: float
    classes: float
    relations: float
    total: float
    # False when a component could not be computed, so the score is unknown
    # rather than genuinely low.
    ok: bool = True
    # Structural tier. For a relational question, a class-to-class chain is a
    # complete candidate and an isolated class/value block is only partial.
    complete: bool = False
    # Index of the block this path came from, in the list handed to score_paths.
    # `source` and `chain` alone do not identify a block: a chainless block has
    # chain "", so two of them in one source are indistinguishable.
    block: int = -1


def humanize(name: str) -> str:
    """Turn an identifier into plain words: "writePaper" -> "write paper"."""
    if not name:
        return ""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    # "hasURIRef" -> "has URI Ref": keep an acronym whole, split off what follows.
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"[_\-]+", " ", s)
    return " ".join(s.split()).lower()


def _graph_for(source: str):
    """The loaded graph of `source`, or None while the store is still loading.

    None means every name below falls back to its split form. Those fallbacks are
    never cached: the store fills in after startup, and a cached fallback would
    outlive the gap it was made for.
    """
    try:
        return get_graph().get_context("http://" + source)
    except Exception:
        return None


def readable_uri(uri: str, source: str) -> str:
    """Readable text of a term URI: its rdfs:label if declared, else its split name."""
    key = (source, uri)
    cached = _LABELS.get(key)
    if cached is not None:
        return cached

    graph = _graph_for(source)
    if graph is None:
        return humanize(local_name(uri))

    label = None
    try:
        value = graph.value(URIRef(uri), RDFS.label)
        if value is not None:
            label = str(value).strip()
    except Exception:
        label = None

    text = label or humanize(local_name(uri))
    _LABELS[key] = text
    return text


def _term(name: str, source: str) -> Tuple[Optional[str], str]:
    """A chain's bare name -> (its URI in this ontology or None, its readable text)."""
    key = (source, name)
    cached = _TERMS.get(key)
    if cached is not None:
        return cached

    graph = _graph_for(source)
    if graph is None:
        return (None, humanize(name))

    uri = None
    try:
        uri = build_label_index(graph).get(name.lower())
    except Exception:
        uri = None

    text = readable_uri(uri, source) if uri else humanize(name)
    resolved = (uri, text)
    _TERMS[key] = resolved
    return resolved


def _cache_key(uri: Optional[str], text: str) -> str:
    return uri or f"text:{text}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return NEUTRAL_SCORE
    return float(np.dot(a, b) / denom)


def _embed(text: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(embed_text(text), dtype=float)
    except Exception as exc:
        print(f"[path_scorer] embedding failed for {text[:60]!r}: {exc}")
        return None


def _prefetch(items: Sequence[Tuple[str, str]]) -> None:
    """Embed every (cache key, text) pair not already cached, each one once."""
    pending: Dict[str, str] = {}
    for key, text in items:
        if key not in _VECTORS and key not in pending:
            pending[key] = text
    for key, text in pending.items():
        vector = _embed(text)
        if vector is not None:
            _VECTORS[key] = vector


def _similarity(question_vec: np.ndarray, key: str, cache: Dict[str, float]) -> float:
    """Cosine between the question and the term behind `key`.

    `cache` holds the scores of one `score_paths` call only: it is keyed by term
    alone, so it is valid exactly as long as `question_vec` stays the same vector.
    """
    score = cache.get(key)
    if score is not None:
        return score

    vector = _VECTORS.get(key)
    score = NEUTRAL_SCORE if vector is None else _cosine(question_vec, vector)
    cache[key] = score
    return score


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else NEUTRAL_SCORE


@dataclass
class _Item:
    """One thing to score: a parsed chain, or a block with no chain at all."""
    source: str
    chain: str
    full_key: str
    full_text: str
    class_keys: List[str]
    # One list of predicate keys per hop; a chainless block gets one pseudo-hop
    # holding its attribute names.
    hop_keys: List[List[str]]
    # Every (cache key, text) this item needs embedded.
    wanted: List[Tuple[str, str]]
    node_texts: Optional[List[str]] = None
    property_texts: Optional[List[str]] = None
    # False when the chain could not be read, so the class and relation
    # components have nothing to score.
    readable: bool = True
    # Index of the block this item came from.
    block: int = -1


# (source, chain) -> its item, carrying no block index
_CHAIN_ITEMS: Dict[Tuple[str, str], _Item] = {}


def _parsed_chain_item(chain: str, source: str) -> _Item:
    """The blockless form of a chain's item: everything but its block index."""
    parsed = _parse_chain(chain)
    if parsed is None:
        # Arrows the path filter reworded: the walk cannot be read, so only the
        # whole-chain text is scored and the two component parts stay neutral.
        text = humanize(chain)
        return _Item(source, chain, f"text:{text}", text, [], [],
                     [(f"text:{text}", text)], readable=False)

    nodes, props, _ = parsed
    node_terms = [_term(n, source) for n in nodes]
    prop_terms = [_term(p, source) for p in props]

    tokens: List[str] = []
    for i, (_, node_text) in enumerate(node_terms):
        tokens.append(node_text)
        if i < len(prop_terms):
            tokens.append(prop_terms[i][1])
    full_text = " ".join(t for t in tokens if t)

    wanted = [(_cache_key(uri, text), text) for uri, text in node_terms + prop_terms]
    wanted.append((f"text:{full_text}", full_text))

    return _Item(
        source=source,
        chain=chain,
        full_key=f"text:{full_text}",
        full_text=full_text,
        class_keys=[_cache_key(uri, text) for uri, text in node_terms],
        hop_keys=[[_cache_key(uri, text)] for uri, text in prop_terms],
        wanted=wanted,
        node_texts=[text for _, text in node_terms],
        property_texts=[text for _, text in prop_terms],
    )


def _chain_item(chain: str, source: str, index: int) -> _Item:
    """The item for `chain` as it appears in block `index`.

    One chain can appear in several blocks — outgoing on one class, incoming on
    another — so it is read once and only its block index differs per occurrence.
    """
    key = (source, chain)
    item = _CHAIN_ITEMS.get(key)
    if item is None:
        item = _parsed_chain_item(chain, source)
        if _graph_for(source) is not None:
            _CHAIN_ITEMS[key] = item
    return replace(item, block=index)


def _blockless_item(block: Dict, index: int) -> _Item:
    """A block with no chain, scored on its class name and its attribute names."""
    source = block.get("source") or ""
    class_uri = block.get("class") or ""
    class_text = (readable_uri(class_uri, source) if class_uri
                  else humanize(block.get("label") or ""))
    class_key = _cache_key(class_uri or None, class_text)

    attrs: List[Tuple[str, str]] = []
    for edge in block.get("values", ()) or ():
        if not edge or len(edge) < 2 or not edge[1]:
            continue
        uri = edge[1]
        pair = (uri, readable_uri(uri, source))
        if pair not in attrs:
            attrs.append(pair)

    full_text = " ".join([class_text] + [text for _, text in attrs]).strip()
    wanted = [(class_key, class_text)]
    wanted += [(_cache_key(uri, text), text) for uri, text in attrs]
    wanted.append((f"text:{full_text}", full_text))

    return _Item(
        source=source,
        chain="",
        full_key=f"text:{full_text}",
        full_text=full_text,
        class_keys=[class_key],
        hop_keys=[[_cache_key(uri, text) for uri, text in attrs]] if attrs else [],
        wanted=wanted,
        node_texts=[class_text],
        property_texts=[text for _, text in attrs],
        block=index,
    )


def _items_for(blocks: List[Dict]) -> List[_Item]:
    items: List[_Item] = []
    for index, block in enumerate(blocks):
        source = block.get("source") or ""
        chains: List[str] = []
        for direction in ("outgoing", "incoming"):
            for chain in block.get(direction) or ():
                if chain and chain not in chains:
                    chains.append(chain)
        if chains:
            items += [_chain_item(chain, source, index) for chain in chains]
        else:
            items.append(_blockless_item(block, index))
    return items


_COVERAGE_STOPWORDS = {
    "a", "an", "are", "at", "by", "for", "from", "has", "have", "in",
    "is", "of", "on", "the", "to", "with",
}


def _coverage_tokens(text: str) -> Set[str]:
    return {
        token for token in humanize(text).split()
        if token and token not in _COVERAGE_STOPWORDS
    }


def _covers_token_sets(nodes: Sequence[Set[str]],
                       required: Sequence[str]) -> bool:
    """True when different path nodes cover every requested resource concept.

    Takes the nodes already reduced to tokens, so a caller that widens a node
    with the names of the classes it is aligned to can pass the widened set.
    """
    wanted = [_coverage_tokens(name) for name in required if _coverage_tokens(name)]
    candidates = [
        [index for index, node in enumerate(nodes) if concept <= node]
        for concept in wanted
    ]
    if any(not choices for choices in candidates):
        return False

    # GTX A compound class such as PaperAbstract must not count as both Paper
    # GTX and Abstract; each requested resource concept needs its own path node.
    def assign(position: int, used: Set[int]) -> bool:
        if position == len(candidates):
            return True
        return any(
            index not in used and assign(position + 1, used | {index})
            for index in candidates[position]
        )

    return assign(0, set())


def _covers_classes(node_texts: Sequence[str], required: Sequence[str]) -> bool:
    """`_covers_token_sets` for callers holding plain node names."""
    return _covers_token_sets([_coverage_tokens(name) for name in node_texts],
                              required)


def _covers_relations(property_texts: Sequence[str], required: Sequence[str]) -> bool:
    available = [_coverage_tokens(name) for name in property_texts]
    return all(
        any(wanted <= actual for actual in available)
        for name in required
        if (wanted := _coverage_tokens(name))
    )


def complete_answer_sources(blocks: Sequence[Dict],
                            required_classes: Sequence[str],
                            required_values: Sequence[str] = (),
                            class_aliases: Optional[
                                Dict[str, Set[str]]] = None) -> Set[str]:
    """Sources whose retained graph connects all requested resource concepts.

    `class_aliases` maps a class URI to the names of the classes it is declared
    equivalent to. Without it the test is purely lexical, so a source naming the
    requested concept differently can never be complete however the alignment
    reads. The alias names widen that class's tokens, which admits a source that
    really does hold the answer under another name and still rejects one whose
    only correspondence is to some broader class.
    """
    class_aliases = class_aliases or {}
    nodes_by_source: Dict[str, Set[str]] = {}
    adjacency_by_source: Dict[str, Dict[str, Set[str]]] = {}
    values_by_source: Dict[str, List[str]] = {}
    # Tokens per node, kept apart from the node names so an alias widens the
    # node it belongs to instead of becoming a second node. A separate node
    # would let one class fill two requested concepts at once.
    tokens_by_source: Dict[str, Dict[str, Set[str]]] = {}

    for block in blocks:
        source = block.get("source") or ""
        if not source:
            continue
        nodes = nodes_by_source.setdefault(source, set())
        adjacency = adjacency_by_source.setdefault(source, {})
        tokens = tokens_by_source.setdefault(source, {})
        label = block.get("label") or humanize(local_name(block.get("class") or ""))
        if label:
            nodes.add(label)
            adjacency.setdefault(label, set())
            widened = tokens.setdefault(label, _coverage_tokens(label))
            for alias in class_aliases.get(block.get("class") or "", ()):
                widened |= _coverage_tokens(alias)

        for edge in block.get("values", ()) or ():
            if edge and len(edge) > 1 and edge[1]:
                values_by_source.setdefault(source, []).append(local_name(edge[1]))

        for side in ("outgoing", "incoming"):
            for chain in block.get(side) or ():
                parsed = _parse_chain(chain)
                if parsed is None:
                    continue
                chain_nodes, _, _ = parsed
                for node in chain_nodes:
                    nodes.add(node)
                    adjacency.setdefault(node, set())
                    tokens.setdefault(node, _coverage_tokens(node))
                for left, right in zip(chain_nodes, chain_nodes[1:]):
                    adjacency[left].add(right)
                    adjacency[right].add(left)

    complete: Set[str] = set()
    for source, nodes in nodes_by_source.items():
        if (required_values
                and not _covers_relations(values_by_source.get(source, []),
                                           required_values)):
            continue

        unseen = set(nodes)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbours = adjacency_by_source[source].get(current, set()) & unseen
                unseen -= neighbours
                component |= neighbours
                frontier.extend(neighbours)
            if _covers_token_sets(
                    [tokens_by_source[source].get(node, _coverage_tokens(node))
                     for node in component],
                    required_classes):
                complete.add(source)
                break
    return complete


def _is_complete_answer_branch(item: _Item,
                               required_classes: Sequence[str],
                               required_relations: Sequence[str]) -> bool:
    if required_relations and not item.chain:
        return False
    return (_covers_classes(item.node_texts or [], required_classes)
            and _covers_relations(item.property_texts or [], required_relations))


def _score_item(item: _Item, question_vec: np.ndarray,
                cache: Dict[str, float], required_classes: Sequence[str],
                required_relations: Sequence[str]) -> PathScore:
    full = _similarity(question_vec, item.full_key, cache)

    keys = item.class_keys
    if CLASS_REL_MODE == "terminals" and len(keys) > 2:
        keys = [keys[0], keys[-1]]
    classes = _mean([_similarity(question_vec, k, cache) for k in keys])

    # Per hop the best-matching predicate speaks for that hop; the chain is the
    # average of those bests, so a long walk is not penalised for its weak hops
    # beyond what averaging already does.
    relations = _mean([
        max(_similarity(question_vec, k, cache) for k in hop) if hop else NEUTRAL_SCORE
        for hop in item.hop_keys
    ])
    # A chainless block with no attributes has nothing to read a relation off, so
    # its class score stands in rather than dragging the total down.
    if not item.hop_keys:
        relations = classes

    w_full, w_classes, w_relations = SCORE_WEIGHTS
    total = w_full * full + w_classes * classes + w_relations * relations

    needed = [item.full_key] + item.class_keys + [k for hop in item.hop_keys for k in hop]
    ok = item.readable and all(key in _VECTORS for key in needed)
    # GTX Complete means that this one branch covers the requested resource
    # GTX concepts and relations, not merely that the block contains any chain.
    complete = _is_complete_answer_branch(
        item, required_classes, required_relations)
    return PathScore(item.source, item.chain, full, classes, relations, total,
                     ok=ok, complete=complete, block=item.block)


def score_terms(question: str, terms: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Cosine of each `(term URI, its source)` against the question, keyed by URI.

    For comparing terms against each other — which of a union's members the
    question is really asking about — where `score_paths` compares whole walks.

    A term whose embedding failed is LEFT OUT rather than given a neutral score,
    so a caller ranking the result can tell "scored low" from "not scored" and
    fall back instead of picking a winner off an incomplete comparison.

    `readable_uri` and the `_VECTORS` cache are shared with `score_paths`, so a
    term read here costs nothing the second time and sits on the same scale.
    """
    if not terms:
        return {}
    question_vec = _embed(question)
    if question_vec is None:
        return {}

    texts = {uri: readable_uri(uri, source) for uri, source in terms}
    _prefetch([(_cache_key(uri, text), text) for uri, text in texts.items()])

    scored: Dict[str, float] = {}
    for uri, text in texts.items():
        vector = _VECTORS.get(_cache_key(uri, text))
        if vector is not None:
            scored[uri] = _cosine(question_vec, vector)
    return scored


def score_paths(question: str, blocks: List[Dict],
                required_classes: Sequence[str] = (),
                required_relations: Sequence[str] = ()) -> List[PathScore]:
    """Score every (block, chain) pair for relevance to `question`.

    One `PathScore` per distinct chain of each block. A block carrying no chain
    (an answer class or a value carrier) still gets exactly one score, computed
    from its class name and its attribute names instead: the class name is the
    class component, the best-matching attribute is the relation component, and
    the two together are the full-text component. With no attributes at all, the
    relation component takes the class score.

    Never raises: a failed embedding leaves that component at `NEUTRAL_SCORE`.
    Such a score carries `ok=False`, so a lookup failure is never read as a
    genuine low score.
    """
    items = _items_for(blocks)
    if not items:
        return []

    question_vec = _embed(question)
    if question_vec is None:
        print("[path_scorer] question embedding failed — returning neutral scores")
        return [PathScore(i.source, i.chain, NEUTRAL_SCORE, NEUTRAL_SCORE,
                          NEUTRAL_SCORE, NEUTRAL_SCORE, ok=False,
                          complete=_is_complete_answer_branch(
                              i, required_classes, required_relations),
                          block=i.block)
                for i in items]

    _prefetch([(key, text) for item in items for key, text in item.wanted if text])

    similarities: Dict[str, float] = {}
    return [_score_item(item, question_vec, similarities,
                        required_classes, required_relations)
            for item in items]
