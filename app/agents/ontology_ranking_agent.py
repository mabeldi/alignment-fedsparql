from typing import Dict, List, Optional, Set
from collections import defaultdict


# How close the early anchor has to stay to the best-scoring surviving source to
# keep its place. A hyperparameter, not derived: raise it to move the anchor more
# readily, lower it to leave the early choice alone. 1.0 means only the top
# scorer may anchor; 0.0 restores the old "survived, so it stays" behaviour.
REANCHOR_MARGIN = 0.9


def _class_matches(matches: Dict[str, List[Dict]], label: str) -> List[Dict]:
    """The Class-typed matches for one extracted label that name a source."""
    return [
        m for m in matches.get(label, [])
        if m.get("payload", {}).get("type") == "Class"
        and m.get("payload", {}).get("source")
    ]


def _ancestors(uri: str, source: str) -> Set[str]:
    """Every class `uri` is a subclass of, inside its own ontology.

    Returns {uri} when the graph is unavailable, so a missing ontology makes the
    specificity test find nothing rather than raise: anchoring must never fail
    because a lookup did.
    """
    try:
        from core.ontology_loader import get_graph
        from rdflib import RDFS, URIRef
        graph = get_graph().get_context("http://" + source)
        return {str(o) for o in graph.transitive_objects(URIRef(uri), RDFS.subClassOf)}
    except Exception:
        return {uri}


def _narrows(label_a: str, label_b: str, matches: Dict[str, List[Dict]]) -> bool:
    """True when concept A is a narrower kind of concept B.

    Judged inside ONE ontology at a time: `confOf:Student subClassOf confOf:Person`
    is a fact of confof, and comparing `confOf:Student` against `edas:Person`
    would be comparing two ontologies' opinions rather than reading one's.
    """
    by_source_b: Dict[str, Set[str]] = {}
    for m in _class_matches(matches, label_b):
        payload = m["payload"]
        by_source_b.setdefault(payload["source"], set()).add(payload["uri"])

    for m in _class_matches(matches, label_a):
        payload = m["payload"]
        siblings = by_source_b.get(payload["source"])
        if not siblings:
            continue
        ancestors = _ancestors(payload["uri"], payload["source"])
        if ancestors & siblings and payload["uri"] not in siblings:
            return True
    return False


def _answer_concept(extracted: Dict[str, List[str]],
                    matches: Dict[str, List[Dict]]) -> Optional[str]:
    """Which extracted class the question actually asks to retrieve.

    The old rule was "take the first one, Agent 1 lists them head-first". That
    assumption is not something Agent 1 guarantees, and when it breaks the anchor
    lands on the wrong concept. Measured on "which of the people are still
    studying?", Agent 1 returned ["Person", "Student"] — so the anchor became
    Person, resolved to the one ontology with no Student class at all, and the
    query came back as "which people are there?", which returns everybody.

    So: prefer the NARROWEST concept instead. When one extracted class is a
    subclass of another inside the same ontology, the subclass is what the
    question narrows down to — asking which people are studying is asking for
    students. Falls back to the first resolvable label when no extracted class
    narrows another, which is the previous behaviour.
    """
    labels = [lbl for lbl in extracted.get("classes", []) if _class_matches(matches, lbl)]
    if not labels:
        return None

    # Most specific = narrows the greatest number of the other labels. Ties keep
    # the original order, so this only ever moves the anchor on real evidence.
    best = max(labels, key=lambda a: sum(_narrows(a, b, matches) for b in labels if b != a))
    if best != labels[0]:
        print(f"[anchor] answer concept: '{best}' (narrower than "
              f"'{labels[0]}', which Agent 1 listed first)")
    return best

def _expected_match_types(
    extracted: Optional[Dict[str, List[str]]],
) -> Dict[str, Set[str]]:
    """Expected ontology term types for each extracted keyword."""
    expected: Dict[str, Set[str]] = defaultdict(set)
    if not extracted:
        return expected

    groups = (
        ("classes", "Class"),
        ("objectProperties", "ObjectProperty"),
        ("dataTypeProperties", "DatatypeProperty"),
    )
    for key, ontology_type in groups:
        for term in extracted.get(key, []):
            expected[str(term).strip().casefold()].add(ontology_type)
    return expected


def rank_ontology_sources(
    matches: Dict[str, List[Dict]],
    extracted: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """
    Rank ontology sources by their best surviving match for each keyword.

    A source may return several near-duplicate terms for one extracted keyword.
    Summing every term rewards that result volume rather than relevance, so only
    the highest score from each (keyword, source) group contributes. Those best
    scores are then summed across keywords, rewarding sources that cover more of
    the question without letting one noisy keyword dominate the ranking.

    Args:
        matches: Dict of keyword -> list of matched concepts (each with a score and payload)

    Returns:
        Dictionary of source -> sum of its per-keyword best scores.
    """
    source_scores = defaultdict(float)
    expected_types = _expected_match_types(extracted)

    for keyword, matched_list in matches.items():
        allowed_types = expected_types.get(str(keyword).strip().casefold())
        best_for_keyword: Dict[str, float] = {}
        for match in matched_list:
            score = float(match.get("score", 0))
            payload = match.get("payload", {})
            source = payload.get("source")
            match_type = payload.get("type")
            if allowed_types and match_type not in allowed_types:
                continue
            if source:
                best_for_keyword[source] = max(
                    best_for_keyword.get(source, score), score
                )

        for source, score in best_for_keyword.items():
            source_scores[source] += score

    return dict(source_scores)


def select_base_source(
    extracted: Dict[str, List[str]],
    matches: Dict[str, List[Dict]],
    top_sources: Dict[str, float],
) -> str | None:
    """Pick the ontology that should be the base graph (starting point) of the
    federated SPARQL query.

    Federation needs a stable anchor, so we start from the ontology that owns
    the question's *answer entity* — the main thing being asked for. Which
    extracted class that is comes from `_answer_concept` (the narrowest one),
    not from the order Agent 1 happened to list them in.

    The source is then the one holding that concept's highest-scoring Class
    match. If no class resolves, we fall back to the source with the highest
    aggregated match score.

    Returns the source name (e.g. "cmt") or None if nothing resolves.
    """
    answer_concept = _answer_concept(extracted, matches)
    if answer_concept:
        class_matches = _class_matches(matches, answer_concept)
        best = max(class_matches, key=lambda m: float(m.get("score", 0)))
        return best["payload"]["source"]

    # Fallback: the source that accumulated the most match relevance overall.
    if top_sources:
        return max(top_sources, key=top_sources.get)
    return None


def reanchor_base_source(
    base_source: str | None,
    finalized_paths: List[Dict],
    top_sources: Dict[str, float],
    scores: Dict[str, float] | None = None,
) -> str | None:
    """Move the anchor onto a source that survived the contextual path filter.

    `select_base_source` runs on Agent 4's output, before Agent 6 decides which
    ontologies contribute anything, so it can name a source Agent 6 then empties.
    Measured on "which papers were accepted?": the anchor was a source with no
    surviving block, and Agent 7 — told to start there and given no vocabulary
    for it — invented a branch on a broader class that matched every paper, not
    just the accepted ones.

    Ranking order is kept: among the surviving sources we take the highest
    scored one, so the anchor is still the most relevant ontology rather than
    whichever block happens to come first.

    `scores` (path scores, one per source) is that ranking when given, and
    `top_sources` is it when they are absent. The two are never mixed inside one
    ranking: `top_sources` sums a best score per keyword, so it grows with the
    length of the question, while a path score is one path's. Filling a hole in
    `scores` with a `top_sources` number therefore compared a sum against a
    single value, and the source missing from `scores` — the one whose paths all
    scored too low to keep — came out on top of the very ranking that rejected
    it. A source `scores` does not list is a source with no surviving path, so
    it now ranks as the 0.0 that says.

    Surviving is also not on its own a reason to stay anchored. `base_source`
    comes from Agent 4 word matches; `scores` is measured after paths are built,
    filtered and checked against the store, so it is the better evidence and
    gets to move the anchor when it clearly prefers another candidate.
    """
    survivors = {p["source"] for p in finalized_paths if p.get("source")}
    if not survivors:
        return base_source

    # A standalone answer class and a complete retained relation are both valid
    # answer-bearing evidence. Agent 6 deliberately leaves `answers_question`
    # false when a relation already carries the answer, so consulting that flag
    # alone excludes precisely the relational sources we want to prefer.
    answering = {p["source"] for p in finalized_paths
                 if p.get("source") and (
                     p.get("answers_question")
                     or p.get("complete_answer_branch")
                     or p.get("source_complete_answer_branch")
                 )}
    candidates = answering or survivors

    rank_by = scores if scores else top_sources
    chosen = max(candidates, key=lambda s: rank_by.get(s, 0.0))

    if base_source in candidates:
        # Without path scores there is nothing better to compare against, so the
        # earlier choice stands — this is the old behaviour for that case.
        if not scores:
            return base_source
        keep, best = rank_by.get(base_source, 0.0), rank_by.get(chosen, 0.0)
        if keep >= REANCHOR_MARGIN * best:
            return base_source
        print(f"[anchor] {base_source} kept a path but scores {keep:.2f} "
              f"against {chosen}'s {best:.2f} — re-anchored to {chosen}")
        return chosen

    reason = ("holds no answer class" if base_source in survivors else "kept no path")
    print(f"[anchor] {base_source} {reason} — re-anchored to {chosen} "
          f"(of {', '.join(sorted(candidates))})")
    return chosen
