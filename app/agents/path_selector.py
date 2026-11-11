"""Cut the paths that do not answer the question, and the sources left with none.

Deterministic. The relevance numbers come from `path_scorer`; everything here is
thresholding and bookkeeping.

ONE KNOB
--------
A path is kept when it scores at least `KEEP_RATIO` of the best score in the
request. That is the whole decision. There used to be a middle band and three
rules (P1/P2/P3) that could rescue a path out of it — sole alignment bridge,
sole cover of a question term, last path of an already-kept source. They are
gone: they made the outcome hard to attribute, since a path could survive for
four different reasons and telling them apart meant reading the table. If a run
shows a needed path being cut, the fix belongs in `path_scorer` or in
`KEEP_RATIO`, not in a new special case here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from agents.path_scorer import PathScore, complete_answer_sources, score_paths
from tools.alignment_store import Mapping, local_name

USE_PATH_SCORING = True

# Fraction of the best score in the request a path must reach to survive.
# A hyperparameter, not derived: raise it to cut harder, lower it to keep more.
KEEP_RATIO = 0.75

# (block index, chain) — identifies one scored path. The source is not part of
# it: two chainless blocks of one source both carry chain "".
_Key = Tuple[int, str]


@dataclass
class PathReport:
    """One row of the decision table, for the console and for the response."""
    source: str
    chain: str
    full: float
    classes: float
    relations: float
    total: float
    complete: bool
    kept: bool
    reason: str


@dataclass
class Selection:
    blocks: List[Dict]
    # Max score of each surviving source's surviving paths.
    source_scores: Dict[str, float] = field(default_factory=dict)
    # Sources that had scored paths and lost every one. They stay out.
    cut_sources: Set[str] = field(default_factory=set)
    rows: List[PathReport] = field(default_factory=list)


# --- helpers ----------------------------------------------------------------

def _key(score: PathScore) -> _Key:
    return (score.block, score.chain)


def _rewrite(blocks: List[Dict], kept: Set[_Key], complete: Set[_Key]) -> List[Dict]:
    """Rewrite kept paths and stamp their structural answer tier on each block."""
    out: List[Dict] = []
    for index, block in enumerate(blocks):
        chains = [c for side in ("outgoing", "incoming") for c in block.get(side) or []]
        if not chains:
            if (index, "") in kept:
                out.append({**block,
                            "complete_answer_branch": (index, "") in complete})
            continue
        surviving = {
            side: [c for c in block.get(side) or [] if (index, c) in kept]
            for side in ("outgoing", "incoming")
        }
        if surviving["outgoing"] or surviving["incoming"]:
            surviving_chains = surviving["outgoing"] + surviving["incoming"]
            out.append({**block, "outgoing": surviving["outgoing"],
                        "incoming": surviving["incoming"],
                        "complete_answer_branch": any(
                            (index, chain) in complete
                            for chain in surviving_chains
                        )})
    return out


def _class_aliases(mappings: Sequence[Mapping]) -> Dict[str, Set[str]]:
    """Class URI -> the local names of the classes it is declared equal to.

    Both directions, because the completeness test asks the question from
    whichever side happens to own a retained path. The restriction that
    narrows a conditional mapping is deliberately dropped here: it says HOW to
    express the concept, which is the generator's problem, while this map only
    answers WHETHER the source can express it at all.
    """
    aliases: Dict[str, Set[str]] = {}
    for mapping in mappings:
        for target in mapping.targets:
            if not target or target == mapping.entity1:
                continue
            aliases.setdefault(mapping.entity1, set()).add(local_name(target))
            aliases.setdefault(target, set()).add(local_name(mapping.entity1))
    return aliases


def stamp_source_completeness(
        blocks: List[Dict],
        extracted: Dict[str, List[str]] | None = None,
        mappings: Sequence[Mapping] = ()) -> List[Dict]:
    """Stamp sources whose retained paths jointly form a complete answer graph."""
    extracted = extracted or {}
    complete = complete_answer_sources(
        blocks,
        extracted.get("classes", []),
        extracted.get("dataTypeProperties", []),
        _class_aliases(mappings),
    )
    # GTX A complete answer can span several retained chains in one ontology;
    # GTX keep this source-level fact separate from per-chain completeness.
    return [
        {**block, "source_complete_answer_branch": block.get("source") in complete}
        for block in blocks
    ]


def _print_table(rows: List[PathReport]) -> None:
    if not rows:
        return
    width = max((len(r.source) for r in rows), default=6)
    print(f"[selector] {sum(r.kept for r in rows)}/{len(rows)} path(s) kept")
    for row in rows:
        tier = "complete" if row.complete else "partial "
        print(f"[selector] {'KEEP' if row.kept else 'CUT '}  "
              f"{row.source:<{width}}  {row.total:.2f}  {tier}  {row.reason}")


# --- entry point ------------------------------------------------------------

def select_paths(question: str, blocks: List[Dict],
                 extracted: Dict[str, List[str]] | None = None) -> Selection:
    """Score every path, cut the irrelevant ones, and report what was cut and why.

    A path is kept when it scores at least `KEEP_RATIO` of the best score in this
    request, and cut otherwise.

    A path whose score could not be computed (`PathScore.ok` false) is kept
    unconditionally and left out of the best-score maximum, so an embedding
    failure never reads as a genuine zero.

    `cut_sources` holds the sources that had paths and lost every one. Sources
    that never had a path are not in it — they are the caller's business.
    """
    if not USE_PATH_SCORING or not blocks:
        return Selection(blocks)

    extracted = extracted or {}
    # GTX Pass the requested semantic shape so "complete" measures coverage of
    # GTX the question rather than treating every non-empty chain as complete.
    scores = score_paths(
        question,
        blocks,
        required_classes=extracted.get("classes", []),
        required_relations=extracted.get("objectProperties", []),
    )
    if not scores:
        return Selection(blocks)

    best_by_tier: Dict[bool, float] = {}
    for score in scores:
        if score.ok:
            best_by_tier[score.complete] = max(
                best_by_tier.get(score.complete, score.total), score.total)
    best = max(best_by_tier.values(), default=0.0)
    source_scores: Dict[str, float] = {}

    if best <= 0.0:
        # Nothing to measure against: every score is unknown, or the whole
        # request came back unrelated. Thresholding on it would cut at random.
        rows = [PathReport(s.source, s.chain, s.full, s.classes, s.relations,
                           s.total, s.complete, True, "no usable scale, kept")
                for s in scores]
        _print_table(rows)
        for score in scores:
            source_scores[score.source] = max(
                source_scores.get(score.source, score.total), score.total)
        all_kept = {_key(score) for score in scores}
        complete_kept = {_key(score) for score in scores if score.complete}
        rewritten = stamp_source_completeness(
            _rewrite(blocks, all_kept, complete_kept), extracted)
        return Selection(rewritten,
                         source_scores, set(), rows)

    kept: Set[_Key] = set()
    reasons: Dict[_Key, str] = {}
    for score in scores:
        if not score.ok:
            kept.add(_key(score))
            reasons[_key(score)] = "score unknown, kept"
        elif score.total >= KEEP_RATIO * best_by_tier.get(score.complete, 0.0):
            kept.add(_key(score))
            reasons[_key(score)] = "at or above tier threshold"
        else:
            reasons[_key(score)] = "below tier threshold"

    rows = sorted(
        (PathReport(s.source, s.chain, s.full, s.classes, s.relations, s.total,
                    s.complete, _key(s) in kept, reasons[_key(s)])
         for s in scores),
        key=lambda r: (not r.complete, -r.total, r.source, r.chain))
    _print_table(rows)

    surviving_scores = [score for score in scores if _key(score) in kept]
    complete_scores = [score for score in surviving_scores if score.complete]
    ranking_pool = complete_scores or surviving_scores
    for score in ranking_pool:
        source_scores[score.source] = max(
            source_scores.get(score.source, score.total), score.total)

    surviving_sources = {score.source for score in surviving_scores}
    cut_sources = {s.source for s in scores} - surviving_sources
    if cut_sources:
        print(f"[selector] source(s) cut, every path below the threshold: "
              f"{', '.join(sorted(cut_sources))}")

    complete_kept = {_key(score) for score in surviving_scores if score.complete}
    rewritten = stamp_source_completeness(
        _rewrite(blocks, kept, complete_kept), extracted)
    return Selection(rewritten, source_scores,
                     cut_sources, rows)
