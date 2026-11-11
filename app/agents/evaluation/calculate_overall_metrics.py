"""Aggregate question metrics with equal question weights; optionally compare an archive."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from calculate_metrics import score_question
from run_reference_queries import DEFAULT_RESULTS_DIR, write_json_atomic


FIELDS = {
    "answer_macro_f1": ("answer", "f1", "answer_f1_mean"),
    "answer_macro_f1_raw_uri": ("answer_raw_uri", "f1", "answer_f1_raw_uri_mean"),
    "exact_match_accuracy": ("answer", "exact_match", "exact_match_accuracy"),
    "source_selection_macro_f1": ("source_selection", "f1", "source_selection_f1_mean"),
    "execution_success_rate": (None, "execution_success", "execution_success_rate"),
}
WILCOXON = ("Wilcoxon signed-rank, exact, two-sided, over per-question scores; "
            "zero differences dropped, tied magnitudes share their average rank")


def validate_question(data, runs):
    """Per-run values of every metric, checked against the stored aggregate."""
    if data.get("status") != "complete" or data.get("selected_runs") != runs:
        raise ValueError("Incomplete metrics or different selected runs; recalculate question metrics.")
    records = data["runs"]
    if [r["run"] for r in records] != runs or data["aggregate"]["run_count"] != len(runs):
        raise ValueError("Run list/count does not match the selected evaluation runs.")
    run_values = {}
    for name, (section, field, aggregate_field) in FIELDS.items():
        values = [(r[section] if section else r)[field] for r in records]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in values):
            raise ValueError(f"Invalid values for {name}.")
        if not math.isclose(mean(values), data["aggregate"][aggregate_field], abs_tol=1e-12):
            raise ValueError(f"Stored aggregate disagrees with per-run values for {name}.")
        run_values[name] = values
    return run_values


def spread_and_stability(questions, runs):
    """How far the overall scores move between repeated runs, and how many
    questions are answered exactly every time rather than only sometimes."""
    per_run = {str(run): {name: mean(q["run_scores"][name][i] for q in questions)
                          for name in FIELDS}
               for i, run in enumerate(runs)}
    spread = {}
    for name in FIELDS:
        values = [scores[name] for scores in per_run.values()]
        spread[name] = {"min": min(values), "max": max(values),
                        "stdev": stdev(values) if len(values) > 1 else None}
    exact = [q["run_scores"]["exact_match_accuracy"] for q in questions]
    return {
        "per_run_overall": per_run,
        "run_spread": spread,
        "stability": {
            "question_count": len(questions),
            "exact_in_every_run": sum(all(v == 1 for v in values) for values in exact),
            "exact_in_at_least_one_run": sum(any(v == 1 for v in values) for values in exact),
        },
    }


def summarize(root, model, alignment, runs, reference_dir=None):
    report = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {"model": model, "alignments_enabled": alignment,
                          "label_source": "user-declared command arguments"},
        "selected_runs": runs,
        "aggregation": "mean across runs within each question, then equal-weight mean across questions",
        "spread": ("per_run_overall averages each run over all questions; run_spread gives "
                   "their min, max and sample standard deviation"),
        "score_scale": "0 to 1", "questions": [], "errors": [], "overall": None,
    }
    folders = sorted(p for p in root.iterdir() if p.is_dir() and
                     any((p / name).exists() for name in ("question.txt", "reference.rq", "metrics.json")))
    report["question_count"] = len(folders)
    report["attempt_count"] = len(folders) * len(runs)
    if not folders:
        report["errors"].append({"error": "No question folders found."})
    policy = None
    for folder in folders:
        try:
            path = folder / "metrics.json"
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8-sig"))
            run_values = validate_question(data, runs)
            if policy is not None and data["policy"] != policy:
                raise ValueError("Question metrics use different comparison policies.")
            policy = data["policy"]
            reference_folder = reference_dir / folder.name if reference_dir else folder
            fresh = score_question(folder, runs, reference_folder)
            if (fresh["status"] != "complete" or fresh["runs"] != data["runs"]
                    or fresh.get("aggregate") != data["aggregate"]
                    or fresh.get("reference", {}).get("query_sha256") != data["reference"]["query_sha256"]):
                raise ValueError("Metrics are stale or artifacts incomplete; rerun calculate_metrics.py.")
            report["questions"].append({
                "folder": folder.name, "question": data["question"],
                "metrics_sha256": hashlib.sha256(raw).hexdigest(),
                "reference_query_sha256": data["reference"]["query_sha256"],
                "reference_result_sha256": hashlib.sha256(
                    (reference_folder / "reference_result.json").read_bytes()).hexdigest(),
                "scores": {name: mean(values) for name, values in run_values.items()},
                "run_scores": run_values,
                "run_range": {name: [min(values), max(values)] for name, values in run_values.items()},
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append({"folder": folder.name, "error": str(exc)})
    report["policy"] = policy
    report["status"] = "needs_attention" if report["errors"] else "complete"
    if report["status"] == "complete":
        report["overall"] = {name: mean(q["scores"][name] for q in report["questions"])
                             for name in FIELDS}
        report.update(spread_and_stability(report["questions"], runs))
    return report


def wilcoxon_signed_rank(differences):
    """Exact two-sided Wilcoxon signed-rank test on paired per-question differences.

    Every +/- sign pattern of the ranks is counted instead of using a normal
    approximation, which is cheap at benchmark size. Ranks are doubled so the
    half ranks that ties produce stay integers.
    """
    nonzero = [d for d in differences if not math.isclose(d, 0.0, abs_tol=1e-12)]
    result = {"method": WILCOXON, "pairs": len(differences), "nonzero_pairs": len(nonzero)}
    if not nonzero:
        return dict(result, statistic=0.0, p_value=1.0)
    order = sorted(range(len(nonzero)), key=lambda i: abs(nonzero[i]))
    doubled_ranks = [0] * len(nonzero)
    start = 0
    while start < len(order):
        end = start
        while (end + 1 < len(order) and math.isclose(
                abs(nonzero[order[end + 1]]), abs(nonzero[order[start]]), abs_tol=1e-12)):
            end += 1
        for position in range(start, end + 1):
            doubled_ranks[order[position]] = start + end + 2  # twice the average rank
        start = end + 1
    observed = sum(rank for rank, d in zip(doubled_ranks, nonzero) if d > 0)
    total = sum(doubled_ranks)
    sums = {0: 1}  # rank sum of the positive signs -> number of sign patterns
    for rank in doubled_ranks:
        extended = dict(sums)
        for value, count in sums.items():
            extended[value + rank] = extended.get(value + rank, 0) + count
        sums = extended
    distance = abs(2 * observed - total)
    extreme = sum(count for value, count in sums.items() if abs(2 * value - total) >= distance)
    return dict(result, statistic=observed / 2,
                p_value=min(1.0, extreme / 2 ** len(nonzero)))


def comparison(current, previous):
    if current["status"] != "complete" or previous.get("status") != "complete":
        raise ValueError("Both summaries must be complete to compare.")
    for key in ("selected_runs", "policy"):
        if current[key] != previous[key]:
            raise ValueError(f"Cannot compare summaries with different {key}.")
    if current["configuration"]["model"] != previous["configuration"]["model"]:
        raise ValueError("An alignment ablation requires the same model.")
    def signature(report):
        return {q["folder"]: (q["question"], q["reference_query_sha256"], q["reference_result_sha256"])
                for q in report["questions"]}
    if signature(current) != signature(previous):
        raise ValueError("Question sets or reference snapshots differ.")
    current_scores = {q["folder"]: q["scores"] for q in current["questions"]}
    previous_scores = {q["folder"]: q["scores"] for q in previous["questions"]}
    return {
        "previous_configuration": previous["configuration"],
        "previous_overall": previous["overall"],
        "previous_run_spread": previous.get("run_spread"),
        "difference_current_minus_previous": {
            name: current["overall"][name] - previous["overall"][name] for name in FIELDS},
        "paired_test": {
            name: wilcoxon_signed_rank([current_scores[f][name] - previous_scores[f][name]
                                        for f in sorted(current_scores)])
            for name in FIELDS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--model", required=True, help="Model used to generate these runs.")
    parser.add_argument("--alignment", required=True, choices=("on", "off"))
    parser.add_argument("--run", type=int, action="append", help="Repeat to select runs; default 1, 2, 3.")
    parser.add_argument("--compare-with", type=Path, help="Archived overall-metrics.json from the other condition.")
    args = parser.parse_args(argv)
    runs = sorted(set(args.run or [1, 2, 3]))
    if not runs or min(runs) < 1:
        parser.error("Run numbers must be positive.")
    root = args.results_dir.resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")
    report = summarize(root, args.model, args.alignment == "on", runs, args.reference_dir)
    if args.compare_with:
        try:
            previous = json.loads(args.compare_with.read_text(encoding="utf-8-sig"))
            report["comparison"] = comparison(report, previous)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["comparison"] = {"status": "needs_attention", "error": str(exc)}
    output = root / "overall-metrics.json"
    write_json_atomic(output, report)
    print(f"Wrote {output}: {report['status']}")
    if report["overall"] is not None:
        for name, value in report["overall"].items():
            spread = report["run_spread"][name]
            print(f"  {name}: {value:.4f} (runs {spread['min']:.4f}-{spread['max']:.4f})")
        stability = report["stability"]
        print(f"  exact in every run: {stability['exact_in_every_run']}/{stability['question_count']}, "
              f"in at least one run: {stability['exact_in_at_least_one_run']}/{stability['question_count']}")
    for error in report["errors"]:
        print(f"  {error}")
    comparison_failed = report.get("comparison", {}).get("status") == "needs_attention"
    if comparison_failed:
        print(report["comparison"]["error"])
    elif "comparison" in report:
        for name, test in report["comparison"]["paired_test"].items():
            difference = report["comparison"]["difference_current_minus_previous"][name]
            print(f"  vs previous {name}: {difference:+.4f} (p={test['p_value']:.4f})")
    return int(report["status"] != "complete" or comparison_failed)


if __name__ == "__main__":
    raise SystemExit(main())
