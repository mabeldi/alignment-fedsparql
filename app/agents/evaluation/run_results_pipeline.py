"""Execute references, generated queries, and metrics for every root question folder."""

import argparse
from pathlib import Path

from calculate_metrics import main as calculate_metrics
from run_generated_queries import main as run_generated_queries
from run_reference_queries import DEFAULT_RESULTS_DIR, main as run_reference_queries, positive_float


EXCLUDED_FOLDERS = {"all_models"}


def discover_questions(root):
    """Return evaluable question folders directly below root in stable order."""
    folders = []
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.casefold() in EXCLUDED_FOLDERS:
            continue
        if (folder / "reference.rq").is_file() and any(folder.glob("run_*.rq")):
            folders.append(folder)
    return sorted(folders, key=lambda folder: folder.name.casefold())


def stage_arguments(root, folder, timeout, coordinator_url, runs):
    common = ["--results-dir", str(root), "--question", folder.name]
    endpoint = (["--coordinator-url", coordinator_url] if coordinator_url else [])
    selected_runs = [item for run in runs for item in ("--run", str(run))]
    return (
        common + ["--timeout", str(timeout)] + endpoint,
        common + ["--timeout", str(timeout)] + endpoint + selected_runs,
        common + selected_runs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--coordinator-url",
                        help="Endpoint used as coordinator for every federated query.")
    parser.add_argument("--timeout", type=positive_float, default=120,
                        help="HTTP timeout per query in seconds (default: 120).")
    parser.add_argument("--run", type=int, action="append",
                        help=("Repeat to execute and score selected runs. By default, execute "
                              "all available runs and score runs 1, 2, and 3."))
    parser.add_argument("--list", action="store_true",
                        help="Show the question order without executing anything.")
    args = parser.parse_args(argv)

    runs = sorted(set(args.run or []))
    if any(run < 1 for run in runs):
        parser.error("Run numbers must be positive.")
    root = args.results_dir.resolve()
    try:
        folders = discover_questions(root)
    except OSError as exc:
        parser.error(str(exc))
    if not folders:
        parser.error(f"No question folders with reference.rq and run_N.rq found in {root}")

    print(f"Evaluating {len(folders)} questions sequentially from {root}", flush=True)
    failed_questions = []
    for index, folder in enumerate(folders, 1):
        print(f"\n[{index}/{len(folders)}] {folder.name}", flush=True)
        if args.list:
            continue
        reference_args, generated_args, metrics_args = stage_arguments(
            root, folder, args.timeout, args.coordinator_url, runs
        )
        stage_codes = []
        try:
            print("  1/3 reference query", flush=True)
            stage_codes.append(run_reference_queries(reference_args))
            print("  2/3 generated queries", flush=True)
            stage_codes.append(run_generated_queries(generated_args))
            print("  3/3 metrics", flush=True)
            stage_codes.append(calculate_metrics(metrics_args))
        except Exception as exc:
            stage_codes.append(1)
            print(f"  pipeline error: {type(exc).__name__}: {exc}", flush=True)
        if any(stage_codes):
            failed_questions.append(folder.name)

    if args.list:
        return 0
    print(f"\nFinished: {len(folders) - len(failed_questions)} complete, "
          f"{len(failed_questions)} need attention.", flush=True)
    if failed_questions:
        print("Need attention: " + ", ".join(failed_questions), flush=True)
    return int(bool(failed_questions))


if __name__ == "__main__":
    raise SystemExit(main())
