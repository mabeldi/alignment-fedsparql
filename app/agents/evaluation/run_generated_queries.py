"""Execute generated run_N.rq files and store run_N_result.json beside each."""

import argparse
from pathlib import Path
import re

from run_reference_queries import DEFAULT_RESULTS_DIR, execute_query, positive_float, print_totals


RUN_FILE = re.compile(r"run_(\d+)\.rq", re.IGNORECASE)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def run_number_from_file(value):
    """The run number of a query file name: run_1.rq -> 1."""
    match = RUN_FILE.fullmatch(Path(value).name)
    if not match:
        raise argparse.ArgumentTypeError("Expected a query file name like run_1.rq.")
    return positive_int(match.group(1))


def discover_runs(root, selected_runs=None, question=None):
    selected = set(selected_runs or [])
    jobs = []
    search_root = root / question if question else root
    for path in search_root.rglob("run_*.rq"):
        match = RUN_FILE.fullmatch(path.name)
        if not match:
            continue
        run_number = int(match.group(1))
        if selected and run_number not in selected:
            continue
        jobs.append((path, run_number))
    return sorted(
        jobs,
        key=lambda item: (str(item[0].parent.relative_to(root)).lower(), item[1]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing generated query files.",
    )
    parser.add_argument(
        "--question",
        help="Execute only this question folder, e.g. List_all_the_papers.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=positive_int,
        help="Execute only this run number; may be repeated. The default executes all runs.",
    )
    parser.add_argument(
        "--query",
        dest="run",
        metavar="FILE",
        action="append",
        type=run_number_from_file,
        help="Execute only this query file, e.g. run_1.rq; may be repeated. Same as --run 1.",
    )
    parser.add_argument(
        "--coordinator-url",
        help="Endpoint that evaluates every query. By default, use its first SERVICE endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=120,
        help="HTTP timeout per query in seconds (default: 120).",
    )
    args = parser.parse_args(argv)

    root = args.results_dir.resolve()
    jobs = discover_runs(root, args.run, args.question)
    if not jobs:
        parser.error(f"No matching run_N.rq files found below {root / (args.question or '')}")

    statuses = []
    print(f"Executing {len(jobs)} generated queries from {root}", flush=True)
    for query_path, run_number in jobs:
        output_path = query_path.with_name(f"run_{run_number}_result.json")
        generation_metadata = query_path.with_suffix(".json")
        metadata = {
            "query_kind": "generated",
            "run": run_number,
            "generation_metadata_file": (
                generation_metadata.name if generation_metadata.exists() else None
            ),
        }
        result = execute_query(
            query_path,
            output_path,
            args.timeout,
            args.coordinator_url,
            metadata,
        )
        relative = query_path.relative_to(root)
        statuses.append(result["status"])
        detail = (f"{result['result_count']} row(s)" if result["status"] == "success"
                  else result["error"]["message"])
        print(f"  {relative}: {result['status']} - {detail}", flush=True)

    return print_totals(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
