"""Generate queries sequentially from existing question.txt files."""

import argparse
from pathlib import Path

from run_evaluation import DEFAULT_OUTPUT, positive_int, reserve_runs, run_once


def load_folder_questions(root):
    questions = []
    for marker in sorted(root.glob("*/question.txt"), key=lambda p: p.parent.name.lower()):
        question = marker.read_text(encoding="utf-8-sig").strip()
        if not question:
            raise ValueError(f"Empty question: {marker}")
        questions.append((marker.parent, question))
    if not questions:
        raise ValueError(f"No question.txt files found directly below {root}")
    return questions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--url", default="http://127.0.0.1:8999/query")
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--list", action="store_true", help="Preview question order without generating queries.")
    args = parser.parse_args(argv)
    args.parallel = 1
    try:
        questions = load_folder_questions(args.results_dir.resolve())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{len(questions)} questions x {args.runs} runs; one request at a time.", flush=True)
    generated = failed = 0
    for index, (folder, question) in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question}", flush=True)
        if args.list:
            continue
        for directory, metadata in reserve_runs(folder, question, args):
            result = run_once(directory, metadata, args.timeout)
            success = result["status"] == "generated"
            generated += int(success)
            failed += int(not success)
            print(f"  run_{result['run']}: {result['status']} ({result['elapsed_seconds']:.1f}s)",
                  flush=True)
    if not args.list:
        print(f"Finished: {generated} generated, {failed} failed.", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
