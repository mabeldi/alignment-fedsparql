"""Repeat questions against the running /query API and archive each response."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def load_questions(args):
    questions = list(args.question or [])
    if args.questions_file:
        path = args.questions_file
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            items = json.loads(text)
            if isinstance(items, dict):
                items = items.get("questions")
            if not isinstance(items, list):
                raise ValueError("JSON must be a list or an object with a 'questions' list.")
            for item in items:
                question = item.get("question") if isinstance(item, dict) else item
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("Every JSON item must contain a nonempty question.")
                questions.append(question)
        else:
            questions.extend(text.splitlines())
    # Duplicate input lines are one question; --runs controls repetitions.
    return list(dict.fromkeys(q.strip() for q in questions if q.strip()))


def question_folder(output, question):
    slug = re.sub(r"[^\w]+", "_", question, flags=re.UNICODE).strip("_")
    slug = slug[:100] or "Question"
    if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", slug):
        slug = "Question_" + slug
    folder = output / slug
    marker = folder / "question.txt"
    if folder.exists() and (not marker.exists() or
                            marker.read_text(encoding="utf-8").strip() != question):
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        folder = output / f"{slug}_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "question.txt"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(question + "\n")
    except FileExistsError:
        if marker.read_text(encoding="utf-8").strip() != question:
            raise ValueError(f"Question folder collision: {folder}")
    return folder


def reserve_runs(folder, question, args):
    """Reserve new numbers so a later invocation never replaces an attempt."""
    numbers = [int(match.group(1)) for path in folder.glob("run_*.*")
               if (match := re.fullmatch(r"run_(\d+)\.(?:rq|json)", path.name))]
    number = max(numbers, default=0) + 1
    jobs = []
    while len(jobs) < args.runs:
        metadata = {
            "question": question, "run": number, "status": "pending",
            "endpoint": args.url, "parallel_requests": args.parallel,
            "server_diagnostics_note": (
                "Server token counters and debug files are shared across requests; "
                "do not use them for per-run measurements during concurrent requests."
            ),
        }
        path = folder / f"run_{number}.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
        except FileExistsError:
            number += 1
            continue
        jobs.append((folder, metadata))
        number += 1
    return jobs


def run_once(folder, metadata, timeout):
    metadata = dict(metadata)
    metadata["started_at"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "running"
    json_path = folder / f"run_{metadata['run']}.json"
    write_json(json_path, metadata)
    start = time.perf_counter()
    query = ""
    try:
        request = Request(
            metadata["endpoint"],
            data=json.dumps({"question": metadata["question"]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            metadata["http_status"] = response.status
            payload = json.loads(response.read().decode("utf-8"))
        metadata["response"] = payload
        if not isinstance(payload, dict):
            raise ValueError("The API response is not a JSON object.")
        if payload.get("question", metadata["question"]) != metadata["question"]:
            raise ValueError("The returned question does not match the request.")
        value = payload.get("query")
        if isinstance(value, str) and value.strip():
            query = value.strip() + "\n"
            metadata["status"] = "generated"
        else:
            metadata["status"] = "empty_query"
            metadata["error"] = "The pipeline returned no query."
    except HTTPError as exc:
        metadata.update(status="http_error", http_status=exc.code,
                        error=str(exc), response_body=exc.read().decode("utf-8", errors="replace"))
    except (URLError, TimeoutError, OSError) as exc:
        metadata.update(status="request_error", error=str(exc))
    except (ValueError, TypeError) as exc:
        metadata.update(status="invalid_response", error=str(exc))
    metadata["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    # Empty .rq files preserve the attempt number on failures; JSON explains why.
    (folder / f"run_{metadata['run']}.rq").write_text(query, encoding="utf-8")
    write_json(json_path, metadata)
    return metadata


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append", help="Question text; may be repeated.")
    parser.add_argument("--questions-file", type=Path,
                        help="UTF-8 text (one question per line) or JSON questions list.")
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--parallel", type=positive_int, default=5,
                        help="Concurrent requests per question (default: 5).")
    parser.add_argument("--url", default="http://127.0.0.1:8999/query")
    parser.add_argument("--timeout", type=positive_int, default=600,
                        help="HTTP timeout in seconds (default: 600). No automatic retries.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not questions:
        parser.error("Supply --question or --questions-file with at least one question.")
    args.parallel = min(args.parallel, args.runs)
    print(f"{len(questions)} questions x {args.runs} runs; up to {args.parallel} concurrent requests.",
          flush=True)
    if args.parallel > 1:
        print("Parallel runs: shared server token counters/debug dumps are not per-run diagnostics.",
              flush=True)
    generated = failed = 0
    # Finish all repetitions of one question before starting the next question.
    for question in questions:
        folder = question_folder(args.output_dir.resolve(), question)
        jobs = reserve_runs(folder, question, args)
        print(f"\n{question}\nOutput: {folder}", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_once, directory, metadata, args.timeout)
                       for directory, metadata in jobs]
            for future in as_completed(futures):
                result = future.result()
                success = result["status"] == "generated"
                generated += int(success)
                failed += int(not success)
                print(f"  run_{result['run']}: {result['status']} "
                      f"({result['elapsed_seconds']:.1f}s)", flush=True)
    print(f"\nFinished: {generated} generated, {failed} failed. "
          "Generated queries still need correctness evaluation.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
