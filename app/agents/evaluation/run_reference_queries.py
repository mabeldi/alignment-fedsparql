"""Execute every reference.rq and store its bindings beside the query."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import requests


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SERVICE_ENDPOINT = re.compile(r"\bSERVICE\s*<([^>]+)>", re.IGNORECASE)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def coordinator_for(query, override=None):
    if override:
        return override
    match = SERVICE_ENDPOINT.search(query)
    if not match:
        raise ValueError(
            "The query has no SERVICE endpoint; supply --coordinator-url."
        )
    return match.group(1)


# HTTP statuses meaning the server could not take the query, not that it was wrong.
UNAVAILABLE_HTTP_STATUS = {502, 503, 504}


def endpoint_unreachable(exc):
    """True when the endpoint never judged the query (down, restarting, overloaded).

    Fuseki that crashes on one query restarts silently, so the NEXT query gets a
    connection error. Scoring that as a wrong query would blame the model for the
    server, so such executions are marked 'unreachable' and must be run again.
    """
    if isinstance(exc, (requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
        return True
    response = getattr(exc, "response", None)
    return (isinstance(exc, requests.HTTPError) and response is not None
            and response.status_code in UNAVAILABLE_HTTP_STATUS)


def print_totals(statuses):
    """Print the end-of-run counts and return the process exit code."""
    counts = {status: statuses.count(status) for status in ("success", "error", "unreachable")}
    print(f"Finished: {counts['success']} succeeded, {counts['error']} failed, "
          f"{counts['unreachable']} unreachable.", flush=True)
    if counts["unreachable"]:
        print("Unreachable queries were never judged. Check Fuseki, then run this script "
              "again. A query that is unreachable on every attempt may be the one "
              "crashing the server.", flush=True)
    return int(counts["error"] + counts["unreachable"] > 0)


def normalize_term(term):
    """Keep the standard SPARQL JSON term fields in a stable order."""
    if not isinstance(term, dict):
        raise ValueError("A SPARQL binding term is not a JSON object.")
    ordered = {}
    for key in ("type", "value", "datatype", "xml:lang"):
        if key in term:
            ordered[key] = term[key]
    for key in sorted(set(term) - set(ordered)):
        ordered[key] = term[key]
    if "type" not in ordered or "value" not in ordered:
        raise ValueError("A SPARQL binding term lacks 'type' or 'value'.")
    return ordered


def normalize_select_result(payload):
    if not isinstance(payload, dict):
        raise ValueError("The endpoint response is not a JSON object.")
    head = payload.get("head")
    results = payload.get("results")
    if not isinstance(head, dict) or not isinstance(head.get("vars"), list):
        raise ValueError("The endpoint response has no SPARQL head.vars list.")
    if not isinstance(results, dict) or not isinstance(results.get("bindings"), list):
        raise ValueError("The endpoint response has no SPARQL results.bindings list.")

    variables = head["vars"]
    if not all(isinstance(variable, str) for variable in variables):
        raise ValueError("The endpoint returned a non-string result variable.")

    bindings = []
    for raw_binding in results["bindings"]:
        if not isinstance(raw_binding, dict):
            raise ValueError("A SPARQL result binding is not a JSON object.")
        names = [name for name in variables if name in raw_binding]
        names.extend(sorted(set(raw_binding) - set(names)))
        bindings.append({name: normalize_term(raw_binding[name]) for name in names})

    # SPARQL does not guarantee row order without ORDER BY. Stable files make
    # later instance-set comparisons and source-control diffs reproducible.
    bindings.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    return variables, bindings


def question_text(folder):
    path = folder / "question.txt"
    if path.exists():
        return path.read_text(encoding="utf-8-sig").strip()
    return folder.name.replace("_", " ")


def execute_query(query_path, output_path, timeout=120, coordinator_url=None, metadata=None):
    folder = query_path.parent
    query = query_path.read_text(encoding="utf-8-sig")
    started_at = utc_now()
    started = time.perf_counter()
    endpoint = None
    record = {
        "format_version": 1,
        "question": question_text(folder),
        "query_file": query_path.name,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "started_at": started_at,
        "status": "running",
    }
    if metadata:
        record.update(metadata)

    try:
        endpoint = coordinator_for(query, coordinator_url)
        record["coordinator_endpoint"] = endpoint
        response = requests.post(
            endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
        response.raise_for_status()
        variables, bindings = normalize_select_result(response.json())
        record.update(
            status="success",
            variables=variables,
            result_count=len(bindings),
            bindings=bindings,
        )
    except Exception as exc:
        record.update(
            status="unreachable" if endpoint_unreachable(exc) else "error",
            variables=[],
            result_count=None,
            bindings=[],
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        if endpoint:
            record["coordinator_endpoint"] = endpoint

    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    record["finished_at"] = utc_now()
    write_json_atomic(output_path, record)
    return record


def execute_reference(query_path, timeout=120, coordinator_url=None):
    return execute_query(
        query_path,
        query_path.parent / "reference_result.json",
        timeout,
        coordinator_url,
        {"query_kind": "reference"},
    )


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing one folder per question.",
    )
    parser.add_argument(
        "--coordinator-url",
        help="Endpoint that evaluates every federated query. By default, use the first SERVICE endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=120,
        help="HTTP timeout per query in seconds (default: 120).",
    )
    parser.add_argument(
        "--question",
        help="Execute only this question folder, e.g. List_all_the_papers.",
    )
    args = parser.parse_args(argv)

    root = args.results_dir.resolve()
    pattern = f"{args.question}/reference.rq" if args.question else "*/reference.rq"
    query_paths = sorted(root.glob(pattern), key=lambda path: path.parent.name.lower())
    if not query_paths:
        parser.error(f"No {pattern} found directly below {root}")

    statuses = []
    print(f"Executing {len(query_paths)} reference queries from {root}", flush=True)
    for query_path in query_paths:
        result = execute_reference(query_path, args.timeout, args.coordinator_url)
        statuses.append(result["status"])
        detail = (f"{result['result_count']} row(s)" if result["status"] == "success"
                  else result["error"]["message"])
        print(f"  {query_path.parent.name}: {result['status']} - {detail}", flush=True)

    return print_totals(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
