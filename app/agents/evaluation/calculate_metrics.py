"""Score saved reference/generated SELECT results without contacting endpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import re
from statistics import mean

from rdflib import URIRef
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

from run_generated_queries import run_number_from_file
from run_reference_queries import DEFAULT_RESULTS_DIR, write_json_atomic


def set_scores(reference, generated):
    tp = len(reference & generated)
    fp = len(generated - reference)
    fn = len(reference - generated)
    precision = tp / len(generated) if generated else float(not reference)
    recall = tp / len(reference) if reference else float(not generated)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "exact_match": int(reference == generated),
    }


def source_endpoints(query):
    """Read actual SERVICE operators, including prefixed IRIs; ignore comments."""
    if not query.strip():
        return set()
    endpoints = set()

    def visit(value):
        if isinstance(value, CompValue):
            if value.name == "ServiceGraphPattern":
                if not isinstance(value["term"], URIRef):
                    raise ValueError("Variable SERVICE endpoints need explicit source annotations.")
                endpoints.add(str(value["term"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    try:
        algebra = prepareQuery(query).algebra
    except Exception as exc:
        raise ValueError(f"Cannot parse query for source selection: {exc}") from exc
    visit(algebra)
    return endpoints


def term_key(term):
    """Compare the saved value exactly, independent of RDF term metadata."""
    if term is None:
        return None  # Unbound variables are not empty strings.
    if not isinstance(term, dict) or not isinstance(term.get("value"), str):
        raise ValueError("Malformed RDF term in saved bindings.")
    return term["value"]


# Each source keeps its own copy of an instance under the same local id:
# http://cmt-instances#conference-1 and http://edas-instances#conference-1 are
# one conference. Merged by that id, a correct answer scores the same whichever
# source's copy it returns.
INSTANCE_ID = re.compile(r"^https?://[^#/]+-instances#(.+)$")


def same_id_key(term):
    """term_key, with every source's copy of an instance reduced to its shared id."""
    value = term_key(term)
    match = INSTANCE_ID.match(value) if value is not None else None
    return "instance:" + match.group(1) if match else value


def rows_for(result, columns, key=term_key):
    return {tuple(key(row.get(column)) for column in columns)
            for row in result["bindings"]}


def load_execution(query_path, result_path):
    query = query_path.read_text(encoding="utf-8-sig")
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if result.get("query_sha256") != digest:
        raise ValueError(f"{result_path.name}: query hash mismatch; execute the current query again.")
    if result.get("status") == "unreachable":
        raise ValueError(f"{result_path.name}: endpoint unreachable, so the query was never "
                         "judged; check Fuseki and execute it again.")
    if result.get("status") not in ("success", "error"):
        raise ValueError(f"{result_path.name}: execution is not finished.")
    if result["status"] == "success":
        variables, bindings = result.get("variables"), result.get("bindings")
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("Missing or invalid variables list.")
        if len(set(variables)) != len(variables):
            raise ValueError("Duplicate result variable names.")
        if not isinstance(bindings, list) or result.get("result_count") != len(bindings):
            raise ValueError("Saved result_count does not match bindings.")
        for row in bindings:
            if not isinstance(row, dict) or set(row) - set(variables):
                raise ValueError("Bindings contain undeclared variables.")
            for term in row.values():
                term_key(term)
    return query, result


def answer_scores(reference, generated, key=term_key):
    """Strict tuple F1 after a single global column assignment per query.

    Select the assignment maximizing complete-row F1, allowing renamed/reordered
    columns and extra output fields. Never rearrange fields independently per row
    or coalesce UNION branch variables. Save the assignment for inspection.
    `key` decides when two saved values are the same answer.
    """
    ref_columns, gen_columns = reference["variables"], generated["variables"]
    ref_rows = rows_for(reference, ref_columns, key)
    if len(gen_columns) < len(ref_columns):
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0,
            "true_positive": 0, "false_positive": len(rows_for(generated, gen_columns, key)),
            "false_negative": len(ref_rows), "column_mapping": None,
            "reason": "Fewer projected columns than required by the reference.",
        }
    if math.perm(len(gen_columns), len(ref_columns)) > 720:
        raise ValueError("Too many possible column mappings for automatic comparison.")
    best = None
    best_rank = None
    for columns in permutations(gen_columns, len(ref_columns)):
        rows = rows_for(generated, columns, key)
        scores = set_scores(ref_rows, rows)
        # Name matches only break equal-score ties, never determine correctness.
        rank = (scores["f1"], scores["true_positive"],
                sum(a == b for a, b in zip(ref_columns, columns)))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = dict(scores, column_mapping=dict(zip(ref_columns, columns)),
                        ignored_generated_columns=[v for v in gen_columns if v not in columns],
                        generated_unique_rows=len(rows), reference_unique_rows=len(ref_rows))
    return best


def score_question(folder, runs, reference_folder=None):
    reference_folder = reference_folder or folder
    output = {
        "format_version": 1, "question_folder": folder.name,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "selected_runs": runs,
        "policy": {
            "answer_comparison": "distinct complete value tuples; best global column assignment",
            "extra_columns": "ignored after assignment; duplicate projected rows removed",
            "split_union_columns": "not coalesced",
            "rdf_identity": ("saved value strings; ignore type, datatype, and language; instance URIs "
                             "<source>-instances#<id> merged by <id> (one thing held by several "
                             "sources); answer_raw_uri keeps every URI distinct"),
            "source_comparison": "distinct SERVICE endpoint IRIs in query text, not coordinator or runtime contacts",
            "failures": ("execution errors score answer F1=0 and exact_match=0; a query that "
                         "neither parses nor executes scores source F1=0"),
            "unreachable_endpoint": "never judged, so unscored; aggregate withheld until executed again",
            "missing_or_stale_artifacts": "unscored; aggregate withheld until repaired",
        },
        "runs": [],
    }
    try:
        query, reference = load_execution(reference_folder / "reference.rq",
                                          reference_folder / "reference_result.json")
        if reference["status"] != "success":
            raise ValueError("Reference execution failed; cannot score against it.")
        ref_sources = source_endpoints(query)
        output["question"] = reference.get("question", folder.name)
        output["reference"] = {
            "folder": str(reference_folder.resolve()),
            "query_sha256": reference["query_sha256"],
            "result_count": reference["result_count"],
            "variables": reference["variables"], "endpoints": sorted(ref_sources),
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        output.update(status="needs_attention", error=str(exc), aggregate=None)
        return output

    for number in runs:
        record = {"run": number, "status": "scored"}
        try:
            query, generated = load_execution(folder / f"run_{number}.rq",
                                              folder / f"run_{number}_result.json")
            executed = generated["status"] == "success"
            record.update(execution_success=int(executed),
                          query_sha256=generated["query_sha256"],
                          result_count=generated.get("result_count"))
            try:
                gen_sources = source_endpoints(query)
            except Exception as exc:
                record["source_error"] = str(exc)
                # Unreadable AND unrunnable is the model's failure: no usable sources.
                # Executed but unreadable means this scorer's parser failed instead.
                gen_sources = None if executed else set()
                if executed:
                    record["status"] = "needs_attention"
            record["source_selection"] = None if gen_sources is None else dict(
                set_scores(ref_sources, gen_sources), endpoints=sorted(gen_sources),
                missing_endpoints=sorted(ref_sources - gen_sources),
                extra_endpoints=sorted(gen_sources - ref_sources))
            # "answer" counts every source's copy of an instance as one; "answer_raw_uri"
            # does not, so it still drops when one source branch returned nothing.
            for field, key in (("answer", same_id_key), ("answer_raw_uri", term_key)):
                if executed:
                    record[field] = answer_scores(reference, generated, key)
                else:
                    record[field] = dict(
                        set_scores(rows_for(reference, reference["variables"], key), set()),
                        precision=0.0, recall=0.0, f1=0.0, exact_match=0, column_mapping=None)
            if not executed:
                record["execution_error"] = generated.get("error")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record.update(status="needs_attention", error=str(exc))
        output["runs"].append(record)

    complete = all(r["status"] == "scored" for r in output["runs"])
    output["status"] = "complete" if complete else "needs_attention"
    output["aggregate"] = None
    if complete:
        output["aggregate"] = {
            "run_count": len(runs),
            "answer_precision_mean": mean(r["answer"]["precision"] for r in output["runs"]),
            "answer_recall_mean": mean(r["answer"]["recall"] for r in output["runs"]),
            "answer_f1_mean": mean(r["answer"]["f1"] for r in output["runs"]),
            "answer_f1_raw_uri_mean": mean(r["answer_raw_uri"]["f1"] for r in output["runs"]),
            "exact_match_accuracy": mean(r["answer"]["exact_match"] for r in output["runs"]),
            "source_selection_f1_mean": mean(r["source_selection"]["f1"] for r in output["runs"]),
            "execution_success_rate": mean(r["execution_success"] for r in output["runs"]),
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--reference-dir", type=Path,
                        help="Optional directory with reference files in matching question folders.")
    parser.add_argument("--run", type=int, action="append", help="Repeat to select runs (default: 1, 2, 3).")
    parser.add_argument("--query", dest="run", metavar="FILE", type=run_number_from_file,
                        action="append",
                        help="Score only this query file, e.g. run_1.rq; same as --run 1.")
    parser.add_argument("--question", help="Score only this question folder, e.g. List_all_the_papers.")
    args = parser.parse_args(argv)
    runs = sorted(set(args.run or [1, 2, 3]))
    if any(r < 1 for r in runs):
        parser.error("Run numbers must be positive.")
    root = args.results_dir.resolve()
    folders = sorted(p for p in root.iterdir() if p.is_dir() and
                     (args.question is None or p.name == args.question) and
                     (any(p.glob("run_*.rq")) or (p / "reference_result.json").exists()))
    if not folders:
        parser.error(f"No question folders found in {root}"
                     + (f" named {args.question}" if args.question else ""))
    failed = 0
    for folder in folders:
        reference_folder = args.reference_dir / folder.name if args.reference_dir else folder
        metrics = score_question(folder, runs, reference_folder)
        write_json_atomic(folder / "metrics.json", metrics)
        aggregate = metrics["aggregate"]
        if aggregate is None:
            failed += 1
            print(f"{folder.name}: needs attention (see metrics.json)", flush=True)
        else:
            print(f"{folder.name}: F1={aggregate['answer_f1_mean']:.3f}, "
                  f"raw-URI F1={aggregate['answer_f1_raw_uri_mean']:.3f}, "
                  f"exact={aggregate['exact_match_accuracy']:.3f}, "
                  f"source F1={aggregate['source_selection_f1_mean']:.3f}, "
                  f"executed={aggregate['execution_success_rate']:.3f}", flush=True)
    print(f"Wrote {len(folders)} metrics.json files; {failed} need attention.")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
