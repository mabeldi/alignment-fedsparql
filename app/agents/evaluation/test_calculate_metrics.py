"""Offline checks for answer identity, endpoint extraction, and evaluation errors."""

import contextlib
import hashlib
import io
import json
import unittest

from calculate_metrics import (answer_scores, main, same_id_key, score_question, set_scores,
                               source_endpoints, term_key)
from test_run_reference_queries import temporary_results


def result(variables, rows):
    return {"variables": variables, "bindings": [
        {key: {"type": "uri", "value": value} for key, value in row.items()}
        for row in rows]}


QUERY = "SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x ?p ?o } }"


def save(folder, stem, rows=None, status="success", query=QUERY):
    (folder / f"{stem}.rq").write_text(query, encoding="utf-8")
    payload = result(["x"], rows or [])
    payload.update(status=status, result_count=len(payload["bindings"]),
                   query_sha256=hashlib.sha256(query.encode()).hexdigest())
    (folder / f"{stem}_result.json").write_text(json.dumps(payload), encoding="utf-8")


class MetricTests(unittest.TestCase):
    def test_one_question_and_one_query_can_be_scored_alone(self):
        with temporary_results() as root:
            for name in ("A", "B"):
                folder = root / name
                folder.mkdir()
                save(folder, "reference", [{"x": "urn:a"}])
                save(folder, "run_1", [{"x": "urn:a"}])
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--results-dir", str(root), "--question", "A", "--query", "run_1.rq"])
            self.assertEqual(code, 0)
            metrics = json.loads((root / "A" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["selected_runs"], [1])
            self.assertEqual(metrics["aggregate"]["answer_f1_mean"], 1.0)
            self.assertFalse((root / "B" / "metrics.json").exists())

    def test_partial_matches_and_equal_counts_are_not_accuracy(self):
        scores = set_scores(set(range(10)), set(range(6)) | {20, 21})
        self.assertEqual(scores["precision"], .75)
        self.assertEqual(scores["recall"], .6)
        self.assertAlmostEqual(scores["f1"], 2 / 3)
        self.assertEqual(set_scores({1, 2}, {3, 4})["f1"], 0)
        self.assertEqual(set_scores(set(), set())["exact_match"], 1)

    def test_global_renaming_reordering_extra_columns_and_duplicates(self):
        ref = result(["person", "event"], [{"person": "urn:a", "event": "urn:b"}])
        gen = result(["other", "y", "x"], [
            {"x": "urn:a", "y": "urn:b", "other": "urn:1"},
            {"x": "urn:a", "y": "urn:b", "other": "urn:2"}])
        score = answer_scores(ref, gen)
        self.assertEqual(score["f1"], 1)
        self.assertEqual(score["column_mapping"], {"person": "x", "event": "y"})
        self.assertEqual(score["generated_unique_rows"], 1)

    def test_tuple_relations_preserved_and_missing_columns_fail(self):
        ref = result(["a", "b"], [{"a": "urn:1", "b": "urn:2"},
                                    {"a": "urn:3", "b": "urn:4"}])
        gen = result(["x", "y"], [{"x": "urn:1", "y": "urn:4"},
                                    {"x": "urn:3", "y": "urn:2"}])
        self.assertEqual(answer_scores(ref, gen)["f1"], 0)
        self.assertEqual(answer_scores(ref, result(["x"], [{"x": "urn:1"}]))["f1"], 0)

    def test_value_identity_ignores_all_term_metadata(self):
        self.assertEqual(term_key({"type": "uri", "value": "a"}),
                            term_key({"type": "literal", "value": "a"}))
        self.assertEqual(term_key({"type": "literal", "value": "a", "xml:lang": "EN"}),
                         term_key({"type": "literal", "value": "a", "xml:lang": "fr"}))
        self.assertEqual(term_key({"type": "bnode", "value": "b0"}), "b0")
        self.assertNotEqual(term_key(None), term_key({"value": ""}))
        self.assertNotEqual(term_key({"value": "01"}), term_key({"value": "1"}))

    def test_typed_and_untyped_dates_match_but_different_values_do_not(self):
        ref = {"variables": ["deadline"], "bindings": [
            {"deadline": {"type": "literal", "value": "2012-06-28T21:00:00"}}]}
        gen = {"variables": ["date"], "bindings": [
            {"date": {"type": "literal", "value": "2012-06-28T21:00:00",
                      "datatype": "http://www.w3.org/2001/XMLSchema#dateTime"}},
            {"date": {"type": "literal", "value": "2012-06-28T21:00:00"}}]}
        self.assertEqual(answer_scores(ref, gen)["exact_match"], 1)
        self.assertEqual(answer_scores(ref, gen)["generated_unique_rows"], 1)
        gen["bindings"].append({"date": {"value": "2012-06-29T21:00:00"}})
        self.assertAlmostEqual(answer_scores(ref, gen)["f1"], 2 / 3)

    def test_sources_are_parsed_not_scraped(self):
        query = '''PREFIX ep: <http://example.org/>
        # SERVICE <http://wrong.example/sparql>
        SELECT ?x WHERE {
          SERVICE SILENT ep:a { ?x ?p ?o }
          SERVICE <http://example.org/b> { ?x ?p ?o }
          SERVICE ep:a { ?x ?p ?o }
        }'''
        self.assertEqual(source_endpoints(query), {"http://example.org/a", "http://example.org/b"})
        with self.assertRaises(ValueError):
            source_endpoints("SELECT broken")

    def test_missing_stale_and_failed_executions(self):
        with temporary_results() as folder:
            save(folder, "reference", [{"x": "urn:a"}])
            save(folder, "run_1", [{"x": "urn:a"}])
            save(folder, "run_2", status="error")
            metrics = score_question(folder, [1, 2])
            self.assertEqual(metrics["aggregate"]["answer_f1_mean"], .5)
            self.assertEqual(metrics["aggregate"]["execution_success_rate"], .5)
            self.assertIsNone(score_question(folder, [1, 2, 3])["aggregate"])
            (folder / "run_1.rq").write_text("changed", encoding="utf-8")
            self.assertIsNone(score_question(folder, [1])["aggregate"])

    def test_same_id_in_different_sources_is_one_answer(self):
        ref = result(["x"], [{"x": "http://cmt-instances#conference-1"},
                             {"x": "http://edas-instances#conference-1"}])
        gen = result(["y"], [{"y": "http://ekaw-instances#conference-1"}])
        self.assertEqual(answer_scores(ref, gen, same_id_key)["exact_match"], 1)
        self.assertEqual(answer_scores(ref, gen)["f1"], 0)
        # Schema-namespace URIs and literals are not source copies; they stay as saved.
        self.assertEqual(same_id_key({"value": "http://edas#evalWithdrawnPaper1"}),
                         "http://edas#evalWithdrawnPaper1")
        self.assertNotEqual(same_id_key({"value": "conference-1"}),
                            same_id_key({"value": "http://cmt-instances#conference-1"}))

    def test_raw_uri_score_still_sees_a_source_that_returned_nothing(self):
        with temporary_results() as folder:
            save(folder, "reference", [{"x": "http://cmt-instances#p1"},
                                       {"x": "http://edas-instances#p1"}])
            save(folder, "run_1", [{"x": "http://cmt-instances#p1"}])
            run = score_question(folder, [1])["runs"][0]
            self.assertEqual(run["answer"]["f1"], 1)
            self.assertAlmostEqual(run["answer_raw_uri"]["f1"], 2 / 3)

    def test_broken_query_scores_zero_instead_of_blocking(self):
        broken = "SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x"
        with temporary_results() as folder:
            save(folder, "reference", [{"x": "urn:a"}])
            save(folder, "run_1", [{"x": "urn:a"}])
            save(folder, "run_2", status="error", query=broken)
            metrics = score_question(folder, [1, 2])
            self.assertEqual(metrics["status"], "complete")
            self.assertEqual(metrics["runs"][1]["source_selection"]["f1"], 0)
            self.assertEqual(metrics["aggregate"]["source_selection_f1_mean"], .5)
            # Executed by the endpoint yet unreadable here: the scorer is at fault.
            save(folder, "run_2", [{"x": "urn:a"}], query=broken)
            self.assertIsNone(score_question(folder, [1, 2])["aggregate"])

    def test_unreachable_endpoint_is_not_scored(self):
        with temporary_results() as folder:
            save(folder, "reference", [{"x": "urn:a"}])
            save(folder, "run_1", status="unreachable")
            metrics = score_question(folder, [1])
            self.assertIsNone(metrics["aggregate"])
            self.assertIn("unreachable", metrics["runs"][0]["error"])


if __name__ == "__main__":
    unittest.main()
