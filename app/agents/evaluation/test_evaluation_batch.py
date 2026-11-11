"""Offline checks for sequential generation and overall aggregation."""

import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

from calculate_metrics import score_question
from calculate_overall_metrics import comparison, summarize, wilcoxon_signed_rank
import run_all_questions
from test_calculate_metrics import save
from test_run_reference_queries import temporary_results


class BatchTests(unittest.TestCase):
    def test_three_requests_per_folder_in_order_and_reference_preserved(self):
        calls = []

        def fake_api(request, timeout):
            question = json.loads(request.data)["question"]
            calls.append(question)
            response = io.BytesIO(json.dumps({"question": question, "query": "SELECT * WHERE {}"}).encode())
            response.status = 200
            return response

        with temporary_results() as root:
            for name, question in (("B", "Second?"), ("A", "First?")):
                folder = root / name
                folder.mkdir()
                (folder / "question.txt").write_text(question, encoding="utf-8-sig")
                (folder / "reference.rq").write_text("reference", encoding="utf-8")
            with patch("run_evaluation.urlopen", side_effect=fake_api), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_all_questions.main(["--results-dir", str(root)]), 0)
            self.assertEqual(calls, ["First?"] * 3 + ["Second?"] * 3)
            for folder in (root / "A", root / "B"):
                self.assertEqual((folder / "reference.rq").read_text(), "reference")
                for run in range(1, 4):
                    record = json.loads((folder / f"run_{run}.json").read_text())
                    self.assertEqual(record["parallel_requests"], 1)
            with patch("run_evaluation.urlopen") as api, contextlib.redirect_stdout(io.StringIO()):
                run_all_questions.main(["--results-dir", str(root), "--list"])
                api.assert_not_called()

    def make_question(self, root, name, reference_rows, generated_rows):
        folder = root / name
        folder.mkdir()
        save(folder, "reference", [{"x": f"urn:{i}"} for i in range(reference_rows)])
        for number, count in enumerate(generated_rows, 1):
            save(folder, f"run_{number}", [{"x": f"urn:{i}"} for i in range(count)])
        metrics = score_question(folder, list(range(1, len(generated_rows) + 1)))
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return folder

    def test_equal_question_weighting_not_weighted_by_returned_rows(self):
        with temporary_results() as root:
            self.make_question(root, "A", 1, [1, 0, 1])
            self.make_question(root, "B", 100, [100, 100, 100])
            report = summarize(root, "luna", True, [1, 2, 3])
            self.assertEqual(report["status"], "complete")
            self.assertAlmostEqual(report["overall"]["answer_macro_f1"], 5 / 6)
            self.assertAlmostEqual(report["overall"]["exact_match_accuracy"], 5 / 6)
            self.assertEqual(report["attempt_count"], 6)
            self.assertEqual(report["overall"]["execution_success_rate"], 1)
            # A misses in run 2 only: exact every time for B, at least once for both.
            self.assertEqual(report["stability"]["exact_in_every_run"], 1)
            self.assertEqual(report["stability"]["exact_in_at_least_one_run"], 2)
            self.assertEqual(report["per_run_overall"]["2"]["answer_macro_f1"], .5)
            self.assertEqual(report["run_spread"]["answer_macro_f1"]["min"], .5)
            self.assertEqual(report["run_spread"]["answer_macro_f1"]["max"], 1)
            self.assertEqual(report["questions"][0]["run_range"]["answer_macro_f1"], [0, 1])
            previous = copy.deepcopy(report)
            previous["configuration"]["alignments_enabled"] = False
            delta = comparison(report, previous)["difference_current_minus_previous"]
            self.assertTrue(all(value == 0 for value in delta.values()))
            paired = comparison(report, previous)["paired_test"]["answer_macro_f1"]
            self.assertEqual(paired["p_value"], 1)
            previous["questions"][0]["reference_query_sha256"] = "different"
            with self.assertRaises(ValueError):
                comparison(report, previous)

    def test_missing_and_stale_question_metrics_withhold_overall_scores(self):
        with temporary_results() as root:
            folder = self.make_question(root, "A", 1, [1, 1, 1])
            (root / "B").mkdir()
            (root / "B" / "question.txt").write_text("Missing results", encoding="utf-8")
            report = summarize(root, "luna", True, [1, 2, 3])
            self.assertIsNone(report["overall"])
            self.assertEqual(report["question_count"], 2)
            self.assertEqual(report["errors"][0]["folder"], "B")
            (folder / "run_1.rq").write_text("edited", encoding="utf-8")
            report = summarize(root, "luna", True, [1, 2, 3])
            self.assertEqual(len(report["errors"]), 2)

    def test_wilcoxon_exact_p_values(self):
        # Five gains of distinct size: only all-plus and all-minus are as extreme.
        self.assertEqual(wilcoxon_signed_rank([.1, .2, .3, .4, .5])["p_value"], 2 / 32)
        # Tied sizes share a rank; the all-plus pattern is still 1 in 32.
        self.assertEqual(wilcoxon_signed_rank([.2] * 5)["p_value"], 2 / 32)
        self.assertEqual(wilcoxon_signed_rank([0, 0])["p_value"], 1)
        differences = [.5, -.1, .3, .25, -.05, .4, .2, -.35]
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            self.skipTest("scipy is not installed")
        self.assertAlmostEqual(wilcoxon_signed_rank(differences)["p_value"],
                               wilcoxon(differences, method="exact").pvalue)


if __name__ == "__main__":
    unittest.main()
