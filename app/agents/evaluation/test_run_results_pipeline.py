"""Offline tests for run_results_pipeline.py."""

import contextlib
import io
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import run_results_pipeline as pipeline


@contextlib.contextmanager
def temporary_results():
    parent = Path(__file__).resolve().parent
    root = parent / ("test_pipeline_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent or not root.name.startswith("test_pipeline_"):
            raise RuntimeError("Unexpected test cleanup path")
        shutil.rmtree(root)


def add_question(root, name):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "reference.rq").write_text("SELECT * WHERE {}", encoding="utf-8")
    (folder / "run_1.rq").write_text("SELECT * WHERE {}", encoding="utf-8")
    return folder


class ResultsPipelineTests(unittest.TestCase):
    def test_each_root_question_runs_all_stages_in_order_and_skips_all_models(self):
        calls = []

        def stage(name, code=0):
            def invoke(argv):
                calls.append((name, argv[argv.index("--question") + 1]))
                return code
            return invoke

        with temporary_results() as root:
            add_question(root, "B_question")
            add_question(root, "A_question")
            add_question(root / "all_models", "Archived_question")
            with patch.object(pipeline, "run_reference_queries", side_effect=stage("reference")), \
                    patch.object(pipeline, "run_generated_queries", side_effect=stage("generated")), \
                    patch.object(pipeline, "calculate_metrics", side_effect=stage("metrics")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(["--results-dir", str(root)]), 0)

        self.assertEqual(calls, [
            ("reference", "A_question"), ("generated", "A_question"),
            ("metrics", "A_question"), ("reference", "B_question"),
            ("generated", "B_question"), ("metrics", "B_question"),
        ])

    def test_failure_is_reported_after_later_questions_are_processed(self):
        calls = []

        def reference(argv):
            question = argv[argv.index("--question") + 1]
            calls.append(question)
            return int(question == "A_question")

        with temporary_results() as root:
            add_question(root, "A_question")
            add_question(root, "B_question")
            with patch.object(pipeline, "run_reference_queries", side_effect=reference), \
                    patch.object(pipeline, "run_generated_queries", return_value=0), \
                    patch.object(pipeline, "calculate_metrics", return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(["--results-dir", str(root)]), 1)
        self.assertEqual(calls, ["A_question", "B_question"])

    def test_default_does_not_filter_generated_runs(self):
        captured = []
        with temporary_results() as root:
            add_question(root, "Question")
            with patch.object(pipeline, "run_reference_queries", return_value=0), \
                    patch.object(pipeline, "run_generated_queries",
                                 side_effect=lambda argv: captured.append(argv) or 0), \
                    patch.object(pipeline, "calculate_metrics", return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(["--results-dir", str(root)]), 0)
        self.assertNotIn("--run", captured[0])


if __name__ == "__main__":
    unittest.main()
