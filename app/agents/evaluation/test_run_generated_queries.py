"""Offline tests for run_generated_queries.py."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import run_generated_queries as runner


@contextlib.contextmanager
def temporary_results():
    parent = Path(__file__).resolve().parent
    root = parent / ("test_generated_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent or not root.name.startswith("test_generated_"):
            raise RuntimeError("Unexpected test cleanup path")
        shutil.rmtree(root)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "head": {"vars": ["item"]},
            "results": {"bindings": [{"item": {"type": "uri", "value": "urn:item"}}]},
        }


def add_run(root, folder_name, run_number):
    folder = root / folder_name
    folder.mkdir(exist_ok=True)
    path = folder / f"run_{run_number}.rq"
    path.write_text(
        "SELECT ?item WHERE { "
        "SERVICE <http://192.168.100.45:3030/cmt/sparql> { ?item ?p ?o } }\n",
        encoding="utf-8",
    )
    return path


class GeneratedRunnerTests(unittest.TestCase):
    def test_numbered_results_are_written_without_replacing_generation_metadata(self):
        with temporary_results() as root:
            query = add_run(root, "Question", 2)
            generation = query.with_suffix(".json")
            generation.write_text('{"status":"generated"}\n', encoding="utf-8")
            with patch("run_reference_queries.requests.post", return_value=FakeResponse()):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--results-dir", str(root)]), 0)

            result = json.loads(
                (query.parent / "run_2_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["query_kind"], "generated")
            self.assertEqual(result["run"], 2)
            self.assertEqual(result["result_count"], 1)
            self.assertEqual(result["generation_metadata_file"], "run_2.json")
            self.assertEqual(json.loads(generation.read_text())["status"], "generated")

    def test_run_filter_selects_only_requested_numbers(self):
        with temporary_results() as root:
            add_run(root, "Question", 1)
            add_run(root, "Question", 2)
            add_run(root, "Question", 3)
            with patch("run_reference_queries.requests.post", return_value=FakeResponse()):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        runner.main(["--results-dir", str(root), "--run", "1", "--run", "3"]),
                        0,
                    )
            self.assertTrue((root / "Question" / "run_1_result.json").exists())
            self.assertFalse((root / "Question" / "run_2_result.json").exists())
            self.assertTrue((root / "Question" / "run_3_result.json").exists())

    def test_question_and_query_select_one_file(self):
        with temporary_results() as root:
            add_run(root, "A", 1)
            add_run(root, "A", 2)
            add_run(root, "B", 2)
            with patch("run_reference_queries.requests.post", return_value=FakeResponse()):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        runner.main(["--results-dir", str(root),
                                     "--question", "A", "--query", "run_2.rq"]),
                        0,
                    )
            self.assertTrue((root / "A" / "run_2_result.json").exists())
            self.assertFalse((root / "A" / "run_1_result.json").exists())
            self.assertFalse((root / "B" / "run_2_result.json").exists())


if __name__ == "__main__":
    unittest.main()
