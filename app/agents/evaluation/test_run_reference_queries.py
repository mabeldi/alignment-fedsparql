"""Offline tests for run_reference_queries.py."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import requests

import run_reference_queries as runner


@contextlib.contextmanager
def temporary_results():
    parent = Path(__file__).resolve().parent
    root = parent / ("test_reference_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != parent or not root.name.startswith("test_reference_"):
            raise RuntimeError("Unexpected test cleanup path")
        shutil.rmtree(root)


def add_reference(root, name="Example"):
    folder = root / name
    folder.mkdir()
    (folder / "question.txt").write_text("Example question?\n", encoding="utf-8")
    (folder / "reference.rq").write_text(
        "SELECT DISTINCT ?item WHERE { "
        "SERVICE <http://192.168.100.45:3030/cmt/sparql> { ?item ?p ?o } }\n",
        encoding="utf-8",
    )
    return folder


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FailingResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class ReferenceRunnerTests(unittest.TestCase):
    def test_success_is_counted_and_sorted_with_rdf_term_metadata(self):
        payload = {
            "head": {"vars": ["item"]},
            "results": {"bindings": [
                {"item": {"value": "urn:b", "type": "uri"}},
                {"item": {"type": "literal", "xml:lang": "en", "value": "A"}},
            ]},
        }
        with temporary_results() as root:
            folder = add_reference(root)
            with patch.object(runner.requests, "post", return_value=FakeResponse(payload)) as post:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--results-dir", str(root)]), 0)

            result = json.loads((folder / "reference_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["variables"], ["item"])
            self.assertEqual(result["result_count"], 2)
            self.assertEqual(result["bindings"][0]["item"]["value"], "A")
            self.assertEqual(result["bindings"][0]["item"]["xml:lang"], "en")
            self.assertEqual(
                result["coordinator_endpoint"],
                "http://192.168.100.45:3030/cmt/sparql",
            )
            self.assertEqual(post.call_args.kwargs["data"]["query"].split()[0], "SELECT")

    def test_failure_is_recorded_and_other_queries_continue(self):
        with temporary_results() as root:
            first = add_reference(root, "First")
            second = add_reference(root, "Second")
            success = FakeResponse({"head": {"vars": ["item"]}, "results": {"bindings": []}})
            with patch.object(
                runner.requests,
                "post",
                side_effect=[requests.ConnectionError("offline"), success],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--results-dir", str(root)]), 1)

            failed = json.loads((first / "reference_result.json").read_text(encoding="utf-8"))
            passed = json.loads((second / "reference_result.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "unreachable")
            self.assertIsNone(failed["result_count"])
            self.assertEqual(failed["error"]["type"], "ConnectionError")
            self.assertEqual(passed["status"], "success")
            self.assertEqual(passed["result_count"], 0)

    def test_server_down_is_unreachable_but_a_rejected_query_is_an_error(self):
        with temporary_results() as root:
            folders = [add_reference(root, name) for name in ("A", "B", "C")]
            failures = [FailingResponse(400), FailingResponse(503),
                        requests.exceptions.ChunkedEncodingError("connection cut")]
            with patch.object(runner.requests, "post", side_effect=failures):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--results-dir", str(root)]), 1)
            statuses = [json.loads((folder / "reference_result.json").read_text(encoding="utf-8"))["status"]
                        for folder in folders]
            self.assertEqual(statuses, ["error", "unreachable", "unreachable"])

    def test_question_option_runs_only_that_folder(self):
        with temporary_results() as root:
            first = add_reference(root, "First")
            second = add_reference(root, "Second")
            empty = FakeResponse({"head": {"vars": ["item"]}, "results": {"bindings": []}})
            with patch.object(runner.requests, "post", return_value=empty):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        runner.main(["--results-dir", str(root), "--question", "Second"]), 0)
            self.assertFalse((first / "reference_result.json").exists())
            self.assertTrue((second / "reference_result.json").exists())


if __name__ == "__main__":
    unittest.main()
