"""Offline checks; no API calls or model usage."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import threading
import unittest
import uuid
from unittest.mock import patch
from urllib.error import HTTPError

import run_evaluation as runner


@contextlib.contextmanager
def temporary_output():
    # Inherit workspace permissions (Windows sandbox cannot access mode-0700
    # directories created by tempfile on some Python versions).
    root = Path(__file__).resolve().parent
    folder = root / ("test_output_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield str(folder)
    finally:
        if folder.resolve().parent != root or not folder.name.startswith("test_output_"):
            raise RuntimeError("Unexpected test cleanup path")
        shutil.rmtree(folder)


class EvaluationTests(unittest.TestCase):
    def test_parallel_responses_are_separate_and_reruns_append(self):
        barrier = threading.Barrier(5)
        lock = threading.Lock()
        calls = []

        def fake_api(request, timeout):
            question = json.loads(request.data)["question"]
            with lock:
                calls.append(question)
                index = len(calls)
            barrier.wait(timeout=10)  # Requires five requests actually in flight.
            response = io.BytesIO(json.dumps({
                "question": question, "query": f"SELECT ?x WHERE {{ ?x <urn:p> {index} }}",
            }).encode())
            response.status = 200
            return response

        with temporary_output() as temporary:
            output = Path(temporary)
            argv = ["--question", "List all the papers.", "--output-dir", str(output)]
            with patch.object(runner, "urlopen", side_effect=fake_api), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(argv), 0)
                self.assertEqual(runner.main(argv), 0)
            folder = output / "List_all_the_papers"
            self.assertEqual(len(list(folder.glob("*.rq"))), 10)
            self.assertEqual(len({p.read_text() for p in folder.glob("*.rq")}), 10)
            for number in range(1, 11):
                metadata = json.loads((folder / f"run_{number}.json").read_text())
                self.assertEqual(metadata["status"], "generated")
                self.assertEqual((folder / f"run_{number}.rq").read_text().strip(),
                                 metadata["response"]["query"])

    def test_failures_are_archived_without_retries(self):
        empty = io.BytesIO(b'{"query": ""}')
        empty.status = 200
        error = HTTPError("http://test/query", 503, "Unavailable", {}, io.BytesIO(b"busy"))
        with temporary_output() as temporary:
            with patch.object(runner, "urlopen", side_effect=[empty, error]) as api, contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(["--question", "Example?", "--runs", "2",
                                      "--parallel", "1", "--output-dir", temporary])
            self.assertEqual(result, 1)
            self.assertEqual(api.call_count, 2)
            folder = Path(temporary) / "Example"
            statuses = [json.loads((folder / f"run_{i}.json").read_text())["status"]
                        for i in (1, 2)]
            self.assertEqual(statuses, ["empty_query", "http_error"])
            self.assertTrue(all(p.read_text() == "" for p in folder.glob("*.rq")))

    def test_question_files_and_folder_collisions(self):
        with temporary_output() as temporary:
            root = Path(temporary)
            text_file = root / "questions.txt"
            text_file.write_text("First?\n\nSecond?\nFirst?\n", encoding="utf-8-sig")
            args = type("Args", (), {"question": None, "questions_file": text_file})()
            self.assertEqual(runner.load_questions(args), ["First?", "Second?"])
            json_file = root / "questions.json"
            json_file.write_text(json.dumps({"questions": [{"question": "First?"},
                                                          {"question": "Second?"}]}))
            args.questions_file = json_file
            self.assertEqual(runner.load_questions(args), ["First?", "Second?"])
            first = runner.question_folder(root, "First?")
            second = runner.question_folder(root, "First!")
            self.assertNotEqual(first, second)
            self.assertEqual(second, runner.question_folder(root, "First!"))
            self.assertEqual(runner.question_folder(root, "CON").name, "Question_CON")


if __name__ == "__main__":
    unittest.main()
