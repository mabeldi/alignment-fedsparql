Run these commands from the repository root with the FastAPI server running and
ontology loading complete.

## Execute and score every question

Run the reference query, every available generated query, and metric calculation
for each question folder sequentially:

```powershell
py app/agents/evaluation/run_results_pipeline.py
```

Only question folders directly under `results/` are processed. The `all_models/`
archive is always skipped. One question completes all three stages before the
next question starts. A failed stage is recorded and reported, while the batch
continues with the remaining questions. By default, metric calculation scores
runs 1, 2, and 3, matching `calculate_metrics.py`.

Use `--list` to preview the order. The script also accepts `--results-dir PATH`,
`--coordinator-url URL`, `--timeout 120`, and repeated `--run N` arguments.

## One question at a time

Every script below also works on a single question. `--question` takes the
folder name under `results/`; `--query` takes a query file name (same as `--run N`).

```powershell
# Run the reference query of one question
py app/agents/evaluation/run_reference_queries.py --question List_all_the_papers

# Run one generated query of that question (repeat --query for more)
py app/agents/evaluation/run_generated_queries.py --question List_all_the_papers --query run_1.rq

# Score that question; without --query it scores runs 1, 2, 3
py app/agents/evaluation/calculate_metrics.py --question List_all_the_papers --query run_1.rq
```

`metrics.json` is rewritten with only the runs you chose, so score all three
runs again before running `calculate_overall_metrics.py`.

## Reference-query results

Execute every `reference.rq` against Fuseki before evaluating generated queries:

```powershell
.\venv\Scripts\python.exe app/agents/evaluation/run_reference_queries.py
```

Each query is sent to its first `SERVICE` endpoint, which acts as the federation
coordinator. Use `--coordinator-url URL` to send every query through a specific
endpoint instead. The runner continues after failures and exits with status 1 if
any query failed.

It writes `reference_result.json` in the same question folder. The JSON contains
the execution status, coordinator endpoint, query SHA-256, projected variables,
`result_count`, and deterministically sorted SPARQL JSON bindings. URI, blank-node,
literal, datatype, and language metadata are retained for later instance-level
comparison. Failed executions are also recorded, with `result_count: null` and an
error object, so a connection failure cannot be mistaken for an empty result set.
Connection failures (refused or reset, HTTP 502/503/504) get status `unreachable`:
the query was never judged, so it is not scored until it is executed again.

Options: `--results-dir PATH`, `--coordinator-url URL`, and `--timeout 120`.

## Metrics from saved executions

Produce the overall summary for the current Luna + alignment experiment:

```powershell
py app/agents/evaluation/calculate_overall_metrics.py --model openai/gpt-5.6-luna --alignment on
```

This writes `results/overall-metrics.json`, including configuration labels, the
question/run counts, five overall metrics, and question-level scores with each
run's value and min-max. `per_run_overall` averages each run over all questions;
`run_spread` gives their min, max and sample standard deviation, and `stability`
counts questions answered exactly in every run versus in at least one. It checks
saved question metrics against current query/result artifacts before averaging.
Questions have equal weight, regardless of answer size. Missing/inconsistent
metrics withhold the overall scores. Model/alignment labels are your declarations
of the experiment that produced the files, not proof of historical server settings.

Archive the whole results directory before clearing generated runs. Preserve
`reference_result.json` as well as `reference.rq` and `question.txt` so both
conditions use exactly the same reference snapshot. After collecting the other
condition and calculating its question metrics, run:

```powershell
py app/agents/evaluation/calculate_overall_metrics.py --model openai/gpt-5.6-luna --alignment off --compare-with PATH/TO/ARCHIVE/overall-metrics.json
```

The optional comparison checks matching questions, runs, model, policies, and
reference snapshots, then records the score differences as current minus archived,
with an exact two-sided Wilcoxon signed-rank test per metric over the per-question
scores (`paired_test`). Treat a difference smaller than the run spread as noise.
Both commands also accept `--results-dir PATH`, `--reference-dir PATH`, and repeated
`--run N` (default runs 1, 2, 3).

```powershell
python app/agents/evaluation/calculate_metrics.py
```

Writes `metrics.json` inside each question folder, using runs 1, 2, and 3 by
default. Use repeated `--run N` to choose a different fixed set. No queries or
model calls are executed. Each file includes per-run answer precision, recall,
F1 and exact match, endpoint-set precision/recall/F1, and execution success.
The aggregate contains the mean answer F1, exact-match accuracy, source-selection
F1, and execution-success rate for that question. Average the question means to
obtain the overall macro metrics; all scores are fractions from 0 to 1.

Answer comparison uses distinct complete value tuples. One global assignment of
generated columns to reference columns is chosen to maximize tuple F1, so names
and column order do not affect scores. Extra columns are ignored; resulting
duplicate rows are removed. The chosen mapping is recorded. Missing required
columns receive zero; separately projected UNION-branch columns are not coalesced.
This is strict tuple scoring, not SPINACH's partial row scoring. Only the saved
`value` string is compared: RDF type, datatype, and language tags are ignored.
A plain date string matches a typed date with identical text. Values are not
parsed, case-folded, or timezone-normalized. Each source keeps its own copy of an
instance under the same id (`http://cmt-instances#conference-1` and
`http://edas-instances#conference-1` are one conference), so answer scores merge
copies by that id: a correct answer counts whichever source's copy it returns.
`answer_raw_uri` repeats the scoring without the merge; it still drops when one
source branch of a query returns nothing.
Unbound fields remain distinct from empty strings. Blank-node labels are compared
as saved strings too; this does not establish identity across executions.

Source selection compares distinct parsed SERVICE IRIs against the reference
query, excluding comments and the coordinator URL. This measures agreement with
the reference plan, not whether an alternative source plan is semantically valid.
For successfully executed queries, both empty sets score 1; one empty set scores 0.
Execution errors always score answer F1 and exact match 0, even with an empty
reference. A generated query that neither parses nor executes is the model's
failure and scores source F1 0 as well; one that executed but cannot be parsed
here is flagged instead. Unreachable executions are never scored. Missing files, failed reference executions, unsupported comparisons,
and changed query hashes produce `needs_attention` and withhold the aggregate.
This prevents unexecuted or stale attempts from being silently dropped.

For an ablation directory with references stored centrally:

```powershell
python app/agents/evaluation/calculate_metrics.py --results-dir app/agents/evaluation/results/without_alignment --reference-dir app/agents/evaluation/results
```

Rerunning replaces `metrics.json`; the queries and saved execution files are read
only. Offline verification: `python -m unittest discover -s app/agents/evaluation`.

## Generating queries

Run every existing question folder sequentially, three times each:

```powershell
py app/agents/evaluation/run_all_questions.py
```

The batch reads `results/*/question.txt` in alphabetical folder order. It finishes
all three requests for one question before starting the next, always with one
request at a time. Responses are saved in those same folders. References and
question files remain intact. Existing runs cause numbering to append; with no
existing runs, numbers start at 1. Errors remain archived and later attempts
continue. A request that times out may still be processing on the server.

Preview order without calling the API: `py app/agents/evaluation/run_all_questions.py --list`.
Options: `--results-dir PATH`, `--runs 3`, `--url http://127.0.0.1:8999/query`,
and `--timeout 600`. Configure the model and alignment switch and restart FastAPI
before launching a new experiment; this script uses the running server settings.
After generation, execute the queries, calculate question metrics, and then the
overall summary using the scripts described here.

One question, five concurrent repetitions:

```powershell
.\venv\Scripts\python.exe app/agents/evaluation/run_evaluation.py --question "List all the papers."
```

For your 15 questions, create a UTF-8 `questions.txt` containing one question per
line, then run:

```powershell
.\venv\Scripts\python.exe app/agents/evaluation/run_evaluation.py --questions-file questions.txt
```

A JSON list of strings, or the `{"questions": [{"question": "..."}]}` format
used in `eval/questions/gold_questions.json`, is also accepted. Every question in
the supplied file is run; use a file containing only your selected questions.
Blank text lines and duplicate question texts are ignored.

Each question finishes its five repetitions before the next question starts.
Results are saved in this script's `results/` directory, regardless of the working directory:

```text
app/agents/evaluation/results/List_all_the_papers/
    question.txt
    run_1.rq
    run_1.json
    ...
    run_5.rq
    run_5.json
```

Each `.rq` contains the query returned by its own HTTP request. Each `.json`
contains the full API response, status, timestamps, and elapsed request time.
Failures produce an empty `.rq` and an explanatory `.json`; they are not retried
automatically or discarded. An interrupted attempt may remain `pending` or
`running`. A subsequent invocation appends new run numbers, preserving previous
attempts. Different questions whose sanitized names collide receive a hash suffix.

Execute the generated SPARQL and save one result file per attempt:

```powershell
.\venv\Scripts\python.exe app/agents/evaluation/run_generated_queries.py --run 1 --run 2 --run 3
```

This writes `run_1_result.json`, `run_2_result.json`, and so on beside the query
and generation metadata. The result format matches `reference_result.json`:
status, query hash, coordinator endpoint, variables, `result_count`, and sorted
SPARQL JSON bindings. Failed or empty generated queries receive an error result
instead of being omitted. Without `--run`, every available `run_N.rq` is executed.

Execution options: `--results-dir PATH`, repeated `--run N`,
`--coordinator-url URL`, and `--timeout 120`.

Options: `--runs 5`, `--parallel 5`, `--url http://127.0.0.1:8999/query`,
`--timeout 600`, and `--output-dir PATH`.

The current server shares token counters and debug dump files between requests.
For token measurements, use `--parallel 1` and avoid other simultaneous requests.
For parallel runs, use the archived responses and queries; shared server debug
files and token totals cannot be attributed reliably to individual attempts.
Parallel elapsed times also include contention. A timed-out server request may
continue processing after the script records the timeout.

This script runs generation only: it does not execute SPARQL against Fuseki or
score correctness. Freeze the model configuration, mappings, and instance data
before collecting evaluation runs. Keep failed attempts in the evaluation.

To enable or disable schema alignments for the entire pipeline, edit
`app/config/pipeline.yaml`:

```yaml
alignments_enabled: false
```

Use `true` for the full approach and `false` for the ablation. Restart FastAPI
after each change and wait for ontology loading to finish before starting runs.
The setting is fixed for the server lifetime. Each API response records
`alignments_enabled`; the evaluation script preserves it in `run_N.json` under
`response`. Existing saved runs are unchanged.

Disabled mode supplies an empty schema-alignment store throughout the pipeline:
no mapping-based candidate expansion, completeness aliases, provider connections,
source additions, variable equivalences, answer-source filtering, or mapping
patterns in LLM prompts. Ontology schemas, embeddings, model settings, validation,
instance data, and stored `owl:sameAs` links remain the same. The LLM may still
infer correspondences from the schema; the ablation removes supplied mappings.

Keep the two configurations in separate result directories:

```powershell
# After enabling alignments and restarting the server:
.\venv\Scripts\python.exe app/agents/evaluation/run_evaluation.py --questions-file questions.txt --runs 3 --parallel 1 --output-dir app/agents/evaluation/results/with_alignment

# After disabling alignments and restarting the server:
.\venv\Scripts\python.exe app/agents/evaluation/run_evaluation.py --questions-file questions.txt --runs 3 --parallel 1 --output-dir app/agents/evaluation/results/without_alignment
```
