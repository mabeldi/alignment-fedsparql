# alignment-fedsparql

Ask a question in plain English and get back a SPARQL query that runs across several
separate data sources at once (a *federated* query).

The pipeline is a chain of LLM agents. They find which terms in which source match the
question, pick the sources that hold the answer, connect them, and write the final query.
It can use an alignment file (`app/resources/mapping.rdf`) that says which terms in
different sources mean the same thing. The experiment compares the pipeline **with** and
**without** that alignment.

The test setup uses five conference-domain sources: `cmt`, `confof`, `ekaw`, `conference`
and `edas`. Each one has its own SPARQL endpoint (its own query address).

---

## What you need

| Thing | Why | Default address |
|---|---|---|
| Python 3.12 | runs the pipeline and the scripts | – |
| Docker | runs the three services below | – |
| **Apache Jena Fuseki** (we used 6.1.0) | holds the data, one dataset per source, and runs the queries | `http://localhost:3030` |
| **Qdrant** | vector database: finds the terms closest in meaning to the question's words | `http://localhost:6333` |
| **Ollama** + model `qwen3-embedding:4b` | turns text into vectors (lists of numbers that capture meaning) for Qdrant | `http://localhost:11434/v1` |
| An LLM API key | the agents call an LLM. The experiment used `meta/muse-spark-1.3` through the Vercel AI Gateway | – |
| Internet access | at startup the server downloads the source schemas from the OAEI website | – |

---

## Repository layout

```text
app/
  main.py                    the server: runs the whole agent pipeline for one question
  agents/                    the agents (agent1 ... agent8) and their helpers
    skills/sparql.md         instructions given to the query-writing agent
    evaluation/              scripts that run the experiment and score it
      results/               experiment questions, generated queries and scores
  config/
    agent_models.yaml        which LLM each agent uses
    pipeline.yaml            alignment on / off switch
  tools/                     LLM clients, path search, output writers
  core/                      loads the source schemas
  embeddings/
    index_ontology.py        fills Qdrant
  resources/
    mapping.rdf              the alignment between sources
    dataset/populated_datasets/   the data loaded into Fuseki
  tests/
eval/
  setup_fuseki.py            creates the five Fuseki datasets
  load_dataset.py            loads the data into them
  questions/                 gold questions and hand-written gold queries
  fixtures/                  extra test data added on top of the main data
  *.py                       scripts that built and checked the benchmark
requirements.txt
BENCHMARK.md                 how the benchmark was built (not needed to run it)
```

---

## Setup

Commands are for Windows PowerShell. On Linux/macOS use `source venv/bin/activate`
instead of `.\venv\Scripts\Activate.ps1`, and `export NAME=value` instead of
`$env:NAME = "value"`.

### 1. Python environment

```powershell
cd alignment-fedsparql
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Settings and keys

```powershell
Copy-Item app\.env.example app\.env
```

Open `app/.env` and fill in:

- the API key for the LLM provider named in `app/config/agent_models.yaml`. For the
  setup we used, that is `VERCEL_AI_API_KEY`.
- `QDRANT_URL`, `OLLAMA_BASE_URL` and `FUSEKI_BASE_URL`, if your services are not on
  the default addresses.

`app/.env` is in `.gitignore`. Never commit it.

### 3. Start the services in Docker

**Qdrant**

```powershell
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

**Ollama** and the embedding model

```powershell
docker run -d --name ollama -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
docker exec ollama ollama pull qwen3-embedding:4b
```

**Fuseki**: run the **full** Apache Jena Fuseki 6.1.0 server in Docker on port 3030,
with a volume so the data survives a restart. We used a `fuseki-full` image. "Full"
means the server with the web UI and the admin API. The minimal image built from
Apache's own Dockerfile
(<https://jena.apache.org/documentation/fuseki2/fuseki-docker.html>) has **no** admin
API, so `eval/setup_fuseki.py` can't create the datasets with it.

Two things must be true, or the setup fails:

1. **The admin API (`/$/...`) accepts requests without a login** from the machine that
   runs the setup scripts. Fuseki's default `shiro.ini` allows this only from
   `localhost`. If the scripts get `HTTP 403`, edit `shiro.ini`.
2. **Fuseki can call its own endpoints through `SERVICE`.** Every generated query
   sends parts of itself to the other endpoints. `setup_fuseki.py` checks this for you.

### 4. Load the data into Fuseki

```powershell
python eval/setup_fuseki.py                    # creates the 5 datasets, then checks SERVICE calls work
python eval/load_dataset.py --level 100        # the level used in our experiment
python eval/load_dataset.py --level 100 --check   # only checks what is loaded, changes nothing
```

`--level` picks how much data the sources share: 0, 20, 40, 60, 80 or 100 (the
script's default is 60). Each level is a folder
`app/resources/dataset/populated_datasets/data_<level>/`. Loading replaces what is in
the datasets. Add `--keep` to add to it instead.

If `setup_fuseki.py` prints `BLOCKED`, `SERVICE` calls don't work. Fix that before
going further, otherwise every query returns zero rows.

### 5. Add the extra test data

At level 100 every source holds a copy of the same data, so a query that reads only
one source could still get every answer right. The extra data fixes that: it adds
items that exist in **only one** source, so a query that skips a needed source loses
answers. Our experiment ran with this data loaded.

```powershell
$fuseki = "http://localhost:3030"      # your FUSEKI_BASE_URL
$fx = "eval\fixtures"

# 500 accepted papers per source, each one in a single source only
foreach ($s in "cmt", "conference", "edas", "ekaw") {
    Invoke-RestMethod -Method Post -Uri "$fuseki/$s/data" -ContentType "text/turtle; charset=utf-8" `
        -InFile "$fx\accepted_papers_unique_v1_500\$s.ttl"
}

# single-source items for the start-date, dinner, early-registration and topic questions
foreach ($s in "cmt", "conference", "confof", "edas", "ekaw") {
    Invoke-RestMethod -Method Post -Uri "$fuseki/$s/data" -ContentType "text/turtle; charset=utf-8" `
        -InFile "$fx\source_coverage_v1\$s.ttl"
}

# the "locations" case in that set is not one of the 15 final questions: take it out again
foreach ($s in "conference", "edas") {
    Invoke-RestMethod -Method Post -Uri "$fuseki/$s/update" -Body @{ update =
        'DELETE { ?x ?p ?o } WHERE { ?x ?p ?o FILTER(CONTAINS(STR(?x), "conference-location-")) }' }
}
```

To check it worked, count the extra items in each source:

```powershell
foreach ($s in "cmt", "conference", "confof", "edas", "ekaw") {
    $q = 'SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x ?p ?o FILTER(CONTAINS(STR(?x), "-aug")) }'
    $r = Invoke-RestMethod -Uri "$fuseki/$s/sparql?query=$([uri]::EscapeDataString($q))" `
        -Headers @{ Accept = "application/sparql-results+json" }
    "$s : $($r.results.bindings[0].n.value)"
}
```

Expected: cmt 2300, conference 1740, confof 1240, edas 1340, ekaw 1740.

The scripts that created this data are described in [BENCHMARK.md](BENCHMARK.md).

### 6. Fill Qdrant

```powershell
python app/embeddings/index_ontology.py              # first time
python app/embeddings/index_ontology.py --recreate   # wipe and rebuild
```

This reads the five source schemas, turns every term into a vector with Ollama, and
stores the vectors in the Qdrant collection `ontology_terms`.

---

## Start the server

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\app"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8999 --loop asyncio
```

The server loads the source schemas and builds its index in the background. **Wait
until the log says loading is finished** before sending questions.

Try one question:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8999/query `
  -ContentType "application/json" -Body '{"question": "Which papers were withdrawn?"}'
```

> The server reads `app/config/*.yaml` and `app/.env` only at startup. **Restart it**
> after changing a model, the alignment switch, or a setting.

---

## Run the experiment

The experiment asks 15 questions, 3 times each, once with the alignment on and once
with it off. Answers vary from run to run, so always use the 3 repeats.

### 1. Put the questions in place

The scripts work on question folders directly under `app/agents/evaluation/results/`.
Each folder needs a `question.txt` and the gold query `reference.rq`. Copy them out of
the stored experiment:

```powershell
$archive = "app\agents\evaluation\results\all_models\muse_with_alignment"
$results = "app\agents\evaluation\results"
$fuseki  = "http://localhost:3030"   # your FUSEKI_BASE_URL

Get-ChildItem $archive -Directory | ForEach-Object {
    $dest = Join-Path $results $_.Name
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $_.FullName "question.txt") $dest
    # The stored gold queries name our Fuseki address. Swap in yours.
    $rq = (Get-Content (Join-Path $_.FullName "reference.rq") -Raw -Encoding UTF8) `
          -replace "http://192\.168\.100\.45:3030", $fuseki
    [IO.File]::WriteAllText((Join-Path $dest "reference.rq"), $rq)
}
```

Why swap the address: the gold queries name each source by its endpoint address, and
scoring compares those addresses with the ones in the generated queries. If they don't
match, the gold queries can't run and the source scores are wrong.

Preview which questions will run:

```powershell
python app/agents/evaluation/run_all_questions.py --list
```

### 2. Choose the condition

Set the switch in `app/config/pipeline.yaml`, then (re)start the server:

```yaml
alignments_enabled: true    # with alignment
# alignments_enabled: false # without alignment
```

### 3. Generate the queries

```powershell
python app/agents/evaluation/run_all_questions.py          # 3 runs per question (default)
```

It asks the server each question, one request at a time. Running it again adds new
runs instead of replacing the old ones.

### 4. Run the queries and score them

```powershell
python app/agents/evaluation/run_results_pipeline.py
```

For each question, this runs the gold query, then every generated query, then scores
them.

### 5. Overall scores

```powershell
python app/agents/evaluation/calculate_overall_metrics.py --model muse-spark-1.3 --alignment on
```

Use `--alignment off` for the other condition. `--model` and `--alignment` are only
labels written into the file. They don't change any settings.

### 6. Save this condition, then run the other one

Move the question folders and `overall-metrics.json` into a new folder, for example
`results/all_models/my_run_with_alignment/`. Then repeat steps 1–5 with the switch
flipped. To compare the two conditions (this includes a statistical test, the Wilcoxon
signed-rank test, on the per-question scores):

```powershell
python app/agents/evaluation/calculate_overall_metrics.py --model muse-spark-1.3 --alignment off `
  --compare-with app\agents\evaluation\results\all_models\my_run_with_alignment\overall-metrics.json
```

Every option of every script is listed in [Script reference](#script-reference) below.

---

## Test one question

To try the pipeline on a single question without the full experiment:

```powershell
python app/agents/evaluation/run_evaluation.py --question "Which papers were withdrawn?" --runs 1 --parallel 1
```

The result goes to `app/agents/evaluation/results/Which_papers_were_withdrawn/`
(`run_1.rq` and `run_1.json`). To run that query and score it too, the folder also
needs a `reference.rq` (experiment step 1). Then:

```powershell
python app/agents/evaluation/run_reference_queries.py --question Which_papers_were_withdrawn
python app/agents/evaluation/run_generated_queries.py --question Which_papers_were_withdrawn --query run_1.rq
python app/agents/evaluation/calculate_metrics.py     --question Which_papers_were_withdrawn --query run_1.rq
```

`--question` takes the **folder name**, not the question text.

---

## Script reference

All scripts are run from the repository root. Defaults are shown in brackets.
`results/` means `app/agents/evaluation/results/`.

### Experiment scripts (`app/agents/evaluation/`)

**`run_evaluation.py`**: sends questions to the server and saves each answer. Good for
testing one or a few questions.

| Option | Meaning |
|---|---|
| `--question "TEXT"` | question to ask. Can be repeated |
| `--questions-file FILE` | a text file with one question per line, or a JSON list |
| `--runs N` | how many times to ask each question [5] |
| `--parallel N` | how many requests at the same time, per question [5] |
| `--url URL` | server address [`http://127.0.0.1:8999/query`] |
| `--timeout SEC` | how long to wait for one answer [600] |
| `--output-dir DIR` | where to save [`results/`] |

**`run_all_questions.py`**: asks every question that has a folder in `results/`, one
request at a time.

| Option | Meaning |
|---|---|
| `--runs N` | runs per question [3] |
| `--list` | only show the order, ask nothing |
| `--url URL` | server address [`http://127.0.0.1:8999/query`] |
| `--timeout SEC` | [600] |
| `--results-dir DIR` | [`results/`] |

**`run_reference_queries.py`**: runs the gold query `reference.rq` of each question and
saves the rows to `reference_result.json`.

| Option | Meaning |
|---|---|
| `--question FOLDER` | only this question folder |
| `--coordinator-url URL` | send every query to this endpoint [the first `SERVICE` endpoint in the query] |
| `--timeout SEC` | per query [120] |
| `--results-dir DIR` | [`results/`] |

**`run_generated_queries.py`**: runs the generated queries `run_N.rq` and saves the
rows to `run_N_result.json`.

| Option | Meaning |
|---|---|
| `--question FOLDER` | only this question folder |
| `--run N` | only this run number. Can be repeated [all runs] |
| `--query FILE` | only this file, e.g. `run_1.rq` (same as `--run 1`) |
| `--coordinator-url URL` | as above |
| `--timeout SEC` | per query [120] |
| `--results-dir DIR` | [`results/`] |

**`calculate_metrics.py`**: scores each question from the saved rows and writes
`metrics.json`. Calls no LLM and runs no queries.

| Option | Meaning |
|---|---|
| `--question FOLDER` | only this question folder |
| `--run N` | runs to score. Can be repeated [1, 2, 3] |
| `--query FILE` | only this file (same as `--run N`) |
| `--reference-dir DIR` | take `reference.rq` / `reference_result.json` from here instead |
| `--results-dir DIR` | [`results/`] |

**`run_results_pipeline.py`**: the three scripts above in one go, question by question:
gold query, then generated queries, then scores. Skips `results/all_models/`.

| Option | Meaning |
|---|---|
| `--run N` | runs to run and score. Can be repeated [runs all, scores 1, 2, 3] |
| `--list` | only show the order |
| `--coordinator-url URL` | as above |
| `--timeout SEC` | per query [120] |
| `--results-dir DIR` | [`results/`] |

**`calculate_overall_metrics.py`**: averages all questions into `overall-metrics.json`
and can compare two conditions.

| Option | Meaning |
|---|---|
| `--model NAME` | **required**. Label for the model used |
| `--alignment on\|off` | **required**. Label for the condition |
| `--compare-with FILE` | `overall-metrics.json` of the other condition, to compare with |
| `--run N` | runs to include. Can be repeated [1, 2, 3] |
| `--reference-dir DIR` | as above |
| `--results-dir DIR` | [`results/`] |

### Setup scripts

| Script | Options |
|---|---|
| `eval/setup_fuseki.py` | none. Uses `FUSEKI_BASE_URL` |
| `eval/load_dataset.py` | `--level 0\|20\|40\|60\|80\|100` [60], `--keep` (add instead of replace), `--check` (only check) |
| `app/embeddings/index_ontology.py` | `--recreate` (wipe the Qdrant collection first) |

### Scripts used to build the benchmark

They are described in [BENCHMARK.md](BENCHMARK.md). You don't need them to run the
experiment.

---

## Where the output goes

**Experiment results**: `app/agents/evaluation/results/<question folder>/`

| File | What it is |
|---|---|
| `question.txt` | the question |
| `reference.rq` | the hand-written gold query |
| `reference_result.json` | rows returned by the gold query |
| `run_N.rq` | the query the pipeline generated in run N |
| `run_N.json` | the full server response for run N |
| `run_N_result.json` | rows returned by the generated query |
| `metrics.json` | scores for this question, per run |

`app/agents/evaluation/results/overall-metrics.json` holds the scores over all
questions: answer F1, exact match, source-selection F1, execution success, the spread
between runs, and (with `--compare-with`) the difference between the two conditions.

**Debug output for the last question asked**: `app/agents/output/`. This holds the
prompt each agent received, the paths found per source (`paths_agent5_*`,
`paths_agent6_*`) and the alignment used (`alignment.txt`). It is overwritten by every
request. The server log in the terminal shows each agent's step, timing and token use.

---

## Our results

Stored in `app/agents/evaluation/results/all_models/`: 15 questions × 3 runs per
condition, model `muse-spark-1.3`.

| Metric (0 to 1) | Without alignment | With alignment |
|---|---|---|
| Answer F1 | 0.767 | 1.000 |
| Exact match | 0.489 | 1.000 |
| Source-selection F1 | 0.764 | 1.000 |
| Queries that ran without error | 0.911 | 1.000 |

Full details per question and per run are in each condition's `overall-metrics.json`.

---

## Tests

They need no LLM and no running services. Set `PYTHONIOENCODING=utf-8` first,
otherwise the Windows console fails to print some log characters.

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m unittest discover -s app/agents/evaluation -p "test_*.py"
cd app; python -m unittest tests/test_alignment_mode.py; cd ..
cd eval; python -m unittest discover -p "test_*.py"; cd ..
```

`eval/test_seed_source_coverage.py` fails: it reads the folder of a question that was
dropped from the benchmark and is not in this repo (see [BENCHMARK.md](BENCHMARK.md)).
All other tests pass.

---

## Common problems

| Symptom | Likely cause |
|---|---|
| Every query returns 0 rows | `SERVICE` calls blocked in Fuseki, or data not loaded (`python eval/load_dataset.py --check`) |
| `HTTP 403` from the setup scripts | Fuseki admin API is localhost-only (`shiro.ini`) |
| Gold queries show as `unreachable` | `reference.rq` still has the old Fuseki address (experiment step 1) |
| A setting change has no effect | the server reads settings only at startup, so restart it |
| A query hangs or Fuseki stops answering | Fuseki has no query time limit by default; restart the container |
| Agent 2 finds nothing | Qdrant is empty (setup step 6) or Ollama/the embedding model is not running |
