# How the benchmark was built

You **don't need** anything here to run the experiment. The main [README](README.md)
covers that. This page explains the files and scripts we used to build the questions,
the gold queries, the alignment file and the extra test data.

---

## Which gold queries are final

The experiment is scored against the **15 `reference.rq` files** in
`app/agents/evaluation/results/all_models/<condition>/<question folder>/`. Both
conditions use the same 15.

`eval/questions/` holds the larger, earlier question set the benchmark started from.
It is kept to show how the benchmark was made, but its files are **not** what the
experiment is scored against.

| Path | What it is |
|---|---|
| `eval/questions/gold_questions.json` | 42 questions, each with a gold query (built by `build_gold_questions.py`) |
| `eval/questions/gold_rq/` | 24 gold queries written by hand, one `.rq` per question id |
| `eval/questions/CQAs/` | the published CQA files (100 folders). CQAs ("competency questions for alignment", Thiéblin et al., IRIT) are questions written once for each source that can answer them |
| `eval/questions/CQAs-validation/` | one federated query per CQA folder (built by `build_cqa_validation.py`) |
| `eval/questions/benchmark_audit.md` | notes from checking each question's wording against its gold query |
| `eval/questions/answer_eval_exclusions.json` | questions left out of the answer scores because the data has nothing for them |
| `eval/questions/README.md` | details on the 42-question set |

---

## Scripts

All scripts are run from the repository root. Scripts that talk to Fuseki read
`FUSEKI_BASE_URL` from the environment unless noted.

### Building the question set

| Script | What it does | Options |
|---|---|---|
| `eval/build_gold_questions.py` | builds `gold_questions.json` from the CQA files | `--check`: build and print, write nothing |
| `eval/build_cqa_validation.py` | writes one federated query per folder of `eval/questions/CQAs/` into `CQAs-validation/` | `--check`: build, write nothing |
| `eval/migrate_gold_sameas.py` | one-time fix: makes the gold queries that join two sources link the same item across them through `owl:sameAs` | `--apply`: write the change (without it, only shows it) |

`build_gold_questions.py` reads the CQA files from `eval/cqa_source/CQAs/`, which is
not in this repository, so it **does not run as is**. A copy of the CQA files is in
`eval/questions/CQAs/`, but the script doesn't look there.

### Checking the question set

| Script | What it does | Options |
|---|---|---|
| `eval/verify_gold_questions.py` | checks every gold query in `gold_questions.json`: it is valid SPARQL, every term it uses exists in the source schemas, and it runs on Fuseki | `--offline`: skip the Fuseki check |

The schemas it downloads are cached in `eval/.cache/`. Delete that folder to download
them again.

### Completing the alignment file (`app/resources/mapping.rdf`)

| Script | What it does | Options |
|---|---|---|
| `eval/merge_reference_alignments.py` | adds the term matches that were missing for three source pairs (cmt–edas, confof–edas, edas–ekaw), taken from the official OAEI alignment. It only adds and never removes. **Already applied** to the `mapping.rdf` in this repo | `--write`: apply (without it, only reports) |
| `eval/derive_mappings_from_cqas.py` | finds term matches that the CQA files imply but `mapping.rdf` is missing, for a person to review | `--check`: write nothing |

`derive_mappings_from_cqas.py` also reads `eval/cqa_source/CQAs/`, so it **does not
run as is** either.

### Extra test data (`eval/fixtures/`)

At data level 100 every source holds a copy of the same data, so a query that reads
only one source could still get every answer right. These two scripts add items that
exist in **only one** source, so skipping a needed source costs answers. Each one has
four actions:
- `prepare` writes the data files and touches nothing else.
- `apply` adds the data to Fuseki and then checks it.
- `verify` only checks.
- `remove` deletes exactly the triples (single facts) it added.

| Script | What it adds | Options |
|---|---|---|
| `eval/seed_unique_accepted_papers.py` | accepted papers that exist in a single source (cmt, conference, edas, ekaw) | `--count N` papers per source [20]. **We used 500.** `--base-url URL` [`FUSEKI_BASE_URL`], `--output-dir DIR` |
| `eval/seed_source_coverage.py` | single-source items for five questions: `locations`, `start_dates`, `dinners`, `early_registration`, `topics` | `--case NAME`: with `remove`, delete only that question's items. `--output-dir DIR` |

What our experiment ran with:
- `seed_unique_accepted_papers.py` with `--count 500`. Its files are in
  `fixtures/accepted_papers_unique_v1_500/`, and the 20-paper set is part of them.
- `seed_source_coverage.py`, with the `locations` case removed again, because its
  question is not one of the final 15. Its files are in `fixtures/source_coverage_v1/`.

**Limits if your Fuseki is not at our address (`http://192.168.100.45:3030`):**

- `seed_source_coverage.py` has that address written in the code instead of reading
  `FUSEKI_BASE_URL`. It also checks the `reference.rq` of each of its five questions
  in `app/agents/evaluation/results/<question folder>/`. The `locations` question was
  dropped from the benchmark and its folder is not in this repo, so **the script
  stops with an error as is**.
- `seed_unique_accepted_papers.py` refuses to reuse a fixture folder made for a
  different address. Pass `--base-url` with your address and a new `--output-dir`.

You don't need either script to reproduce our data. The main README (setup step 5)
loads the same files directly into Fuseki, which works with any address.

### Tests

```powershell
$env:PYTHONIOENCODING = "utf-8"
cd eval; python -m unittest discover -p "test_*.py"; cd ..
```

`test_seed_source_coverage.py` reads the folder of the dropped `locations` question,
which is not in this repo, so it fails. The other tests pass.
