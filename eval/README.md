# Evaluation

Everything needed to measure the pipeline instead of reading its output by eye.

Until now the endpoints in `SPARQL_ENDPOINTS` were fake (`http://example.org/sparql/cmt`),
so no generated query could ever run and no answer-based metric was possible.
This folder fixes that, step by step.

---

## Plan

| Step | What | State |
|---|---|---|
| 1 | Real endpoints — one Fuseki dataset per ontology | **done** |
| 2 | Gold question set + gold queries | **done** — 42 questions, [`questions/README.md`](questions/README.md) |
| 3 | Query-comparison harness + metrics | **built, not yet run** — [`harness/README.md`](harness/README.md) |
| 4 | Ablations | not started |

**No instance data.** The endpoints stay empty and the evaluation compares the
*generated query* against a *gold query*, not answers against answers. What this
costs and what it does not is spelled out under Step 3.

---

## Step 1 — endpoints (done)

**Target:** Fuseki 6.1.0 in Docker at `http://192.168.100.44:3030`.

### What was done

1. **Checked reachability.** `GET /$/ping` answered. The admin API (`/$/*`) was
   initially localhost-only (`HTTP 403 — Access denied : only localhost access
   allowed`, Fuseki's default `shiro.ini`); this was opened up so the datasets
   could be created remotely.

2. **Created three TDB2 datasets** — `cmt`, `confof`, `ekaw` — one per ontology,
   via `POST /$/datasets?dbType=tdb2&dbName=<name>`. All returned HTTP 200.

   One dataset per ontology, not one dataset with three named graphs: each
   endpoint must hold **its own instances** for federation to mean anything. A
   single store would make `SERVICE` a formality and any coverage measurement
   meaningless.

3. **Verified the endpoints answer queries.** All three returned `0` triples —
   correct, they are empty.

4. **Verified `SERVICE` federation works.** A query sent to `/cmt/sparql`
   reaching out to `/confof/sparql` and `/ekaw/sparql` returned a row.

   This check is the reason step 1 is not just "create datasets". Every
   generated query wraps each source in its own `SERVICE` block, so if Jena
   blocked outbound `SERVICE` the entire evaluation would return empty result
   sets — which looks exactly like a pipeline failure and would cost a day to
   diagnose.

5. **Pointed the pipeline at them.** `SPARQL_ENDPOINTS` in
   `app/agents/agent7_sparql_query_generator.py` now builds from
   `FUSEKI_BASE_URL` (default `http://192.168.100.44:3030`):

   ```
   cmt     http://192.168.100.44:3030/cmt/sparql
   confof  http://192.168.100.44:3030/confof/sparql
   ekaw    http://192.168.100.44:3030/ekaw/sparql
   ```

   An env var rather than a literal: generated queries embed these URLs verbatim
   in their `SERVICE` blocks, so moving machines would otherwise mean editing
   queries already recorded as results.

### Reproducing it

```bash
python eval/setup_fuseki.py
```

Idempotent — existing datasets are left alone, so it is safe to re-run. It
creates whatever is missing, prints triple counts, and ends with the federation
check. A `BLOCKED` result there means stop and fix before going further.

Override the host with `FUSEKI_BASE_URL` if Fuseki moves.

### Note

Datasets were created through the admin API rather than a mounted Fuseki config
file, because the container's filesystem is not reachable from here. They live
in the container's TDB2 state, so **a container rebuild without a volume loses
them** — re-run the script if that happens.

---

## Step 1b — added `conference` and `edas`

Five ontologies now: `cmt`, `confof`, `ekaw`, `conference`, `edas`.

A source has to be registered in **four** places, and each omission fails
differently:

| File | Constant | Omitting it means |
|---|---|---|
| `app/main.py` | `ONTOLOGIES` | no graph — no paths can be built |
| `app/embeddings/index_ontology.py` | `ONTOLOGY_URLS` | not retrievable at all |
| `app/agents/agent7_sparql_query_generator.py` | `SPARQL_ENDPOINTS` | nameable but unreachable |
| `eval/setup_fuseki.py` | `DATASETS` | endpoint answers nothing |

Datasets `conference` and `edas` were created on Fuseki; federation re-checked
across all five: OK.

### Two bugs this exposed

**Blank nodes were being indexed as ontology terms.** OWL uses blank nodes for
anonymous class expressions (restrictions, unions, intersections) and types them
`owl:Class` like everything else. `parse_rdf_to_json` indexed them, producing
terms whose `uri` was a generated id such as `n7372e572dda54117...`. With no
namespace, `source_of` read that id **as the ontology name**, so every one became
its own source — and Agent 4 groups by source, so each would have triggered its
own LLM call.

Counts: cmt 3, confof 11, ekaw 13, conference 22, edas 16 — 65 total, 27 of them
already in the index. Now filtered to named URIs only. Domain and range are
checked too, since Agent 4 recovers Class terms directly from those fields.

Term counts after the fix, one clean source each:

```
cmt          88   confof  74   ekaw  106   conference  123   edas  153     (544)
```

**Version mismatch.** `main.py` loaded the 2022 OAEI files while the indexer
downloaded 2020 ones. Term counts are identical between the two years, so
nothing was actually broken, but the index and the graphs must come from the
same files or a retrieved term can be absent from the graph it is looked up in.
Both now use 2022.

### Alignment coverage

Correspondences per ontology pair in `app/resources/mapping.rdf` (245 total):

```
                cmt     confof       ekaw conference       edas
cmt               .         24         18         32          0
confof           24          .         40         29          2
ekaw             18         40          .         50         16
conference       32         29         50          .         33
edas              0          2         16         33          .
```

Two things to know before writing questions:

- **`cmt` ↔ `edas` has zero correspondences.** A question needing both can only
  bridge them through a third ontology, and the planner does not chain
  correspondences — it looks up direct ones. Either avoid that pair or make
  transitive bridging an explicit research question.
- **`confof` ↔ `edas` has only 2.** Too thin to build a category on.
- `conference` is the best-connected (144 correspondences), which makes it a
  natural federation hub.

One malformed entry exists: `edas:Author = edas:Author`, a self-correspondence
inside one ontology. Harmless (both sides are the same source, so it can never
propose a cross-source join) but it inflates the total by one.

---

## Step 1c — completed the alignment

### What was wrong

`mapping.rdf` was assembled by hand, and three pairs were never finished. The
`edas - confOf` banner (line ~1418) is followed by 2 real confOf-edas cells
(`Country`, both directions) and then 20 confOf-**ekaw** cells pasted under the
wrong header — 19 of which already exist in the proper `Ekaw - confOf` section.
Someone opened the section, added one correspondence, pasted another pair's
block below it, and never came back. `cmt - edas` had no section at all.

Adding `edas` to the pipeline is what exposed it; the other seven pairs are fine.

**Counts are direction-doubled.** The file stores most correspondences twice,
once per direction, so `store.stats()` totals are roughly 2x the number of
distinct correspondences. Any number reported from this file needs that said out
loud, or it looks twice as rich as it is.

### What was done

`python eval/merge_reference_alignments.py --write` merged the official OAEI
2022 conference reference alignment into the three broken pairs only:

| pair | before | after | added |
|---|---:|---:|---|
| cmt – edas | 0 | 26 | 13 correspondences |
| confof – edas | 2 | 38 | 18 |
| edas – ekaw | 16 | 56 | 20 |
| *other seven pairs* | | *unchanged* | 0 |

Total 245 -> 347. Existing correspondences are matched direction-insensitively,
so nothing already present was re-added, and the seven complete pairs were not
touched.

**The EDOAL entries survived intact** — `restriction_type 7,
restriction_value 1, restriction_occurrence 2, union 3`, the same 13 as before.
These are the conditional correspondences (`ekaw:Accepted_Paper = cmt:Paper AND
(hasDecision -> Acceptance)`), they are not in the official level-0 reference,
and they are the more interesting half of the contribution.

A backup of the original sits at `app/resources/mapping.rdf.bak`.

### Two things to know

**14 of the 51 added correspondences are property-level** (`hasAuthor =
isWrittenBy`), because the official reference includes them. They are currently
**inert**: `AlignmentStore` indexes every correspondence by URI, but
`plan_federation` only ever looks up the *class* URIs of path blocks, so a
property correspondence is stored and never retrieved. Harmless, and useful if
property alignment becomes a direction later — but do not count them when
reporting how much alignment the system actually uses.

**Coverage after the merge** — every ontology now reaches every other:

```
                cmt     confof       ekaw conference       edas
cmt               .         24         18         32         26
confof           24          .         40         29         38
ekaw             18         40          .         50         56
conference       32         29         50          .         33
edas             26         38         56         33          .
```

`cmt:Paper` now has counterparts in all four other ontologies
(`confof:Contribution`, `edas:Paper`, `ekaw:Paper`, `conference:Accepted_contribution`,
plus the conditional `ekaw:Accepted_Paper` / `ekaw:Rejected_Paper`), and so does
`cmt:Author`. Before the merge, any question touching `cmt` and `edas` together
had nothing to join on.

---

## Step 2 — gold question set (done)

**42 questions** in [`questions/gold_questions.json`](questions/gold_questions.json).
Full write-up, categories and judgement calls: [`questions/README.md`](questions/README.md).

| category | n | |
|---|---:|---|
| multi-source coverage | 13 | same concept, different names — UNION over SERVICE blocks |
| cross-source join | 14 | two CQAs composed, neither source can answer both halves |
| conditional correspondence | 8 | covers 10 of the 13 non-simple cells in `mapping.rdf` |
| distractors | 7 | sound cross-source, are single-source |

**No question names an ontology.** The system is meant to look like one system:
which ontologies answer a question is its own business, not the asker's. A
category of five single-source questions was dropped for breaking this — their
wording named the ontology, which hands the source set to the pipeline and then
measures whether it can read it back. The distractors already cover
single-source answers across all five ontologies, honestly.

Per-source SPARQL is lifted verbatim from the published IRIT CQA dataset
(`cqa_source/`), so the vocabulary is the dataset's choice and not ours; the
upstream id is kept on every entry. Nothing came from the pipeline's output.

```bash
python eval/build_gold_questions.py     # rebuild the JSON from the CQAs
python eval/verify_gold_questions.py    # parses / URIs exist / executes
```

Last verification: 47/47 parse, 47/47 URI-checked against the OAEI 2022 files
(117 `SERVICE` blocks), 47/47 execute against Fuseki without an HTTP error.

### Also: correspondences the CQAs imply and we do not have

A side product of reading the CQAs — see
[`derived_mappings/report.md`](derived_mappings/report.md). 295 implied
correspondences, 137 already in `mapping.rdf`, **152 new, 125 of them complex**.
33 could be written as EDOAL cells; the rest have no shape in the format this
project parses. `app/resources/mapping.rdf` was not touched.

---

## Step 3 — metrics, without instance data

The endpoints are empty, so answer-set F1 is not available: every query returns
zero rows and a correct one scores the same as a broken one. The comparison
happens at the query level instead.

**What this costs.** No end-to-end proof that a query returns the right answer.
A query can be structurally right and still wrong on real data — a missing
`DISTINCT`, a join on the wrong variable, an `OPTIONAL` that should be required.

**What it does not cost.** The claim being made is about *vocabulary and source
selection*, not about data. Every stage of the pipeline decides which ontology,
which class, which property, which endpoint — and all of that is visible in the
query text. Answer F1 would in fact blur it, since a wrong source often still
returns plausible rows.

Built in [`harness/`](harness/README.md) — how to run it is documented there.
What follows is why these numbers and not others.

### Source selection — the headline number

Which endpoints the query calls, against the gold source set. This is the
cleanest measurement of the contribution, needs no data at all, and is the least
disturbed by the model rewording things.

Scored on the sources **actually emitted in the query**, not on
`participating_sources`. The planner counts a source as participating whenever
any correspondence reaches it, so it almost never excludes anything — on
"which papers were accepted?" it returned all five while the query used four.
Scoring its output would flatter the result.

### Query-level F1

Both queries are normalised (prefixes expanded, split into `SERVICE` blocks,
patterns extracted) and compared as sets of triple patterns per block, at two
strictnesses:

- **lenient** — variable names blanked out. Did it pick the right classes and
  properties, in the right ontology?
- **strict** — each variable renamed after the predicates and positions it
  appears in. Did it wire them together correctly?

Both are needed: a query can use every right term and still join them wrongly,
and one number cannot say which happened. Strict is computed per ontology —
computed across the whole query, one wrong pattern dragged every other
ontology's correct patterns to zero with it.

**Join F1** is separate: which pairs of sources share a variable *inside the
same UNION branch*. Two blocks on either side of a `UNION` are alternatives, not
a join, and counting them as one would make every union question look like a
perfect cross-source join.

**Exact match** is computed but stays a footnote. With a non-deterministic model
it measures luck as much as correctness.

### Alignment actually used

Every gold question lists the correspondences it is designed to need. The
harness checks how many of them reached the query, and separately how much of
what the planner retrieved was taken up. Low recall with high uptake means
retrieval is the problem; high uptake alongside missing correspondences means
the generator ignored what it was handed. Those are different bugs and were
confused for each other before this was measured.

### Validity

% that parse (`rdflib.plugins.sparql.prepareQuery`), and % that execute without
an HTTP error against the real endpoints. Both work on empty stores. These
separate "wrong query" from "broken query" — several recent bugs were the second
kind, and an F1 alone hides them.

### Per stage

How much of the gold vocabulary still existed after Agent 2 and after Agent 4,
and which sources Agent 6 kept. Recall only: these stages are meant to
over-supply, so precision would punish them for offering options the generator
did not need. This is what lets a low final score be traced to the stage that
dropped it.

### Repetition

**Run each question 3–5 times and report mean ± spread.** Agent 6 is
non-deterministic (temperature 0.2, no seed, reasoning model, batched concurrent
calls), and its per-ontology output is all-or-nothing — a borderline judgment
drops a whole source, not one path. A single-run table would be misleading.

The harness flags a question as *unstable* when the repeats do not agree. That
split matters more than the average: a wrong answer that repeats identically is
a structural bug worth fixing, one that varies is model noise. Start at
`--repeats 1` to find the first kind; raise it once they are gone.

---

## Step 4 — ablations

Same questions, four configurations:

1. no alignment — dump all five schemas, let the model sort it out
2. alignment only at generation time (drop `expand_matches_with_alignment`) —
   isolates the "alignment as a retrieval operator" claim
3. full pipeline
4. full pipeline with a frontier model — separates the pipeline's contribution
   from the model's
