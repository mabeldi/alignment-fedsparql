# Source-exclusive accepted papers (CC-01)

Adds 20 accepted papers to each of CMT, Conference, EDAS, and EKAW (80 papers).
ConfOf receives no data because it has no branch in this accepted-paper query.
The existing ontology and instance namespaces are unchanged. Only resource IDs
are new, e.g. `http://cmt-instances#paper-001-aug`.
Disjoint local-ID ranges prevent the evaluator's `same_id_key` from merging
papers across sources: CMT 001-020, Conference 021-040, EDAS 041-060, EKAW 061-080.
Fresh CMT supporting resources use `decision-001-aug` and `administrator-001-aug`.

The data uses minimal acceptance patterns, rather than copying entire existing
records and their identity links. CMT includes fresh decisions and one fresh
administrator per paper, supporting both `hasDecision / Acceptance` and
`acceptedBy` / `acceptPaper` representations. No `owl:sameAs` links are added.
These are evaluation fixtures, not complete conference-management records.

From the repository root:

```powershell
.\venv\Scripts\python.exe -X utf8 eval/seed_unique_accepted_papers.py prepare
.\venv\Scripts\python.exe -X utf8 eval/seed_unique_accepted_papers.py apply
.\venv\Scripts\python.exe -X utf8 eval/seed_unique_accepted_papers.py verify
```

To grow the existing augmentation to **500 total papers per source** automatically:

```powershell
.\venv\Scripts\python.exe -X utf8 eval/seed_unique_accepted_papers.py apply --count 500
```

This adds only the remaining 480 papers per source. IDs are allocated in fixed
20-ID blocks cycling through the four sources, preserving all existing IDs and
keeping their local parts globally unique. At 500 per source, the local paper IDs
span `paper-001-aug` through `paper-2000-aug`. No existing data is deleted.
The 500-paper manifest, triples, and verification report are stored separately in
`eval/fixtures/accepted_papers_unique_v1_500/`. Use `verify --count 500` or
`remove --count 500` for this expanded fixture. The latter removes all 500 per
source, including the original 20; the 20-paper directory remains a historical
snapshot, not the manifest for the expanded dataset.

`apply` first checks all five datasets for conflicting triples involving the
fixture IDs, then inserts only missing triples into each default graph. Repeating
it with the same parameters adds no duplicates. A partial network failure can be
recovered by repeating `apply`.

Optional rollback:

```powershell
.\venv\Scripts\python.exe -X utf8 eval/seed_unique_accepted_papers.py remove
```

Removal deletes only the fixture's exact triples, never clears a dataset. Keep
the same count and base URL when applying, verifying, or removing this version.
The generated `.ttl`, `.insert.ru`, `.remove.ru`, and `manifest.json` files record
the complete fixture. `before.json` records the first application baseline.
`verification.json` records full-reference and leave-one-source-out scores;
`reference_bindings.json` records the verified full reference answers.

Use the same augmented data in both alignment modes. Rerun reference and generated
query execution before recalculating metrics: existing saved results describe the
old dataset and query hashes alone cannot detect a dataset change. Existing model
archives are not rewritten by this script. These new Paper instances also affect
other paper-list queries, so reevaluate those against the same dataset version.

With the synthetic 80-paper subset alone, omitting one source loses 20 answers:
recall is 75% and F1 is approximately 85.71% if there are no false positives.
The full-dataset score drop depends on how many existing answers there are; consult
`verification.json` for the measured effect. Synthetic-subset scores are diagnostic
and are reported separately from the full-dataset scores.
