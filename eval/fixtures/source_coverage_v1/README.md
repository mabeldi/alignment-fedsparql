# Source coverage augmentation

This fixed fixture targets question runs whose saved normalized answer F1 was 1
while source-selection F1 was below 1. `manifest.json` records the inspected
references, source sets, qualifying runs, and selected counts. All metrics under
`app/agents/evaluation/results`, including model archives, are inspected; data is
added once per qualifying question, not once per model. The dinner question
qualifies through run 2 even though its aggregate answer F1 is below 1.

| Question | Added cases | Sources |
| --- | --- | --- |
| Conference locations and registration deadlines | 20 linked pairs | Conference + EDAS |
| Conference start dates | 20 exclusive conferences per source | Conference, ConfOf, EDAS |
| Conference dinners | 20 exclusive dinners per source | ConfOf, EDAS, EKAW |
| Early registration | 400 exclusive participants per source | Conference, ConfOf, EKAW |
| Topics | 800 exclusive topics per source | CMT, Conference, ConfOf, EDAS, EKAW |

For UNION cases, counts target at least a ten-percentage-point recall loss for
one omitted source: n / (existing normalized answers + source count * n) >= 0.10.
Use a minimum of 20; larger counts round up to the next 100. Sources receive
balanced additions; the code does not only enrich whichever source a model missed.

All new resource IDs end in `-aug`, with globally distinct local IDs for UNION
answers. Namespace IRIs are unchanged. IDs use question-specific prefixes and
fixed 20-ID blocks across sources, separate from existing `paper-NNN-aug` data.
No existing resource is duplicated or overwritten.

The location/deadline question is a JOIN, so it receives source-local conference
copies with matching IDs and explicit `owl:sameAs` links in both endpoints.
Location text exists only in Conference, and deadlines only in EDAS; there are
no ConfOf copies. Joining the required sources recovers the complete tuples.
The other four cohorts have no cross-source identity links.

The current early-registration reference was corrected from plain string
`confof:earlyRegistration "true"` to boolean `confof:earlyRegistration true`.
The ontology declares xsd:boolean, and live ConfOf values are xsd:boolean. The
original string pattern returned no ConfOf answers. This correction makes the
reference match the existing representation and the new valid boolean data.
This is the only reference edit; generation prompts and scoring code are unchanged.

Run from the repository root:

```powershell
.\venv\Scripts\python.exe -X utf8 eval/seed_source_coverage.py prepare
.\venv\Scripts\python.exe -X utf8 eval/seed_source_coverage.py apply
.\venv\Scripts\python.exe -X utf8 eval/seed_source_coverage.py verify
```

`prepare` generates exact Turtle, INSERT/DELETE statements, and manifest files,
then validates all reference queries and source omissions against local graphs.
`apply` inspects every endpoint before writing, refuses unexpected data involving
fixture IDs, and adds only missing triples. It is safe to repeat after partial
network failure. `remove` deletes only the fixture's exact triples and does not
revert the reference's boolean correction or remove accepted-paper augmentation.

`before.json` and `after.json` contain ordinary answer precision/recall/F1 for the
unchanged qualifying generated queries, using the existing evaluator. They are
diagnostic measurements, not a new evaluation metric. Full reference bindings are
saved alongside them; `verified.json` is written only after live verification.

The script does not overwrite existing `run_N_result.json`, `reference_result.json`,
or `metrics.json` files. Reexecute references and generated queries and recalculate
metrics for a consistent augmented dataset snapshot. Use the same snapshot in
both alignment modes and for all models. New conference/participant instances can
also affect other questions, so rerun the complete evaluation rather than mixing
old and new execution results.
