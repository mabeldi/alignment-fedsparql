# Federated CQA validation queries

Each `.sparql` file corresponds to one folder in `../CQAs`. The compact file
name is the CQA identifier, and the first comment contains the complete natural-
language question.

The ontology-specific queries inside each source folder independently answer
the same question and project the same variables. Their `SERVICE` blocks are
therefore combined with `UNION`, which collects answers from every source.

For a question that instead requests complementary information from different
sources, use `owl:sameAs` to bridge the source-local individual to the anchor
individual. Do not assume two endpoints use the same IRI for one real entity.
For example:

```sparql
SERVICE <http://192.168.100.44:3030/conference/sparql> {
  ?paper a <http://conference#Written_contribution> .
  ?paper <http://conference#has_an_abstract> ?abstract .
}
OPTIONAL {
  SERVICE <http://192.168.100.44:3030/edas/sparql> {
    ?paper_edas <http://www.w3.org/2002/07/owl#sameAs> ?paper .
    ?paper_edas <http://edas#hasTopic> ?topic .
  }
}
```

Use the shared anchor (`?paper`) as the join key. Wrap the complementary service
in `OPTIONAL` only when its information is allowed to be absent; otherwise keep
it as a required `SERVICE` join.

Regenerate and validate the files from the repository root:

```powershell
python eval\build_cqa_validation.py
python eval\build_cqa_validation.py --check
```

Set `FUSEKI_BASE_URL` before generation to use a Fuseki server other than the
default `http://192.168.100.44:3030`.
