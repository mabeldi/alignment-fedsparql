"""Check the gold question set three ways before anything is scored against it.

A gold query that names a term the ontology does not have corrupts every metric
silently: the pipeline is marked wrong for not producing a term that could never
exist. So this runs before the harness, not after a suspicious result.

  1. every gold query parses            (rdflib prepareQuery)
  2. every class/property URI exists     (the OAEI 2022 files themselves)
  3. every gold query executes           (the real Fuseki endpoints)

Empty result sets are expected and are not a failure — the endpoints hold no
instance data. Check 3 only separates "well-formed and reachable" from "broken".

Run:  python eval/verify_gold_questions.py
      python eval/verify_gold_questions.py --offline   # skip check 3
Env:  FUSEKI_BASE_URL (default http://192.168.100.44:3030)

Ontology files are cached under eval/.cache/ontologies so a re-run does not
re-download them; delete that folder to refresh.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import rdflib
import requests
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.plugins.sparql import prepareQuery

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "questions" / "gold_questions.json"
CACHE = ROOT / "eval" / ".cache" / "ontologies"

BASE = os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030").rstrip("/")

# Must be the same files app/main.py loads: a term present in one OAEI year and
# absent in another would pass here and fail in the pipeline.
ONTOLOGY_URL = {
    s: f"http://oaei.ontologymatching.org/2022/conference/data/{s}.owl"
    for s in ("cmt", "confof", "ekaw", "conference", "edas")
}

_CLASS_TYPES = (OWL.Class, RDFS.Class)
_PROPERTY_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty,
                   OWL.AnnotationProperty, RDF.Property)

# Vocabularies the CQAs borrow from. They are not in any conference ontology and
# do not need to be.
EXTERNAL_PREFIXES = ("http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                     "http://www.w3.org/2000/01/rdf-schema#",
                     "http://www.w3.org/2001/XMLSchema#",
                     # owl:sameAs carries every cross-source join. It is standard
                     # vocabulary, so no domain ontology declares it — the triples
                     # that use it are loaded next to the instance data instead.
                     "http://www.w3.org/2002/07/owl#")


def load_terms():
    """{source: set of URIs it declares}, downloading once into eval/.cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    terms = {}
    for source, url in ONTOLOGY_URL.items():
        path = CACHE / f"{source}.owl"
        if not path.exists():
            path.write_bytes(requests.get(url, timeout=60).content)
        graph = rdflib.Graph()
        graph.parse(str(path))
        known = set()
        for rdf_type in (*_CLASS_TYPES, *_PROPERTY_TYPES):
            known.update(str(s) for s in graph.subjects(RDF.type, rdf_type)
                         if isinstance(s, rdflib.URIRef))
        terms[source] = known
    return terms


def check_parse(questions):
    failures = []
    for q in questions:
        try:
            prepareQuery(q["gold_query"])
        except Exception as exc:
            failures.append((q["id"], f"{type(exc).__name__}: {exc}"))
    return failures


def check_terms(questions, terms):
    """Every URI a block uses must be declared by that block's own ontology."""
    failures = []
    for q in questions:
        for source, block in q["sources"].items():
            for uri in block["classes"] + block["properties"]:
                if uri.startswith(EXTERNAL_PREFIXES):
                    continue
                if uri not in terms[source]:
                    failures.append((q["id"], f"{source}: {uri} not declared"))
    return failures


def check_execute(questions):
    """Send each gold query to its anchor endpoint and demand an HTTP 200.

    Sent to the first gold source's endpoint, the way the pipeline's own queries
    are driven from one source and reach the rest through SERVICE.
    """
    failures = []
    for q in questions:
        driver = q["gold_sources"][0]
        url = f"{BASE}/{driver}/sparql"
        # Retried, because a host that briefly stops accepting connections
        # produces the same red output as 47 malformed queries and would send
        # someone hunting a bug in the gold set that is not there.
        for _ in range(3):
            try:
                response = requests.get(
                    url, params={"query": q["gold_query"]},
                    headers={"Accept": "application/sparql-results+json"}, timeout=60)
                if response.ok:
                    break
                why = f"HTTP {response.status_code} {response.text.strip()[:160]}"
            except Exception as exc:
                why = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            failures.append((q["id"], why))
    return failures


def report(name: str, total: int, failures) -> bool:
    print(f"{name:24} {total - len(failures):3}/{total}")
    for qid, why in failures:
        print(f"    {qid}  {why}")
    return not failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip the endpoint execution check")
    args = ap.parse_args()

    questions = json.loads(GOLD.read_text(encoding="utf-8"))["questions"]
    print(f"{GOLD}  ({len(questions)} questions)\n")

    ok = report("parse", len(questions), check_parse(questions))
    ok &= report("URIs exist", len(questions), check_terms(questions, load_terms()))

    blocks = sum(len(q["sources"]) for q in questions)
    print(f"{'SERVICE blocks':24} {blocks}")

    if args.offline:
        print("\n--offline: endpoints not contacted")
    else:
        print(f"\nendpoints: {BASE}")
        ok &= report("execute (HTTP 200)", len(questions), check_execute(questions))
        print("empty result sets are expected - the endpoints hold no instances")

    print("\nOK" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
