"""Create and verify the evaluation endpoints — one Fuseki dataset per ontology.

Idempotent: existing datasets are left alone, so this is safe to re-run.

Run:  python eval/setup_fuseki.py
Env:  FUSEKI_BASE_URL (default http://192.168.100.44:3030)

The federation check at the end is not optional decoration. Every generated query
puts each source in its own SERVICE block, so if outbound SERVICE is blocked the
whole evaluation returns empty result sets and looks like a pipeline failure.
Better to find that out here than halfway through a run.
"""

import os
import sys

import requests

BASE = os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030").rstrip("/")

# Must match ONTOLOGIES in app/main.py and ONTOLOGY_URLS in
# app/embeddings/index_ontology.py — a source without a dataset here has an
# endpoint in the prompt that nothing answers.
DATASETS = ["cmt", "confof", "ekaw", "conference", "edas"]

DB_TYPE = "tdb2"


def existing() -> set:
    r = requests.get(f"{BASE}/$/datasets", timeout=10)
    r.raise_for_status()
    return {d["ds.name"].lstrip("/") for d in r.json().get("datasets", [])}


def create(name: str) -> None:
    r = requests.post(f"{BASE}/$/datasets",
                      params={"dbType": DB_TYPE, "dbName": name}, timeout=30)
    r.raise_for_status()


def count_triples(name: str) -> int:
    r = requests.get(f"{BASE}/{name}/sparql",
                     params={"query": "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"},
                     headers={"Accept": "application/sparql-results+json"}, timeout=30)
    r.raise_for_status()
    return int(r.json()["results"]["bindings"][0]["n"]["value"])


def federation_works() -> bool:
    """One dataset queried, the others reached through SERVICE — the shape every
    generated query uses."""
    driver, *others = DATASETS
    blocks = " ".join(
        f"SERVICE <{BASE}/{d}/sparql> {{ SELECT (COUNT(*) AS ?n{i}) WHERE {{ ?s ?p ?o }} }}"
        for i, d in enumerate(others)
    )
    r = requests.get(f"{BASE}/{driver}/sparql",
                     params={"query": f"SELECT * WHERE {{ {blocks} }}"},
                     headers={"Accept": "application/sparql-results+json"}, timeout=60)
    return r.ok and bool(r.json()["results"]["bindings"])


def main() -> int:
    print(f"Fuseki: {BASE}")
    try:
        have = existing()
    except Exception as exc:
        print(f"  cannot reach the admin API ({exc})")
        print("  the admin API is localhost-only by default — check shiro.ini")
        return 1

    for name in DATASETS:
        if name in have:
            print(f"  {name:8} already exists")
        else:
            create(name)
            print(f"  {name:8} created ({DB_TYPE})")

    print("\nTriple counts:")
    for name in DATASETS:
        print(f"  {name:8} {count_triples(name)}")

    ok = federation_works()
    print(f"\nSERVICE federation: {'OK' if ok else 'BLOCKED'}")
    if not ok:
        print("  a cross-endpoint query returned nothing — every evaluation query")
        print("  would come back empty. Fix this before generating any data.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
