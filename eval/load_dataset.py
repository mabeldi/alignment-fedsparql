"""Load the populated instance data into the Fuseki endpoints — one dataset per ontology.

Run:  python eval/load_dataset.py              # loads data_60 (default)
      python eval/load_dataset.py --level 100
      python eval/load_dataset.py --keep        # add to what's there instead of replacing
      python eval/load_dataset.py --check       # verify only, load nothing
Env:  FUSEKI_BASE_URL (default http://192.168.100.44:3030)

Each dataset gets TWO files, both into its DEFAULT graph:

  <source>_<level>.ttl            the ontology plus its own instances
  alignment_<source>_<level>.ttl  owl:sameAs from this source's instances to
                                  their twins in the other four ontologies

The alignment file is not optional decoration. Two endpoints never share a URI,
so a cross-source join has to hop through owl:sameAs, and the generated queries
resolve that hop inside the TARGET endpoint:

    SERVICE <cmt>        { ?a a <http://cmt#Paper> . }
    SERVICE <conference> { ?b owl:sameAs ?a . ?b ...  }

That only works if conference's own alignment file sits in conference's dataset.
Load the alignments anywhere else — a shared dataset, a named graph — and every
join in the benchmark silently returns zero rows.

Run eval/setup_fuseki.py first; this script assumes the datasets already exist.
"""

import argparse
import os
import sys
from pathlib import Path

import requests

BASE = os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030").rstrip("/")

DATA_ROOT = (Path(__file__).resolve().parents[1]
             / "app" / "resources" / "dataset" / "populated_datasets")

# Fuseki dataset name -> file-name stem. They differ: the endpoints are all
# lowercase, the upstream files keep the ontology's own casing ("confOf").
DATASETS = {
    "cmt": "cmt",
    "confof": "confOf",
    "ekaw": "ekaw",
    "conference": "conference",
    "edas": "edas",
}

SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"

# Long, because the biggest file is ~42 MB and TDB2 parses on the way in.
UPLOAD_TIMEOUT = 900


def ask(dataset: str, query: str, timeout: int = 120) -> dict:
    r = requests.get(f"{BASE}/{dataset}/sparql", params={"query": query},
                     headers={"Accept": "application/sparql-results+json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def count(dataset: str, query: str) -> int:
    return int(ask(dataset, query)["results"]["bindings"][0]["n"]["value"])


def count_all(dataset: str) -> int:
    return count(dataset, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }")


def count_sameas(dataset: str) -> int:
    return count(dataset, f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{SAME_AS}> ?o }}")


def clear(dataset: str) -> None:
    """Drop the default graph. Re-loading without this would duplicate the blank
    nodes in the OWL restrictions — plain triples dedupe, blank nodes don't."""
    r = requests.delete(f"{BASE}/{dataset}/data", params={"default": ""}, timeout=300)
    if r.status_code not in (200, 204, 404):
        r.raise_for_status()


def upload(dataset: str, path: Path) -> None:
    """POST into the default graph. No graph= parameter on purpose: a named graph
    would hide these triples from an ordinary WHERE clause."""
    with path.open("rb") as fh:
        r = requests.post(f"{BASE}/{dataset}/data",
                          params={"default": ""},
                          data=fh,
                          headers={"Content-Type": "text/turtle; charset=utf-8"},
                          timeout=UPLOAD_TIMEOUT)
    r.raise_for_status()


def files_for(level: int, stem: str) -> list:
    """Instance file first, then alignment. data_0 has no alignment file — zero
    overlap means there is nothing to link."""
    root = DATA_ROOT / f"data_{level}"
    out = [root / f"{stem}_{level}.ttl"]
    alignment = root / f"alignment_{stem}_{level}.ttl"
    if alignment.exists():
        out.append(alignment)
    return out


def bridge_works(anchor: str, target: str) -> int:
    """The exact shape a generated cross-source query uses: anchor binds an
    individual, the target endpoint resolves its own twin from that URI.
    Returns how many of 5 sampled individuals found a twin."""
    q = f"""
    SELECT ?a ?b WHERE {{
      {{ SELECT ?a WHERE {{ SERVICE <{BASE}/{anchor}/sparql> {{ ?a <{SAME_AS}> ?any . }} }} LIMIT 5 }}
      SERVICE <{BASE}/{target}/sparql> {{ ?b <{SAME_AS}> ?a . }}
    }}"""
    return len(ask(anchor, q, timeout=180)["results"]["bindings"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=60,
                    help="overlap level: 0, 20, 40, 60, 80 or 100 (default 60)")
    ap.add_argument("--keep", action="store_true",
                    help="add to existing contents instead of replacing them")
    ap.add_argument("--check", action="store_true",
                    help="verify what is loaded, change nothing")
    args = ap.parse_args()

    root = DATA_ROOT / f"data_{args.level}"
    if not args.check and not root.is_dir():
        print(f"no such dataset: {root}")
        return 1

    print(f"Fuseki: {BASE}")

    if not args.check:
        missing = [p for stem in DATASETS.values()
                   for p in files_for(args.level, stem) if not p.exists()]
        if missing:
            for p in missing:
                print(f"  missing file: {p}")
            return 1

        print(f"Source:  {root}\n")
        for ds, stem in DATASETS.items():
            try:
                before = count_all(ds)
            except Exception as exc:
                print(f"  {ds:11} cannot query it ({exc}) — run eval/setup_fuseki.py first")
                return 1

            if not args.keep:
                if before:
                    print(f"  {ds:11} clearing {before} existing triples")
                clear(ds)

            for path in files_for(args.level, stem):
                size = path.stat().st_size / 1e6
                print(f"  {ds:11} loading {path.name} ({size:.0f} MB) ...", flush=True)
                upload(ds, path)
        print()

    print("Loaded:")
    bad = []
    for ds in DATASETS:
        total, links = count_all(ds), count_sameas(ds)
        print(f"  {ds:11} {total:>9} triples   {links:>8} owl:sameAs")
        if total == 0:
            bad.append(f"{ds} is empty")
        elif links == 0 and args.level != 0:
            bad.append(f"{ds} has no owl:sameAs — it can never be a join target")

    # One real cross-endpoint hop per source, each anchored somewhere else, so
    # every dataset gets tested as a target at least once.
    if args.level != 0:
        print("\nCross-source bridges (5 sampled individuals each):")
        names = list(DATASETS)
        for i, target in enumerate(names):
            anchor = names[(i + 1) % len(names)]
            try:
                hits = bridge_works(anchor, target)
            except Exception as exc:
                print(f"  {anchor:11} -> {target:11} FAILED ({exc})")
                bad.append(f"bridge {anchor} -> {target} errored")
                continue
            print(f"  {anchor:11} -> {target:11} {hits}/5")
            if hits == 0:
                bad.append(f"bridge {anchor} -> {target} found nothing")

    if bad:
        print("\nProblems:")
        for b in bad:
            print(f"  {b}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
