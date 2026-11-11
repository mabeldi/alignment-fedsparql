"""Add a reversible CC-01 fixture to the existing Fuseki DEFAULT graphs.

prepare writes reviewable Turtle and SPARQL update files without network access.
apply checks all five endpoints for collisions, inserts only missing triples,
and verifies answer loss when each participating source is omitted.
remove deletes only this fixture's exact triples. No CLEAR/DROP is used.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import requests
from rdflib import Graph, Namespace, RDF, URIRef


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "agents" / "evaluation"))
from calculate_metrics import same_id_key, set_scores

VERSION = "eval-cc01-v1"
SOURCES = ("cmt", "conference", "edas", "ekaw")
ALL_SOURCES = (*SOURCES, "confof")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixtures" / "accepted_papers_unique_v1"
# Keep the first installed 20 IDs per source stable when increasing --count.
# Allocate successive 20-ID blocks in source order: 1-20 CMT, 21-40 Conference,
# 41-60 EDAS, 61-80 EKAW, 81-100 CMT, and so on.
ID_BLOCK_SIZE = 20
PATTERNS = {
    "cmt": "?paper <http://cmt#hasDecision> ?decision . ?decision a <http://cmt#Acceptance> .",
    "conference": "?paper a <http://conference#Accepted_contribution> .",
    "edas": "?paper a <http://edas#AcceptedPaper> .",
    "ekaw": "?paper a <http://ekaw#Accepted_Paper> .",
}


def make_fixture(count):
    if count < 1:
        raise ValueError("count must be positive")
    graphs, papers = {}, {}
    for source_index, source in enumerate(SOURCES):
        ns = Namespace(f"http://{source}#")
        instances = Namespace(f"http://{source}-instances#")
        graph = Graph()
        papers[source] = []
        for offset in range(count):
            block, position = divmod(offset, ID_BLOCK_SIZE)
            index = (block * len(SOURCES) + source_index) * ID_BLOCK_SIZE + position + 1
            paper = instances[f"paper-{index:03d}-aug"]
            papers[source].append(str(paper))
            graph.add((paper, RDF.type, ns.Paper))
            if source == "cmt":
                decision = instances[f"decision-{index:03d}-aug"]
                administrator = instances[f"administrator-{index:03d}-aug"]
                graph.add((paper, RDF.type, ns.PaperFullVersion))
                graph.add((paper, ns.hasDecision, decision))
                graph.add((decision, RDF.type, ns.Acceptance))
                graph.add((administrator, RDF.type, ns.Administrator))
                graph.add((paper, ns.acceptedBy, administrator))
                graph.add((administrator, ns.acceptPaper, paper))
            else:
                accepted = {"conference": "Accepted_contribution",
                            "edas": "AcceptedPaper", "ekaw": "Accepted_Paper"}[source]
                graph.add((paper, RDF.type, ns[accepted]))
        graphs[source] = graph
    return graphs, papers


def triples(graph):
    return "\n".join(sorted(" ".join(term.n3() for term in triple) + " ." for triple in graph))


def reference_query(base, sources=SOURCES):
    branches = [f"{{ SERVICE <{base}/{s}/sparql> {{ {PATTERNS[s]} }} }}" for s in sources]
    return "SELECT DISTINCT ?paper WHERE {\n" + "\nUNION\n".join(branches) + "\n}\n"


def prepare(output, count, base):
    graphs, papers = make_fixture(count)
    manifest = {"fixture": VERSION, "papers_per_source": count,
                "base_url": base, "papers": papers,
                "triples_per_source": {s: len(g) for s, g in graphs.items()},
                "identity_policy": "paper-NNN-aug IDs use disjoint number ranges across sources; no owl:sameAs links."}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing fixture parameters differ; use a separate output directory.")
    output.mkdir(parents=True, exist_ok=True)
    for source, graph in graphs.items():
        body = triples(graph)
        (output / f"{source}.ttl").write_text(body + "\n", encoding="utf-8")
        for operation, filename in (("INSERT", "insert"), ("DELETE", "remove")):
            (output / f"{source}.{filename}.ru").write_text(
                f"{operation} DATA {{\n{body}\n}}\n", encoding="utf-8")
    (output / "reference.rq").write_text(reference_query(base), encoding="utf-8")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return graphs, papers


class Fuseki:
    def __init__(self, base):
        self.base = base

    def select(self, source, query):
        response = requests.post(f"{self.base}/{source}/sparql", data={"query": query},
                                 headers={"Accept": "application/sparql-results+json"}, timeout=45)
        response.raise_for_status()
        return response.json()["results"]["bindings"]

    def related_triples(self, source, subjects):
        values = " ".join(URIRef(uri).n3() for uri in sorted(subjects))
        query = ("CONSTRUCT { ?s ?p ?o } WHERE { "
                 f"{{ VALUES ?s {{ {values} }} ?s ?p ?o }} UNION "
                 f"{{ VALUES ?o {{ {values} }} ?s ?p ?o }} }}")
        response = requests.post(f"{self.base}/{source}/sparql", data={"query": query},
                                 headers={"Accept": "text/turtle"}, timeout=45)
        response.raise_for_status()
        return Graph().parse(data=response.text, format="turtle")

    def update(self, source, operation, graph):
        response = requests.post(f"{self.base}/{source}/update",
                                 data={"update": f"{operation} DATA {{\n{triples(graph)}\n}}"},
                                 timeout=45)
        response.raise_for_status()


def inspect_fixture(client, graphs):
    subjects = {str(s) for graph in graphs.values() for s in graph.subjects()}
    found = {}
    # Read every endpoint before any write, including ConfOf, which gets no data.
    for source in ALL_SOURCES:
        graph = client.related_triples(source, subjects)
        expected = set(graphs.get(source, Graph()))
        unexpected = set(graph) - expected
        if unexpected:
            raise ValueError(f"{source}: fixture IDs have unexpected triples or cross-source links; refusing mutation.")
        found[source] = graph
    return found


def normalized(bindings):
    return {same_id_key(row["paper"]) for row in bindings}


def verify(client, graphs, papers, output):
    found = inspect_fixture(client, graphs)
    for source in SOURCES:
        if set(found[source]) != set(graphs[source]):
            raise ValueError(f"{source}: fixture is incomplete")
    full_query = reference_query(client.base)
    full_bindings = client.select("cmt", full_query)
    full = normalized(full_bindings)
    synthetic = {same_id_key({"value": uri}) for uris in papers.values() for uri in uris}
    if len(synthetic) != sum(map(len, papers.values())) or not synthetic <= full:
        raise ValueError("Synthetic papers are missing or collapse under evaluator normalization")
    report = {"fixture": VERSION, "verified_at": datetime.now(timezone.utc).isoformat(),
              "base_url": client.base, "reference_raw_rows": len(full_bindings),
              "reference_normalized_rows": len(full), "synthetic_papers": len(synthetic),
              "accepted_papers_per_source": {},
              "omitted_sources": {}}
    for source in SOURCES:
        count_rows = client.select(source,
            f"SELECT (COUNT(DISTINCT ?paper) AS ?count) WHERE {{ {PATTERNS[source]} }}")
        report["accepted_papers_per_source"][source] = int(count_rows[0]["count"]["value"])
        remaining = tuple(s for s in SOURCES if s != source)
        rows = normalized(client.select(remaining[0], reference_query(client.base, remaining)))
        expected_missing = {same_id_key({"value": uri}) for uri in papers[source]}
        missing_synthetic = synthetic - rows
        if missing_synthetic != expected_missing:
            raise ValueError(f"Omitting {source} did not lose exactly its synthetic papers")
        report["omitted_sources"][source] = {
            "missing_synthetic": len(missing_synthetic), "missing_total": len(full - rows),
            "full_answer_scores": set_scores(full, rows),
            "synthetic_answer_scores": set_scores(synthetic, rows & synthetic),
        }
        print(f"Without {source}: missing {len(missing_synthetic)} synthetic papers; "
              f"full-answer recall={len(rows & full) / len(full):.6f}", flush=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "reference_bindings.json").write_text(json.dumps({"query": full_query,
        "verified_at": report["verified_at"], "bindings": full_bindings}, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "verify", "remove"))
    parser.add_argument("--count", type=int, default=20,
                        help="Total augmented papers per source, including previously added papers (default: 20).")
    parser.add_argument("--base-url", default=os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030"))
    parser.add_argument("--output-dir", type=Path,
                        help="Artifact directory; defaults to a separate directory for each count.")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.output_dir is None:
        args.output_dir = (DEFAULT_OUTPUT if args.count == 20 else
                           DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.name}_{args.count}"))
    graphs, papers = prepare(args.output_dir, args.count, base)
    print(f"Fixture: {args.count} papers x {len(SOURCES)} sources; {args.output_dir}", flush=True)
    if args.action == "prepare":
        return
    client = Fuseki(base)
    if args.action == "verify":
        verify(client, graphs, papers, args.output_dir)
        return
    found = inspect_fixture(client, graphs)
    if args.action == "apply":
        before = {s: len(client.select(s, f"SELECT DISTINCT ?paper WHERE {{ {PATTERNS[s]} }}"))
                  for s in SOURCES}
        # Preserve the first baseline across repeated applications.
        baseline = args.output_dir / "before.json"
        if not baseline.exists():
            baseline.write_text(json.dumps({"accepted_papers": before,
                "existing_fixture_triples": {s: len(g) for s, g in found.items()}}, indent=2) + "\n", encoding="utf-8")
        for source in SOURCES:
            missing = graphs[source] - found[source]
            if missing:
                client.update(source, "INSERT", missing)
            print(f"{source}: inserted {len(missing)} triples", flush=True)
        verify(client, graphs, papers, args.output_dir)
    else:
        for source in SOURCES:
            if found[source]:
                client.update(source, "DELETE", found[source])
            print(f"{source}: removed {len(found[source])} fixture triples", flush=True)
        if any(inspect_fixture(client, graphs).values()):
            raise ValueError("Fixture removal is incomplete")
        print("Fixture removed; existing dataset triples preserved.", flush=True)


if __name__ == "__main__":
    main()
