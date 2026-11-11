"""Augment reference cases whose saved answer F1 hides missing sources.

prepare is offline; apply inserts only missing fixture triples into default graphs;
verify executes reference and generated queries into separate diagnostic artifacts;
remove deletes only exact fixture triples. Existing evaluation results are untouched.
"""

import argparse
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import requests
from rdflib import Dataset, Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef, XSD

from seed_unique_accepted_papers import Fuseki, ROOT, ALL_SOURCES, triples
from calculate_metrics import answer_scores, rows_for, same_id_key, source_endpoints


RESULTS = ROOT / "app" / "agents" / "evaluation" / "results"
OUTPUT = Path(__file__).resolve().parent / "fixtures" / "source_coverage_v1"
BASE = "http://192.168.100.45:3030"
VERSION = "source-coverage-v1"
CASES = {
    "locations": {
        "folder": "Retrieve_a_list_of_conferences_including_their_locations_and_registration_deadlines",
        "sources": ["conference", "edas"], "shape": "join", "prefix": "conference-location"},
    "start_dates": {
        "folder": "What_is_the_start_date_of_each_conference",
        "sources": ["conference", "confof", "edas"], "shape": "union", "prefix": "conference-start"},
    "dinners": {
        "folder": "Which_conference_dinners_are_organised",
        "sources": ["confof", "edas", "ekaw"], "shape": "union", "prefix": "dinner"},
    "early_registration": {
        "folder": "Which_participants_registered_early",
        "sources": ["conference", "confof", "ekaw"], "shape": "union", "prefix": "participant-early"},
    "topics": {
        "folder": "Which_topics_are_listed",
        "sources": list(ALL_SOURCES), "shape": "union", "prefix": "topic"},
}
SERVICE = re.compile(r"\bSERVICE\s+(<[^>]+>)", re.I)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_count(baseline, source_count, target_loss=0.10):
    # n / (baseline + source_count*n) >= target_loss, rounded up for repeatability.
    minimum = math.ceil(target_loss * baseline / (1 - target_loss * source_count))
    return 20 if minimum <= 20 else math.ceil(minimum / 100) * 100


def discover():
    known = {case["folder"]: key for key, case in CASES.items()}
    cases, scanned = {}, 0
    for path in sorted(RESULTS.rglob("metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))
        scanned += 1
        qualifying = [run["run"] for run in metrics.get("runs", [])
                      if (run.get("answer") or {}).get("f1") == 1
                      and (run.get("source_selection") or {}).get("f1", 1) < 1]
        if not qualifying:
            continue
        if path.parent.name not in known:
            raise ValueError(f"Unsupported qualifying reference: {path}; inspect before adding data.")
        key = known[path.parent.name]
        if key in cases:
            continue  # One data fixture per question, not per archived model.
        definition = CASES[key]
        reference = (path.parent / "reference.rq").read_text(encoding="utf-8-sig")
        expected_endpoints = {f"{BASE}/{s}/sparql" for s in definition["sources"]}
        if source_endpoints(reference) != expected_endpoints:
            raise ValueError(f"{key}: reference endpoints changed; review fixture definition.")
        saved = json.loads((path.parent / "reference_result.json").read_text(encoding="utf-8-sig"))
        baseline = len(rows_for(saved, saved["variables"], same_id_key))
        count = 20 if definition["shape"] == "join" else select_count(baseline, len(definition["sources"]))
        cases[key] = {**definition, "path": str(path.parent.relative_to(ROOT)),
                      "qualifying_runs": qualifying, "saved_baseline_normalized_rows": baseline,
                      "count": count, "reference": reference, "reference_sha256": digest(reference)}
    return {"fixture": VERSION, "base_url": BASE, "metrics_scanned": scanned, "cases": cases}


def ns(source):
    return Namespace(f"http://{'confOf' if source == 'confof' else source}#")


def resource(source, kind, index):
    # Namespace unchanged; resource kind distinguishes these from paper-NNN-aug.
    stem = "confOf" if source == "confof" else source
    return URIRef(f"http://{stem}-instances#{kind}-{index:03d}-aug")


def local_index(source, index):
    block, position = divmod(index, 20)
    return (block * len(ALL_SOURCES) + ALL_SOURCES.index(source)) * 20 + position + 1


def build_fixture(plan):
    graphs = {s: Graph() for s in ALL_SOURCES}
    cases = {key: {s: Graph() for s in ALL_SOURCES} for key in plan["cases"]}
    subjects = {key: {s: [] for s in case["sources"]} for key, case in plan["cases"].items()}
    schema = {s: Graph().parse(ROOT / "app" / "agents" / "schema-dataset" / f"{s}.owl", format="xml")
              for s in ALL_SOURCES}

    def typed(g, source, entity, cls):
        # Include explicit named ancestors for endpoints without RDFS entailment.
        pending, seen = [cls], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            g.add((entity, RDF.type, current))
            pending.extend(p for p in schema[source].objects(current, RDFS.subClassOf) if isinstance(p, URIRef))

    for key, case in plan["cases"].items():
        for source in case["sources"]:
            g, ontology = cases[key][source], ns(source)
            for offset in range(case["count"]):
                number = offset + 1 if case["shape"] == "join" else local_index(source, offset)
                entity = resource(source, case["prefix"], number)
                subjects[key][source].append(str(entity))
                if key == "topics":
                    cls = {"cmt": "SubjectArea", "ekaw": "Research_Topic"}.get(source, "Topic")
                    typed(g, source, entity, ontology[cls])
                elif key == "early_registration":
                    if source == "conference":
                        typed(g, source, entity, ontology.Conference_participant)
                        typed(g, source, entity, ontology.Early_paid_applicant)
                    elif source == "confof":
                        typed(g, source, entity, ontology.Participant)
                        g.add((entity, ontology.earlyRegistration, Literal(True)))
                    else:
                        typed(g, source, entity, ontology["Early-Registered_Participant"])
                elif key == "dinners":
                    cls = {"confof": "Banquet", "edas": "ConferenceDinner", "ekaw": "Conference_Banquet"}[source]
                    typed(g, source, entity, ontology[cls])
                    if source == "ekaw":
                        organiser = resource(source, "dinner-organiser", number)
                        typed(g, source, organiser, ontology.Organisation)
                        g.add((entity, ontology.organisedBy, organiser))
                        g.add((organiser, ontology.organises, entity))
                else:
                    typed(g, source, entity, ontology.Conference_volume if source == "conference" else ontology.Conference)
                    day = (date(2027, 1, 1) + timedelta(days=offset)).isoformat()
                    if key == "start_dates":
                        if source == "conference":
                            dates = resource(source, "important-dates-start", number)
                            typed(g, source, dates, ontology.Important_dates)
                            g.add((entity, ontology.has_important_dates, dates))
                            g.add((dates, ontology.belong_to_a_conference_volume, entity))
                            g.add((dates, ontology.is_a_starting_date, Literal(day, datatype=XSD.date)))
                        elif source == "confof":
                            g.add((entity, ontology.starts_on, Literal(day)))
                        else:
                            g.add((entity, ontology.startDate, Literal(day + "T09:00:00", datatype=XSD.dateTime)))
                    elif key == "locations":
                        other = "edas" if source == "conference" else "conference"
                        g.add((entity, OWL.sameAs, resource(other, case["prefix"], number)))
                        if source == "conference":
                            g.add((entity, ontology.has_a_location, Literal(f"Augmented location {number:03d}")))
                        else:
                            g.add((entity, ontology.registrationDueOn, Literal(day + "T17:00:00", datatype=XSD.dateTime)))
        for source in ALL_SOURCES:
            graphs[source] += cases[key][source]
    return graphs, cases, subjects


def local_result(query, graphs):
    dataset = Dataset()
    for source, graph in graphs.items():
        target = dataset.graph(URIRef(f"{BASE}/{source}/sparql"))
        for triple in graph:
            target.add(triple)
    local_query = SERVICE.sub(r"GRAPH \1", query)
    result = dataset.query(local_query)
    return json.loads(result.serialize(format="json"))


def validate_local(plan, graphs, subjects):
    for key, case in plan["cases"].items():
        result = local_result(case["reference"], graphs)
        normalized = rows_for({"bindings": result["results"]["bindings"]}, result["head"]["vars"], same_id_key)
        expected = case["count"] * (1 if case["shape"] == "join" else len(case["sources"]))
        if len(normalized) != expected:
            raise ValueError(f"{key}: reference returns {len(normalized)} fixture rows, expected {expected}")
        for source in case["sources"]:
            omitted = local_result(case["reference"], {s: g for s, g in graphs.items() if s != source})
            remaining = rows_for({"bindings": omitted["results"]["bindings"]}, omitted["head"]["vars"], same_id_key)
            loss = expected if case["shape"] == "join" else case["count"]
            if len(normalized - remaining) != loss:
                raise ValueError(f"{key}: incorrect answer loss without {source}")
    return True


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        plan = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        plan = discover()
    for case in plan["cases"].values():
        path = ROOT / case["path"] / "reference.rq"
        if digest(path.read_text(encoding="utf-8-sig")) != case["reference_sha256"]:
            raise ValueError(f"Reference changed after fixture planning: {path}")
    graphs, cohorts, subjects = build_fixture(plan)
    validate_local(plan, graphs, subjects)
    for source, graph in graphs.items():
        body = triples(graph)
        (output / f"{source}.ttl").write_text(body + "\n", encoding="utf-8")
        for operation in ("INSERT", "DELETE"):
            (output / f"{source}.{operation.lower()}.ru").write_text(f"{operation} DATA {{\n{body}\n}}\n", encoding="utf-8")
    for key, case in plan["cases"].items():
        (output / f"{key}.reference.rq").write_text(case["reference"], encoding="utf-8")
    write_json(output / "subjects.json", subjects)
    write_json(manifest, plan)
    return plan, graphs, subjects


def inspect(client, graphs):
    uris = sorted({str(s) for graph in graphs.values() for s in graph.subjects()})
    found = {}
    for source in ALL_SOURCES:
        graph = Graph()
        # Bounded VALUES payloads, including incoming links from any subject.
        for offset in range(0, len(uris), 400):
            graph += client.related_triples(source, uris[offset:offset + 400])
        if set(graph) - set(graphs[source]):
            raise ValueError(f"{source}: IDs have unexpected data or identity links; refusing mutation.")
        found[source] = graph
    return found


def execute(client, query):
    endpoints = source_endpoints(query)
    allowed = {f"{BASE}/{s}/sparql" for s in ALL_SOURCES}
    if not endpoints or not endpoints <= allowed:
        raise ValueError("Unexpected SERVICE endpoint in evaluation query")
    source = sorted(endpoints)[0].split("/")[-2]
    # Keep result variable order including queries returning zero rows.
    response = requests.post(f"{BASE}/{source}/sparql", data={"query": query},
        headers={"Accept": "application/sparql-results+json"}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    return {"variables": payload["head"]["vars"], "bindings": payload["results"]["bindings"]}


def measure(client, plan, output, label):
    report = {"fixture": VERSION, "measured_at": datetime.now(timezone.utc).isoformat(), "cases": {}}
    for key, case in plan["cases"].items():
        result = execute(client, case["reference"])
        reference_count = len(rows_for(result, result["variables"], same_id_key))
        entry = {"reference_raw_rows": len(result["bindings"]), "reference_normalized_rows": reference_count,
                 "reference_sha256": case["reference_sha256"], "runs": {}}
        write_json(output / f"{key}.{label}.reference-results.json", result)
        for run in case["qualifying_runs"]:
            query = (ROOT / case["path"] / f"run_{run}.rq").read_text(encoding="utf-8-sig")
            generated = execute(client, query)
            entry["runs"][str(run)] = {"query_sha256": digest(query),
                "source_endpoints": sorted(source_endpoints(query)),
                "answer": answer_scores(result, generated, same_id_key)}
        report["cases"][key] = entry
        print(f"{label} {key}: {reference_count} reference answers; " + ", ".join(
            f"run {run} F1={v['answer']['f1']:.4f}" for run, v in entry["runs"].items()), flush=True)
    write_json(output / f"{label}.json", report)
    return report


def verify(client, plan, graphs, subjects, output):
    found = inspect(client, graphs)
    if any(set(found[s]) != set(graphs[s]) for s in ALL_SOURCES):
        raise ValueError("Fixture is incomplete")
    report = measure(client, plan, output, "after")
    for key, case in plan["cases"].items():
        result = execute(client, case["reference"])
        expected = local_result(case["reference"], graphs)
        expected_rows = rows_for({"bindings": expected["results"]["bindings"]}, expected["head"]["vars"], same_id_key)
        actual_rows = rows_for(result, result["variables"], same_id_key)
        if not expected_rows <= actual_rows:
            raise ValueError(f"{key}: synthetic reference answers missing on server")
        print(f"Verified {key}: all {len(expected_rows)} augmented answer tuples are present", flush=True)
    write_json(output / "verified.json", {"fixture": VERSION,
        "verified_at": datetime.now(timezone.utc).isoformat(), "triples": {s: len(g) for s, g in found.items()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "verify", "remove"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--case", choices=tuple(CASES),
                        help="With remove, delete only this question's synthetic cohort")
    args = parser.parse_args()
    if args.case and args.action != "remove":
        parser.error("--case is supported only with the remove action")
    plan, graphs, subjects = prepare(args.output_dir)
    _, cohorts, _ = build_fixture(plan)
    print("Prepared cases: " + ", ".join(f"{key}={c['count']}" for key, c in plan["cases"].items()), flush=True)
    if args.action == "prepare":
        return
    client = Fuseki(BASE)
    if args.action == "verify":
        verify(client, plan, graphs, subjects, args.output_dir)
        return
    if args.action == "remove" and args.case:
        graphs = cohorts[args.case]
    found = inspect(client, graphs)
    if args.action == "apply":
        if not (args.output_dir / "before.json").exists():
            measure(client, plan, args.output_dir, "before")
        for source in ALL_SOURCES:
            missing = graphs[source] - found[source]
            batch = Graph()
            for triple in sorted(missing, key=str):
                batch.add(triple)
                if len(batch) == 1000:
                    client.update(source, "INSERT", batch)
                    batch = Graph()
            if batch:
                client.update(source, "INSERT", batch)
            print(f"{source}: inserted {len(missing)} triples", flush=True)
        verify(client, plan, graphs, subjects, args.output_dir)
    else:
        for source in ALL_SOURCES:
            if found[source]:
                client.update(source, "DELETE", found[source])
            print(f"{source}: removed {len(found[source])} exact fixture triples", flush=True)
        if any(inspect(client, graphs).values()):
            raise ValueError("Fixture removal incomplete")


if __name__ == "__main__":
    main()
