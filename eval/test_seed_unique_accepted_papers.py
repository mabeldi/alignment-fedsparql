"""Offline checks for source-exclusive accepted-paper evaluation data."""

import unittest

from rdflib import Graph, Namespace, OWL, URIRef
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate

from seed_unique_accepted_papers import (
    ALL_SOURCES, PATTERNS, SOURCES, inspect_fixture, make_fixture,
    normalized, reference_query, triples,
)


class AcceptedPaperFixtureTests(unittest.TestCase):
    def setUp(self):
        self.graphs, self.papers = make_fixture(20)

    def test_every_source_returns_twenty_distinct_answers_after_normalization(self):
        all_answers = set()
        for source, graph in self.graphs.items():
            rows = graph.query(f"SELECT DISTINCT ?paper WHERE {{ {PATTERNS[source]} }}")
            bindings = [{"paper": {"type": "uri", "value": str(row.paper)}} for row in rows]
            answers = normalized(bindings)
            self.assertEqual(len(answers), 20)
            self.assertFalse(answers & all_answers)
            all_answers |= answers
        self.assertEqual(len(all_answers), 80)

    def test_cmt_alternative_acceptance_patterns_agree(self):
        graph = self.graphs["cmt"]
        for pattern in (PATTERNS["cmt"], "?paper <http://cmt#acceptedBy> ?admin .",
                        "?admin <http://cmt#acceptPaper> ?paper ."):
            actual = {str(row.paper) for row in graph.query(f"SELECT ?paper WHERE {{ {pattern} }}")}
            self.assertEqual(actual, set(self.papers["cmt"]))
        # acceptPaper is functional: use a separate administrator per paper.
        cmt = Namespace("http://cmt#")
        for administrator in graph.subjects(cmt.acceptPaper):
            self.assertEqual(len(list(graph.objects(administrator, cmt.acceptPaper))), 1)

    def test_growth_to_500_preserves_old_data_and_unique_answers(self):
        expanded, papers = make_fixture(500)
        all_answers = set()
        for source in SOURCES:
            self.assertTrue(set(self.graphs[source]) <= set(expanded[source]))
            self.assertTrue(set(self.papers[source]) <= set(papers[source]))
            self.assertEqual(len(papers[source]), 500)
            self.assertEqual(len(set(papers[source]) - set(self.papers[source])), 480)
            rows = expanded[source].query(f"SELECT DISTINCT ?paper WHERE {{ {PATTERNS[source]} }}")
            answers = normalized([{"paper": {"type": "uri", "value": str(r.paper)}} for r in rows])
            self.assertEqual(len(answers), 500)
            self.assertFalse(answers & all_answers)
            all_answers |= answers
        self.assertEqual(len(all_answers), 2000)

    def test_fixture_uses_existing_namespaces_and_no_identity_links(self):
        subjects = set()
        for source, graph in self.graphs.items():
            mine = set(graph.subjects())
            self.assertFalse(subjects & mine)
            self.assertTrue(all(str(s).startswith(f"http://{source}-instances#") for s in mine))
            self.assertTrue(all(str(s).endswith("-aug") for s in mine))
            self.assertFalse(list(graph.triples((None, OWL.sameAs, None))))
            subjects |= mine

    def test_reference_and_exact_updates_parse(self):
        parseQuery(reference_query("http://localhost:3030"))
        for graph in self.graphs.values():
            for operation in ("INSERT", "DELETE"):
                parseUpdate(f"{operation} DATA {{ {triples(graph)} }}")

    def test_collision_in_other_ontology_is_rejected_before_mutation(self):
        graphs = {**self.graphs, "confof": Graph()}
        graphs["confof"].add((URIRef("http://confOf-instances#existing"), OWL.sameAs,
                              URIRef(self.papers["cmt"][0])))

        class FakeClient:
            def related_triples(self, source, subjects):
                return graphs[source]

        with self.assertRaisesRegex(ValueError, "confof.*unexpected"):
            inspect_fixture(FakeClient(), self.graphs)


if __name__ == "__main__":
    unittest.main()
