"""Validate reference compatibility and missing-source sensitivity offline."""

import unittest

from rdflib import Literal, OWL, XSD

from seed_source_coverage import (
    ALL_SOURCES, CASES, RESULTS, build_fixture, local_result, ns,
    rows_for, same_id_key, select_count, validate_local,
)


class SourceCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = {"cases": {key: {**definition, "count": 3,
            "reference": (RESULTS / definition["folder"] / "reference.rq").read_text(encoding="utf-8-sig")}
            for key, definition in CASES.items()}}
        cls.graphs, cls.cohorts, cls.subjects = build_fixture(cls.plan)

    def test_references_and_each_source_omission_have_expected_answers(self):
        self.assertTrue(validate_local(self.plan, self.graphs, self.subjects))

    def test_union_resources_are_unique_under_actual_evaluator(self):
        for key, case in self.plan["cases"].items():
            if case["shape"] != "union":
                continue
            seen = set()
            for source, uris in self.subjects[key].items():
                values = {same_id_key({"value": uri}) for uri in uris}
                self.assertFalse(seen & values)
                seen |= values
                self.assertFalse(list(self.cohorts[key][source].triples((None, OWL.sameAs, None))))

    def test_location_answers_require_both_sources_and_no_confof_copy(self):
        cohort = self.cohorts["locations"]
        self.assertEqual(len(cohort["confof"]), 0)
        self.assertEqual(len(list(cohort["conference"].triples((None, OWL.sameAs, None)))), 3)
        self.assertEqual(len(list(cohort["edas"].triples((None, OWL.sameAs, None)))), 3)
        query = self.plan["cases"]["locations"]["reference"]
        result = local_result(query, cohort)
        self.assertEqual(len(result["results"]["bindings"]), 3)
        for source in ("conference", "edas"):
            result = local_result(query, {s: g for s, g in cohort.items() if s != source})
            self.assertEqual(result["results"]["bindings"], [])

    def test_early_registration_uses_schema_boolean(self):
        values = list(self.cohorts["early_registration"]["confof"].objects(None, ns("confof").earlyRegistration))
        self.assertEqual(len(values), 3)
        self.assertTrue(all(v == Literal(True) and v.datatype == XSD.boolean for v in values))

    def test_counts_give_at_least_ten_percent_loss_for_one_missing_source(self):
        for baseline, sources, expected in [(5, 3, 20), (2155, 3, 400), (3563, 5, 800)]:
            count = select_count(baseline, sources)
            self.assertEqual(count, expected)
            self.assertGreaterEqual(count / (baseline + sources * count), 0.10)


if __name__ == "__main__":
    unittest.main()
