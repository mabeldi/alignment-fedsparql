"""Offline coverage for the schema-alignment ablation (no LLM/Fuseki calls)."""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import pipeline_config
from tools import alignment_store
from tools.alignment_store import AlignmentStore, Mapping
from tools.path_sparql import VarNamer
from tools.prompt_format import _answer_sources
from agents.federation_planner import (
    expand_matches_with_alignment, format_alignment_block,
    participating_sources, plan_federation,
)


DINNER = "http://edas#ConferenceDinner"
BANQUET = "http://ekaw#Conference_Banquet"
ACCEPTED = "http://edas#AcceptedPaper"
PAPER = "http://cmt#Paper"
SOURCES = {"edas", "ekaw", "cmt", "confof"}


class AlignmentModeTests(unittest.TestCase):
    def setUp(self):
        store = AlignmentStore()
        store.mappings = [
            Mapping(DINNER, "simple", [BANQUET]),
            Mapping(ACCEPTED, "restriction_type", [PAPER],
                    {"property": "http://cmt#hasDecision", "value": "http://cmt#Acceptance"}),
            Mapping("http://confOf#OtherEvent", "simple", ["http://ekaw#Workshop"]),
        ]
        store._build_index()
        self.store = store
        self.cached = patch.object(alignment_store, "_STORE", store)
        self.cached.start()
        self.addCleanup(self.cached.stop)

    def test_disabled_store_neither_loads_nor_exposes_cached_mappings(self):
        with patch.object(pipeline_config, "_ALIGNMENTS_ENABLED", False), \
                patch.object(AlignmentStore, "load", side_effect=AssertionError("must not load")):
            for cached in (None, self.store):
                with patch.object(alignment_store, "_STORE", cached):
                    store = alignment_store.get_alignment_store()
                    self.assertEqual(store.stats()["total"], 0)
                    self.assertEqual(store.get(DINNER), [])
                    self.assertEqual(store.counterparts(DINNER, SOURCES), [])
                    self.assertEqual(store.find_for_classes({ACCEPTED}, SOURCES), [])
        with patch.object(pipeline_config, "_ALIGNMENTS_ENABLED", True):
            self.assertIs(alignment_store.get_alignment_store(), self.store)

    def test_expansion_planning_and_mapping_only_sources_follow_switch(self):
        matches = {"Dinner": [{"payload": {"type": "Class", "uri": DINNER,
                                             "source": "edas"}}]}
        blocks = [{"source": "edas", "class": ACCEPTED}]
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(pipeline_config, "_ALIGNMENTS_ENABLED", enabled):
                expanded = expand_matches_with_alignment(deepcopy(matches), SOURCES)
                self.assertEqual(len(expanded["Dinner"]), 2 if enabled else 1)
                mappings = plan_federation(blocks, SOURCES)
                self.assertEqual(bool(mappings), enabled)
                participants = participating_sources(blocks, mappings)
                self.assertEqual(participants, {"edas", "cmt"} if enabled else {"edas"})
                rendered = format_alignment_block(mappings)
                self.assertEqual("hasDecision" in rendered, enabled)
                self.assertEqual("Acceptance" in rendered, enabled)

    def test_implicit_variable_naming_and_answer_filtering_follow_switch(self):
        answers = {"edas": {DINNER}, "ekaw": {BANQUET},
                   "confof": {"http://confOf#OtherEvent"}}
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(pipeline_config, "_ALIGNMENTS_ENABLED", enabled):
                namer = VarNamer(SOURCES)
                self.assertEqual(namer.var_for(DINNER) == namer.var_for(BANQUET), enabled)
                kept = _answer_sources(list(answers), answers, "edas")
                self.assertEqual("confof" in kept, not enabled)

    def test_agent7_gets_no_mapping_payload_but_retains_instance_link_convention(self):
        # Only isolate the LLM transport import; use the real prompt builder,
        # graph renderer, mapping planner, and SPARQL guidance.
        router = ModuleType("tools.llm_router")
        router.run_agent_async = lambda *args, **kwargs: None
        module_name = "agents.agent7_sparql_query_generator"
        with patch.dict(sys.modules, {"tools.llm_router": router}):
            generator = importlib.import_module(module_name)
        self.addCleanup(sys.modules.pop, module_name, None)
        blocks = [{"source": "edas", "class": ACCEPTED, "label": "AcceptedPaper",
                   "outgoing": [], "incoming": [], "answers_question": True}]
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(pipeline_config, "_ALIGNMENTS_ENABLED", enabled):
                mappings = plan_federation(blocks, SOURCES)
                messages = generator.build_messages(
                    "List accepted papers.", blocks, "edas", format_alignment_block(mappings),
                    participating_sources(blocks, mappings), {"classes": ["Accepted Paper"]})
                prompt = "\n".join(message["content"] for message in messages)
                self.assertEqual("hasDecision" in prompt, enabled)
                self.assertEqual("cmt:Acceptance" in prompt, enabled)
                self.assertIn("http://www.w3.org/2002/07/owl#sameAs", prompt)

    def test_config_requires_an_explicit_boolean(self):
        for text, expected in [("alignments_enabled: true", True),
                               ("alignments_enabled: false", False)]:
            with patch.object(Path, "read_text", return_value=text):
                self.assertIs(pipeline_config._read_alignment_setting(), expected)
        for text in ["", "{}", "alignments_enabled: 'false'", "alignments_enabled: 0"]:
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text):
                with self.assertRaises(ValueError):
                    pipeline_config._read_alignment_setting()


if __name__ == "__main__":
    unittest.main()
