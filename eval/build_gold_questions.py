"""Compile the gold question set (eval/questions/gold_questions.json) from the
published IRIT CQA files.

WHY A SCRIPT AND NOT A HAND-WRITTEN JSON
----------------------------------------
Each entry has to carry the gold federated query AND the gold triple patterns
per source. Written by hand those two drift apart the first time a question is
edited, and the metrics read the patterns while a human reads the query — so the
drift is invisible until the numbers are already wrong. Here the patterns are
derived from the same text the query is built from, by parsing it, so they
cannot disagree.

The per-source SPARQL is never authored here. It is lifted verbatim from
eval/cqa_source/CQAs/<cqa>/<onto>.sparql, which is the published dataset, so the
vocabulary in a gold query is the dataset's choice and not ours. The only things
authored in this file are: the natural-language wording, which sources a
question is expected to reach, how the blocks combine (UNION vs join), and the
variable names.

A handful of entries have no CQA behind them (the distractors, and two
correspondences the CQAs never exercise). Those carry `body` written directly
against the ontology, and say so in `notes`.

Run:  python eval/build_gold_questions.py
      python eval/build_gold_questions.py --check   # compile, print, write nothing

An existing gold_questions.json whose content would change is copied to
gold_questions.json.bak first — the file is edited by hand often enough that
silently replacing it would lose work.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from rdflib.plugins.sparql import prepareQuery

ROOT = Path(__file__).resolve().parent.parent
CQA_DIR = ROOT / "eval" / "cqa_source" / "CQAs"
OUT = ROOT / "eval" / "questions" / "gold_questions.json"

FUSEKI_BASE = "http://192.168.100.45:3030"
ENDPOINT = {s: f"{FUSEKI_BASE}/{s}/sparql"
            for s in ("cmt", "confof", "ekaw", "conference", "edas")}

# Our source names are lower-case (they come from the URI host); the CQA folders
# keep the ontology's own casing. Only confOf differs.
CQA_FILENAME = {"cmt": "cmt", "confof": "confOf", "conference": "conference",
                "edas": "edas", "ekaw": "ekaw"}

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# The link that carries a cross-source join. Each endpoint stores, for its own
# instances, the identifiers the other sources use for the same thing (loaded by
# eval/load_dataset.py next to the instance data).
SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"


# --- reading a CQA ----------------------------------------------------------

def _strip_prefixes(text: str):
    """Return (body-ready text, {prefix: namespace}) with the PREFIX lines gone."""
    prefixes = {}
    for m in re.finditer(r"PREFIX\s+([A-Za-z][\w.-]*)?\s*:\s*<([^>]+)>", text):
        prefixes[m.group(1) or ""] = m.group(2)
    return re.sub(r"PREFIX\s+([A-Za-z][\w.-]*)?\s*:\s*<[^>]+>", " ", text), prefixes


def _expand_curies(text: str, prefixes: dict) -> str:
    """Rewrite `pre:Local` as `<namespace + Local>`, leaving <...> alone.

    Every gold body ends up prefix-free, so a SERVICE block can be moved between
    queries without dragging a PREFIX header along with it.
    """
    out, pos = [], 0
    for m in re.finditer(r"<[^>]*>|\"[^\"]*\"", text):     # skip URIs and literals
        out.append(_expand_segment(text[pos:m.start()], prefixes))
        out.append(m.group(0))
        pos = m.end()
    out.append(_expand_segment(text[pos:], prefixes))
    return "".join(out)


def _expand_segment(seg: str, prefixes: dict) -> str:
    def sub(m):
        pre, local = m.group(1) or "", m.group(2)
        ns = prefixes.get(pre)
        return f"<{ns}{local}>" if ns else m.group(0)
    # `?x`, `^^`, and bare keywords must not match: require a name after the colon
    # and nothing word-like before it. The local name may not contain a dot —
    # SPARQL allows it, but no term in these ontologies uses one and permitting it
    # swallows the `.` that ends the triple (`:Submission_event.`).
    return re.sub(r"(?<![?$\w])([A-Za-z][\w-]*)?:([A-Za-z_][\w-]*)", sub, seg)


def _where_body(text: str) -> str:
    """The contents of the outermost WHERE { ... }, braces excluded."""
    i = text.upper().find("WHERE")
    start = text.find("{", i if i >= 0 else 0)
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    raise ValueError("unbalanced braces in CQA body")


def _rename_vars(body: str, mapping: dict, fallback_prefix: str) -> str:
    """Apply the spec's variable names; give every other variable a per-block name.

    The blocks of a joined query share one variable scope, so two SERVICE blocks
    that both happen to use the CQA's `?o2` would be joined on it by accident —
    a join nobody wrote and nobody sees. Prefixing the leftovers removes the
    possibility instead of relying on the CQAs not to collide.
    """
    def sub(m):
        name = m.group(1)
        return "?" + mapping.get(name, f"{fallback_prefix}_{name}")
    return re.sub(r"[?$](\w+)", sub, body)


def load_block(cqa: str, source: str, rename: dict, var_prefix: str) -> str:
    path = CQA_DIR / cqa / f"{CQA_FILENAME[source]}.sparql"
    if not path.exists():
        raise SystemExit(f"no CQA file: {path}")
    text, prefixes = _strip_prefixes(path.read_text(encoding="utf-8"))
    body = _where_body(text)
    body = _expand_curies(body, prefixes)
    body = _rename_vars(body, rename, var_prefix)
    return " ".join(body.split())


# --- reading a block back as data -------------------------------------------

def _walk(node, hit):
    if not hasattr(node, "name"):
        return
    hit(node)
    for value in node.values():
        if hasattr(value, "name"):
            _walk(value, hit)
        elif isinstance(value, list):
            for item in value:
                if hasattr(item, "name"):
                    _walk(item, hit)


# Algebra nodes every SELECT has; anything else is a real feature of the pattern
# and is worth recording, because a flat triple-pattern set does not show it.
_STRUCTURAL = {"SelectQuery", "Project", "BGP", "ToMultiSet", "values", "Join"}


def analyse(body: str):
    """(triple patterns, operator names) of a block, by parsing it."""
    query = prepareQuery(f"SELECT * WHERE {{ {body} }}")
    triples, ops = [], set()

    def visit(node):
        ops.add(node.name)
        if node.name == "BGP":
            triples.extend(node.triples)

    _walk(query.algebra, visit)
    patterns = [[t.n3() for t in triple] for triple in triples]
    return patterns, sorted(ops - _STRUCTURAL)


def vocabulary(patterns):
    """Split a block's URIs into the classes and properties it commits to."""
    classes, properties = set(), set()
    for s, p, o in patterns:
        if p == f"<{RDF_TYPE}>":
            if o.startswith("<"):
                classes.add(o[1:-1])
        elif p.startswith("<"):
            properties.add(p[1:-1])
    return sorted(classes), sorted(properties)


# --- assembling the gold query ----------------------------------------------

def bridge_block(body: str, join_var: str, source: str) -> str:
    """Rewrite a non-anchor block so it joins the anchor through the link.

    Two endpoints never share an identifier: the same paper is named one way in
    one source and another way in the next. So repeating the anchor's variable
    here would ask for a URI stored by both, which none is — the query runs, scans
    both sides in full, and returns nothing. (It also costs enough memory to take
    the server down, which is how this was found.)

    Instead the entity gets a name of its own in this block, and the block opens
    by looking up which of its own instances the anchor's URI refers to. That
    lookup runs INSIDE this endpoint, against the links it holds about its own
    instances, with the anchor's variable already bound — so it matches one thing
    rather than fanning out over every source.
    """
    local = f"{join_var}_{source}"
    body = re.sub(rf"[?$]{re.escape(join_var)}\b", f"?{local}", body)
    return f"?{local} <{SAME_AS}> ?{join_var} . {body}"


def build_query(select, blocks, combine: str) -> str:
    """blocks = [(source, body)]. UNION for the same concept in several sources,
    plain juxtaposition for complementary attributes joined through the link line
    that bridge_block() has already put in every non-anchor body."""
    services = [f"  SERVICE <{ENDPOINT[src]}> {{ {body} }}" for src, body in blocks]
    if combine == "union":
        inner = "\n  UNION\n".join(f"  {{\n  {s.strip()}\n  }}" for s in services)
    else:
        inner = "\n".join(services)
    projection = " ".join(f"?{v}" for v in select)
    return f"SELECT DISTINCT {projection} WHERE {{\n{inner}\n}}"


# --- the questions ----------------------------------------------------------
#
# `parts` is [(source, cqa, {cqa_var: gold_var})] or, where no CQA covers the
# concept, [(source, None, body)]. A union question repeats one CQA across its
# sources; a join question takes one part from each of two CQAs.

def one_cqa(cqa, sources, rename):
    return [{"source": s, "cqa": cqa, "rename": rename} for s in sources]


# NO QUESTION MAY NAME AN ONTOLOGY.
#
# The system is meant to look like one system. A user asks a question; which
# ontologies answer it, and how they are stitched together, is the system's job
# and is invisible from the outside. A question worded "In the CMT system, what
# is the name of each person?" hands over the answer to the hardest part of the
# problem — choosing the sources — and then measures whether the pipeline can
# read it back. That measures nothing.
#
# A category of single-source questions was removed for exactly this reason: the
# only thing making them single-source was the ontology name in the wording.
# Nothing was lost. The distractor questions below already answer from one
# source each, covering all five ontologies, and they do it honestly — the
# source has to be worked out from the words of the question.

QUESTIONS = [

    # 1 ── multi-source coverage ─────────────────────────────────────────────
    # One concept, a different name in every source. Each source holds its own
    # instances, so the answer is the UNION over all of them.
    {
        "id": "MC-01", "category": "multi-source",
        "question": "List all the papers.",
        "select": ["paper"], "combine": "union",
        "parts": one_cqa("paper", ["cmt", "confof", "conference", "edas", "ekaw"], {"s": "paper"}),
    },
    {
        "id": "MC-02", "category": "multi-source",
        "question": "Who are the authors?",
        "select": ["person"], "combine": "union",
        "parts": one_cqa("author", ["cmt", "confof", "conference", "edas", "ekaw"], {"s": "person"}),
    },
    {
        "id": "MC-03", "category": "multi-source",
        "question": "Which research areas are listed?",
        "select": ["topic"], "combine": "union",
        "parts": one_cqa("topic", ["cmt", "confof", "conference", "edas", "ekaw"], {"s": "topic"}),
    },
    {
        "id": "MC-04", "category": "multi-source",
        "question": "Which conferences are described?",
        "select": ["conf"], "combine": "union",
        "parts": one_cqa("conference", ["cmt", "confof", "conference", "edas", "ekaw"], {"s": "conf"}),
    },
    {
        "id": "MC-05", "category": "multi-source",
        "question": "Which papers are there, and what is the title of each one?",
        "select": ["paper", "title"], "combine": "union",
        # cmt and confof only. The CQA also covers conference, edas and ekaw,
        # but none of those three declares a title property — their branches
        # read the title off rdfs:label, which no .owl file declares, so no
        # schema-driven system can write them. See eval/questions/benchmark_audit.md.
        "parts": one_cqa("paper_title", ["cmt", "confof"], {"s": "paper", "o": "title"}),
    },
    {
        "id": "MC-06", "category": "multi-source",
        "question": "Who took part in the event?",
        "select": ["person"], "combine": "union",
        "parts": [
            *one_cqa("participant", ["confof", "conference", "edas"], {"s": "person"}),
            {"source": "ekaw", "body": "?person a <http://ekaw#Conference_Participant> ."},
        ],
        "notes": ("the ekaw CQA answers with Conference_Participant AND (Early- or "
                  "Late-Registered), which wrongly excludes participants never "
                  "tagged either way (e.g. chairs/presenters attending); narrowed "
                  "to the base participant type here."),
    },
    {
        "id": "MC-07", "category": "multi-source",
        "question": "Who chaired the programme committee?",
        "select": ["person"], "combine": "union",
        "parts": one_cqa("pc_chair", ["cmt", "confof", "conference", "ekaw"], {"s": "person"}),
    },
    {
        "id": "MC-08", "category": "multi-source",
        "question": "Which outings are organised for the attendees?",
        "select": ["outing"], "combine": "union",
        "parts": one_cqa("trip", ["confof", "edas", "ekaw"], {"s": "outing"}),
    },
    {
        "id": "MC-09", "category": "multi-source",
        "question": "Which formal dinners are on the schedule?",
        "select": ["dinner"], "combine": "union",
        "parts": one_cqa("gala_dinner", ["confof", "edas", "ekaw"], {"s": "dinner"}),
    },
    {
        "id": "MC-10", "category": "multi-source",
        "question": "Which higher-education institutions appear?",
        "select": ["org"], "combine": "union",
        "parts": one_cqa("university", ["confof", "edas", "ekaw"], {"s": "org"}),
    },
    {
        "id": "MC-11", "category": "multi-source",
        "question": "Which of the people are still studying?",
        "select": ["person"], "combine": "union",
        "parts": one_cqa("student", ["confof", "ekaw"], {"s": "person"}),
    },
    {
        "id": "MC-12", "category": "multi-source",
        "question": "Which businesses appear?",
        "select": ["org"], "combine": "union",
        "parts": one_cqa("company", ["confof", "edas"], {"s": "org"}),
    },

    # 2 ── cross-source join ─────────────────────────────────────────────────
    # Two CQAs composed on a shared entity, and the pair of sources is chosen so
    # that NEITHER source can supply both halves. Collapsing to one endpoint
    # therefore loses an answer column, not just some instances.
    {
        "id": "CJ-01", "category": "cross-source-join",
        "question": "Which papers are there, which summary belongs to each one, and what subjects does it cover?",
        "select": ["paper", "abstract", "topic"], "combine": "join", "join_on": "paper",
        "parts": [
            {"source": "conference", "cqa": "abstract_of_paper", "rename": {"s": "paper", "o": "abstract"}},
            {"source": "edas", "cqa": "topic_of_paper", "rename": {"s": "paper", "o": "topic"}},
        ],
    },
    {
        "id": "CJ-02", "category": "cross-source-join",
        "question": "Which papers are there, which summary belongs to each one, and which conference was it sent to?",
        "select": ["paper", "abstract", "conf"], "combine": "join", "join_on": "paper",
        "parts": [
            {"source": "cmt", "cqa": "abstract_of_paper", "rename": {"s": "paper", "o": "abstract"}},
            {"source": "edas", "cqa": "paper_submitted_at_conference", "rename": {"s": "paper", "o": "conf"}},
        ],
    },
    {
        "id": "CJ-03", "category": "cross-source-join",
        "question": "Which reviews were written, which paper does each one review, and to which conference was that paper submitted?",
        "select": ["review", "paper", "conf"], "combine": "join", "join_on": "paper",
        "parts": [
            {"source": "ekaw", "cqa": "review_of_paper", "rename": {"s": "review", "o": "paper"}},
            {"source": "edas", "cqa": "paper_submitted_at_conference", "rename": {"s": "paper", "o": "conf"}},
        ],
    },
    {
        "id": "CJ-04", "category": "cross-source-join",
        "question": "Which papers are there, who is the lead writer of each, and what subjects does it cover?",
        "select": ["author", "paper", "topic"], "combine": "join", "join_on": "paper",
        "parts": [
            {"source": "cmt", "cqa": "paper_1st_author", "rename": {"s": "author", "o2": "paper"}},
            {"source": "confof", "cqa": "topic_of_paper", "rename": {"s": "paper", "o": "topic"}},
        ],
    },
    {
        "id": "CJ-05", "category": "cross-source-join",
        "question": "Which people ran a workshop, what is each one's surname, and which workshop did they run?",
        "select": ["person", "surname", "workshop"], "combine": "join", "join_on": "person",
        "parts": [
            {"source": "edas", "cqa": "person_last_name", "rename": {"s": "person", "o": "surname"}},
            {"source": "ekaw", "cqa": "chair_of_workshop", "rename": {"s": "person", "o": "workshop"}},
        ],
    },
    {
        "id": "CJ-06", "category": "cross-source-join",
        "question": "Which administrative staff belong to a conference, and to which one?",
        "select": ["person", "conf"], "combine": "join", "join_on": "person",
        "parts": [
            {"source": "confof", "cqa": "administrator", "rename": {"s": "person"}},
            {"source": "edas", "cqa": "member_of_conference", "rename": {"s": "person", "o": "conf"}},
        ],
    },
    {
        "id": "CJ-07", "category": "cross-source-join",
        "question": "Which people are still studying, and what is each one's given name?",
        "select": ["person", "firstname"], "combine": "join", "join_on": "person",
        "parts": [
            {"source": "ekaw", "cqa": "student", "rename": {"s": "person"}},
            {"source": "confof", "cqa": "person_first_name", "rename": {"s": "person", "o": "firstname"}},
        ],
    },
    {
        "id": "CJ-08", "category": "cross-source-join",
        "question": "Which people brought in extra reviewers, whom did they bring in, and what is the inviter's given name?",
        "select": ["person", "firstname", "invitee"], "combine": "join", "join_on": "person",
        "parts": [
            {"source": "cmt", "cqa": "invite_reviewer", "rename": {"s": "person", "o": "invitee"}},
            {"source": "confof", "cqa": "person_first_name", "rename": {"s": "person", "o": "firstname"}},
        ],
    },
    {
        "id": "CJ-09", "category": "cross-source-join",
        "question": "Which people sit on the organising committee, and what is each member's surname?",
        "select": ["person", "surname"], "combine": "join", "join_on": "person",
        "parts": [
            {"source": "ekaw", "cqa": "oc_member", "rename": {"s": "person"}},
            {"source": "confof", "cqa": "person_last_name", "rename": {"s": "person", "o": "surname"}},
        ],
    },
    {
        "id": "CJ-10", "category": "cross-source-join",
        "question": "Which conferences are there, where is each one held, and by when must you register?",
        "select": ["conf", "place", "deadline"], "combine": "join", "join_on": "conf",
        "parts": [
            # The CQA ends on `?location rdfs:label ?place`, but ekaw:Location
            # declares no datatype property at all, so that label is the only
            # text on it and no .owl file declares it. Return the place itself,
            # as every other question returns the thing rather than its name.
            {"source": "ekaw",
             "body": "?conf a <http://ekaw#Conference> . ?conf <http://ekaw#heldIn> ?place ."},
            {"source": "edas", "cqa": "conference_regis_date", "rename": {"s": "conf", "o": "deadline"}},
        ],
    },
    {
        "id": "CJ-11", "category": "cross-source-join",
        "question": "Which conferences are there, what is the web address of each one, and on what day does it end?",
        "select": ["conf", "url", "enddate"], "combine": "join", "join_on": "conf",
        "parts": [
            {"source": "cmt", "cqa": "site_url_of_conference", "rename": {"s": "conf", "o": "url"}},
            {"source": "edas", "cqa": "conference_end_date", "rename": {"s": "conf", "o": "enddate"}},
        ],
    },
    {
        "id": "CJ-12", "category": "cross-source-join",
        "question": "Which workshops are there, what is each one about, and when does it begin?",
        "select": ["workshop", "topic", "startdate"], "combine": "join", "join_on": "workshop",
        "parts": [
            {"source": "conference", "cqa": "workshop_topic", "rename": {"s": "workshop", "o": "topic"}},
            {"source": "edas", "cqa": "workshop_start_date", "rename": {"s": "workshop", "o": "startdate"}},
        ],
    },
    {
        "id": "CJ-13", "category": "cross-source-join",
        "question": "Which workshops belong to a conference, which conference is it, and when is their submission deadline?",
        "select": ["workshop", "conf", "deadline"], "combine": "join", "join_on": "workshop",
        "parts": [
            {"source": "ekaw", "cqa": "workshop_of_conference", "rename": {"s": "workshop", "o": "conf"}},
            {"source": "confof", "cqa": "workshop_submission_date", "rename": {"s": "workshop", "o": "deadline"}},
        ],
    },
    {
        "id": "CJ-14", "category": "cross-source-join",
        "question": "Which papers were accepted, and what subject does each deal with?",
        "select": ["paper", "topic"], "combine": "join", "join_on": "paper",
        "parts": [
            {"source": "conference", "cqa": "accepted_paper", "rename": {"s": "paper"}},
            {"source": "confof", "cqa": "topic_of_paper", "rename": {"s": "paper", "o": "topic"}},
        ],
    },

    # 3 ── conditional correspondence ────────────────────────────────────────
    # The concept exists as a named class in one source and only as "class plus a
    # condition on a property" in another. Dropping the condition changes the
    # answer, so these are the entries where a plain equivalence is not enough.
    {
        "id": "CC-01", "category": "conditional",
        "question": "Which papers were accepted?",
        "select": ["paper"], "combine": "union",
        "parts": one_cqa("accepted_paper", ["cmt", "conference", "edas", "ekaw"], {"s": "paper"}),
        "exercises": [
            "ekaw:Accepted_Paper = cmt:Paper AND (cmt:hasDecision -> cmt:Acceptance)",
            "conference:Accepted_contribution = cmt:Paper AND (cmt:hasDecision -> cmt:Acceptance)",
        ],
    },
    {
        "id": "CC-02", "category": "conditional",
        "question": "Which papers were turned down?",
        "select": ["paper"], "combine": "union",
        "parts": one_cqa("rejected_paper", ["cmt", "conference", "edas", "ekaw"], {"s": "paper"}),
        "exercises": [
            "ekaw:Rejected_Paper = cmt:Paper AND (cmt:hasDecision -> cmt:Rejection)",
            "conference:Rejected_contribution = cmt:Paper AND (cmt:hasDecision -> cmt:Rejection)",
        ],
    },
    {
        "id": "CC-03", "category": "conditional",
        "question": "Who signed up early?",
        "select": ["person"], "combine": "union",
        "parts": one_cqa("early_registered_participant", ["confof", "conference", "ekaw"], {"s": "person"}),
        "exercises": [
            "ekaw:Early-Registered_Participant = confof:Participant AND (confof:earlyRegistration xsd:Equal true)",
        ],
    },
    {
        "id": "CC-04", "category": "conditional",
        "question": "Who signed up late?",
        "select": ["person"], "combine": "union",
        "parts": one_cqa("late_registered_participant", ["confof", "conference", "ekaw"], {"s": "person"}),
        "exercises": [],
        "notes": ("mirror of CC-03 with the boolean flipped. mapping.rdf has the "
                  "'early' cell only, so the 'late' correspondence is one of the "
                  "gaps eval/derived_mappings/ records."),
    },
    {
        "id": "CC-05", "category": "conditional",
        "question": "Which reviewers actually have papers to review?",
        "select": ["person"], "combine": "union",
        "parts": [
            {"source": "cmt",
             "body": "?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://cmt#Reviewer> . "
                     "?person <http://cmt#hasBeenAssigned> ?cmt_o2 ."},
            *one_cqa("assigned_reviewer", ["confof"], {"s": "person"}),
            {"source": "conference",
             "body": "?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://conference#Reviewer> . "
                     "?person <http://conference#was_a_member_of> ?conference_o2 . "
                     "?conference_o2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://conference#Program_committee> . "
                     "?person <http://conference#contributes> ?conference_o3 . "
                     "?conference_o3 <http://conference#reviews> ?conference_o4 ."},
            *one_cqa("assigned_reviewer", ["ekaw"], {"s": "person"}),
        ],
        "exercises": ["conference:Reviewer = confof:Member_PC AND (confof:reviewes exists)"],
        "notes": ("cmt and conference previously only checked the Reviewer role; "
                  "both now also require a real assignment (hasBeenAssigned / "
                  "contributes+reviews), mirroring CC-08's pattern, so 'actually "
                  "have papers to review' isn't satisfied by the role alone."),
    },
    {
        "id": "CC-06", "category": "conditional",
        "question": "Which talks are about a submitted paper?",
        "select": ["talk"], "combine": "union",
        "parts": one_cqa("paper_presentation", ["conference", "edas", "ekaw"], {"s": "talk"}),
        "exercises": ["conference:Presentation = edas:Paper AND (edas:isWrittenBy -> edas:Presenter)"],
        "notes": ("the edas CQA answers with edas:PaperPresentation, while the "
                  "correspondence phrases the same concept as edas:Paper plus a "
                  "condition. Both are in the gold: the CQA side is the gold "
                  "pattern, the correspondence is what has to fire to reach edas "
                  "from conference at all."),
    },
    {
        "id": "CC-07", "category": "conditional",
        "question": "Which conferences are there, and by when must the final version of a paper be handed in?",
        "select": ["conf", "date"], "combine": "union",
        "parts": one_cqa("conference_camera_date", ["confof", "conference", "edas"],
                         {"s": "conf", "o": "date"}),
        "exercises": ["confof:Camera_Ready_event = ekaw:Individual_Presentation AND "
                      "(ekaw:presentationOfPaper -> ekaw:Camera_Ready_Paper)"],
        "notes": ("the ekaw half of that correspondence has no CQA, so ekaw is NOT "
                  "in the gold source set. The correspondence is still the reason "
                  "confof reaches this question."),
    },
    {
        "id": "CC-08", "category": "conditional",
        "question": "Which papers already have a reviewer looking at them?",
        "select": ["paper"], "combine": "union",
        "parts": [
            {"source": "cmt", "cqa": "is_assigned_reviewer_of", "rename": {"s": "reviewer", "o": "paper"}},
            {"source": "confof", "cqa": "is_assigned_reviewer_of", "rename": {"s": "reviewer", "o": "paper"}},
            {"source": "conference", "cqa": "is_assigned_reviewer_of", "rename": {"s": "reviewer", "o": "paper"}},
            # No CQA asks this from the paper's side. Both blocks come from the
            # alignment cell itself, which is the point of the question.
            {"source": "ekaw", "body": "?paper a <http://ekaw#Assigned_Paper> ."},
            {"source": "edas", "body": "?paper a <http://edas#Paper> . "
                                       "?paper <http://edas#isReviewedBy> ?edas_reviewer ."},
        ],
        "exercises": ["ekaw:Assigned_Paper = edas:Paper AND (edas:isReviewedBy exists)"],
        "notes": ("the three CQA blocks ask it from the reviewer's side and are "
                  "reused with the paper as the answer variable; the ekaw and edas "
                  "blocks are written from the correspondence, which no CQA covers."),
    },
    {
        "id": "CC-09", "category": "multi-source",
        "question": "Which items make up the conference programme?",
        "select": ["event"], "combine": "union",
        "parts": [
            {"source": "ekaw", "body": "?event a <http://ekaw#Event> ."},
            {"source": "edas",
             "body": "{ ?event a <http://edas#Conference> . } "
                     "UNION { ?event a <http://edas#ConferenceEvent> . } "
                     "UNION { ?event a <http://edas#ConferenceSession> . }"},
        ],
        "exercises": ["ekaw:Event = edas:Conference OR edas:ConferenceEvent OR edas:ConferenceSession"],
        "notes": ("hand-written from the alignment: no CQA exercises any of the "
                  "three union correspondences, and this is the only category-4 "
                  "entry with no upstream id. Recategorized from conditional to "
                  "multi-source: it's a plain union with no filter/condition."),
    },

    # 4 ── distractors ───────────────────────────────────────────────────────
    # Worded like a general conference question, and the surrounding concept
    # (papers, reviews, people, sessions) really is in every source — but the
    # distinction the question turns on is modelled by exactly one ontology.
    # Without these a system that always calls all five endpoints scores a
    # perfect source-selection F1.
    {
        "id": "DS-01", "category": "distractor",
        "question": "Which papers were withdrawn?",
        "select": ["paper"], "combine": "single",
        "parts": [{"source": "edas", "body": "?paper a <http://edas#WithdrawnPaper> ."}],
        "notes": "papers exist in all five; withdrawal only in edas.",
    },
    {
        "id": "DS-02", "category": "distractor",
        "question": "Which reviews were positive?",
        "select": ["review"], "combine": "single",
        "parts": [{"source": "ekaw", "body": "?review a <http://ekaw#Positive_Review> ."}],
        "notes": "reviews exist in four sources; a verdict on the review only in ekaw.",
    },
    {
        "id": "DS-03", "category": "distractor",
        "question": "Who is registered as a user of the submission system?",
        "select": ["person"], "combine": "single",
        "parts": [{"source": "cmt", "body": "?person a <http://cmt#User> ."}],
        "notes": "people exist in all five; a system account only in cmt.",
    },
    {
        "id": "DS-04", "category": "distractor",
        "question": "Who volunteered to help out?",
        "select": ["person"], "combine": "single",
        "parts": [{"source": "confof", "body": "?person a <http://confOf#Volunteer> ."}],
        "notes": "roles exist everywhere; this one only in confof.",
    },
    {
        "id": "DS-05", "category": "distractor",
        "question": "Which announcements were published about the conference?",
        "select": ["item"], "combine": "single",
        "parts": [{"source": "conference", "body": "?item a <http://conference#Conference_announcement> ."}],
        "notes": "conferences are in all five; their announcements only in conference.",
    },
    {
        "id": "DS-06", "category": "distractor",
        "question": "Which poster sessions are in the programme?",
        "select": ["session"], "combine": "single",
        "parts": [{"source": "ekaw", "body": "?session a <http://ekaw#Poster_Session> ."}],
        "notes": "sessions and events are widespread; poster sessions only in ekaw.",
    },
    {
        "id": "DS-07", "category": "distractor",
        "question": "Which travel grants are on offer?",
        "select": ["grant"], "combine": "single",
        "parts": [{"source": "edas", "body": "?grant a <http://edas#TravelGrant> ."}],
        "notes": "money and registration appear in several sources; grants only in edas.",
    },
]


# --- compile ----------------------------------------------------------------

def compile_question(spec: dict) -> dict:
    blocks, sources = [], {}
    # The first part is the anchor: its block comes first and binds the entity,
    # and every later block reaches it through the link. Bridging happens here,
    # before analyse(), so triple_patterns and the vocabulary sets describe the
    # block as the gold query actually contains it.
    anchor = spec["parts"][0]["source"]
    join_var = spec.get("join_on")
    for part in spec["parts"]:
        src = part["source"]
        if "body" in part:
            body, cqa = " ".join(part["body"].split()), None
        else:
            cqa = part["cqa"]
            body = load_block(cqa, src, part["rename"], src)
        if spec["combine"] != "union" and join_var and src != anchor:
            body = bridge_block(body, join_var, src)
        patterns, ops = analyse(body)
        classes, properties = vocabulary(patterns)
        sources[src] = {
            "cqa_id": cqa,
            "cqa_file": (f"eval/cqa_source/CQAs/{cqa}/{CQA_FILENAME[src]}.sparql" if cqa else None),
            "endpoint": ENDPOINT[src],
            "body": body,
            "triple_patterns": patterns,
            "operators": ops,
            "classes": classes,
            "properties": properties,
        }
        blocks.append((src, body))

    upstream = sorted({p["cqa"] for p in spec["parts"] if "cqa" in p})
    entry = {
        "id": spec["id"],
        "category": spec["category"],
        "question": spec["question"],
        "cqa_ids": upstream,
        "gold_sources": sorted(sources),
        "combination": spec["combine"],
        "join_on": spec.get("join_on"),
        "select": spec["select"],
        "gold_query": build_query(spec["select"], blocks, spec["combine"]),
        "sources": sources,
        "gold_classes": sorted({c for s in sources.values() for c in s["classes"]}),
        "gold_properties": sorted({p for s in sources.values() for p in s["properties"]}),
        "exercises": spec.get("exercises", []),
        "notes": spec.get("notes"),
    }
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compile and report, write nothing")
    args = ap.parse_args()

    entries = [compile_question(q) for q in QUESTIONS]

    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate question id")

    by_cat = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    for cat, group in by_cat.items():
        print(f"{cat:20} {len(group):3}")
    print(f"{'TOTAL':20} {len(entries):3}")

    payload = {
        "description": ("Gold questions for the federated-SPARQL evaluation. Per-source "
                        "SPARQL comes from the IRIT CQA dataset (eval/cqa_source); the "
                        "wording, source sets and combination are ours. Built by "
                        "eval/build_gold_questions.py — edit that, not this file."),
        "endpoints": ENDPOINT,
        "categories": {c: len(g) for c, g in sorted(by_cat.items())},
        "count": len(entries),
        "questions": entries,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        print("\n--check: nothing written")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text(encoding="utf-8") != text:
        backup = OUT.with_suffix(".json.bak")
        shutil.copy2(OUT, backup)
        print(f"\nprevious version kept at {backup}")
    OUT.write_text(text, encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
