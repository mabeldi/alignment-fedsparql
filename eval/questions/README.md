# Gold question set

42 questions with a hand-checked federated SPARQL query for each. This is what
the metrics in [`../README.md`](../README.md) (step 3) read.

| File | What |
|---|---|
| `gold_questions.json` | **the artifact.** One JSON file, load this. |
| `../build_gold_questions.py` | builds it. Edit the recipe there, never the JSON. |
| `../verify_gold_questions.py` | checks it parses, the URIs exist, and it runs. |

```bash
python eval/build_gold_questions.py     # rebuild gold_questions.json
python eval/verify_gold_questions.py    # all three checks (needs the endpoints)
```

Both scripts keep a `.bak` copy before replacing a file they did not just produce.

---

## Where the queries come from

The per-source SPARQL is **not written by us**. It is copied out of
`eval/cqa_source/CQAs/`, the published *Competency Questions for Alignment*
dataset (Thiéblin et al., IRIT). Each folder there is one question written once
per ontology that can answer it:

```
CQAs/accepted_paper/cmt.sparql          ?s cmt:hasDecision ?o . ?o a cmt:Acceptance   UNION   ?s cmt:acceptedBy ?o
CQAs/accepted_paper/conference.sparql   ?s a conference:Accepted_contribution
CQAs/accepted_paper/edas.sparql         ?s a edas:AcceptedPaper
CQAs/accepted_paper/ekaw.sparql         ?s a ekaw:Accepted_Paper
```

So the vocabulary in a gold query is the published dataset's choice, not ours.
Every entry keeps its upstream folder name in `cqa_ids`, so a result traces back.
44 of the 100 CQAs are used; 8 questions have no CQA behind them and say so.

The only things we author are:

* the **wording** of the question,
* which **sources** the answer should reach,
* how the blocks **combine** (UNION vs join),
* the **variable names**.

Nothing was ever taken from the pipeline's own output. Grading a system against
its own answers measures nothing.

## What each entry carries

```jsonc
{
  "id": "CJ-01",
  "category": "cross-source-join",
  "question": "For every paper, give its summary and the subjects it deals with.",
  "cqa_ids": ["abstract_of_paper", "topic_of_paper"],   // upstream provenance
  "gold_sources": ["conference", "edas"],               // scored directly
  "combination": "join",                                // union | join | single
  "join_on": "paper",
  "select": ["paper", "abstract", "topic"],
  "gold_query": "SELECT DISTINCT ... SERVICE <...> { ... } ...",
  "sources": {
    "conference": {
      "cqa_id": "abstract_of_paper",
      "cqa_file": "eval/cqa_source/CQAs/abstract_of_paper/conference.sparql",
      "endpoint": "http://192.168.100.44:3030/conference/sparql",
      "body": "?paper a <http://conference#Written_contribution> . ...",
      "triple_patterns": [["?paper", "<...#type>", "<http://conference#Written_contribution>"], ...],
      "operators": [],            // UNION / MINUS / Filter / Extend, when present
      "classes": ["http://conference#Written_contribution"],
      "properties": ["http://conference#has_an_abstract"]
    },
    "edas": { ... }
  },
  "gold_classes": [...], "gold_properties": [...],
  "exercises": [...],   // the alignment cells this question depends on
  "notes": null
}
```

`triple_patterns` is the point of the file. The primary metric compares triple
patterns per `SERVICE` block, and making the harness re-parse `gold_query` to
recover them would be one more thing that can go wrong. They are produced by
parsing the same body the query is built from, so the two cannot disagree.

Patterns are 3-element arrays in N3 (`"?paper"`, `"<http://…>"`,
`"\"true\"^^<…#boolean>"`), URIs fully expanded, no prefixes anywhere.

**`operators` is a warning, not decoration.** Seven of the 117 blocks are not a
plain set of triples — they contain a `UNION` or a `MINUS`. The flat pattern set
loses that structure, so a system that produced the triples without the `MINUS`
would score full marks on the pattern metric. Read `operators` before trusting a
per-question score on those seven.

## Numbers

| Category | Questions |
|---|---:|
| multi-source coverage | 13 |
| cross-source join | 14 |
| conditional correspondence | 8 |
| distractors | 7 |
| **total** | **42** |

Gold source-set size — 7 questions have one source, so a system that always
calls all five endpoints cannot score well on source selection:

| sources | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| questions | 7 | 17 | 7 | 5 | 6 |

How often each ontology appears in a gold source set: `edas` 27, `ekaw` 27,
`confof` 24, `conference` 19, `cmt` 15. 112 `SERVICE` blocks, 165 gold triple
patterns in total.

---

## No question names an ontology

The system is meant to look like one system. Someone asks a question; which
ontologies answer it, and how the pieces are stitched together, is the system's
own business and is invisible from outside.

A category of five single-source questions was removed for breaking this. Their
wording named the ontology — *"In the CMT system, what is the name of each
person?"* — which hands the answer to the hardest part of the problem straight
to the pipeline, and then measures whether it can read it back. That measures
nothing.

Nothing was lost by removing them. The seven distractors below each answer from
a single source and cover all five ontologies between them, and they do it
honestly: the source has to be worked out from the words.

---

## The four categories

### 1. multi-source coverage (12)

One concept, a different name in every ontology. This is what the CQAs natively
are, and the gold query is the per-source patterns in `SERVICE` blocks joined by
`UNION`. Each endpoint holds its own instances, so leaving one out silently
returns a fraction of the answer.

Chosen for genuinely different names where possible — `confOf:Trip` /
`edas:Excursion` / `ekaw:Conference_Trip`, `confOf:Banquet` /
`edas:ConferenceDinner` / `ekaw:Conference_Banquet`, `cmt:SubjectArea` /
`ekaw:Research_Topic`, `confOf:Scholar` / `ekaw:Student`, `confOf:Company` /
`edas:IndustryOrganization`. Sizes run from 2 to 5 sources.

### 2. cross-source join (14)

The valuable one, and the one the CQAs do not give directly. Each question
composes **two** CQAs on a shared entity, and the pair of sources is picked so
that **neither source can answer both halves**:

> CJ-01 — *"For every paper, give its summary and the subjects it deals with."*
> `abstract_of_paper` exists for `cmt` and `conference` only.
> `topic_of_paper` exists for `cmt`, `confOf`, `edas`, `ekaw` only.
> Take the first from `conference` and the second from `edas`: `conference` has
> no topic query, `edas` has no abstract query. Neither endpoint alone answers
> the question, so the join is forced rather than optional.

That test — *source A answers half A and not half B, source B the reverse* — is
the rule for all 14. It is stricter than picking any two CQAs that overlap
somewhere, and it is what stops a system from scoring full marks by collapsing
to one endpoint.

Ontology pairs covered: `edas–ekaw` ×4, `conference–edas` ×2, `cmt–edas` ×2,
`cmt–confof` ×2, `confof–ekaw` ×2, `confof–edas`, `conference–confof`.
`cmt–edas` is deliberate: that pair had **zero** correspondences before the
merge described in `../README.md` step 1c, so it is the pair most likely to
expose a gap.

Join entities: paper (5), person (5), conference (2), workshop (2).

### 3. conditional correspondence (9)

`mapping.rdf` holds 13 correspondences that are not plain equivalences — the
concept is a named class in one ontology and *"class plus a condition on a
property"* in another:

```
ekaw:Accepted_Paper = cmt:Paper AND (cmt:hasDecision -> cmt:Acceptance)
conference:Reviewer = confof:Member_PC AND (confof:reviewes exists)
ekaw:Early-Registered_Participant = confof:Participant AND (confof:earlyRegistration = true)
ekaw:Event = edas:Conference OR edas:ConferenceEvent OR edas:ConferenceSession
```

Dropping the condition changes the answer, so these are where a system that
treats the alignment as a list of synonyms goes wrong.

How many of the 13 the CQAs let us reach:

| cell | question | note |
|---|---|---|
| `ekaw:Accepted_Paper = cmt:Paper AND (hasDecision -> Acceptance)` | CC-01 | |
| `conference:Accepted_contribution = cmt:Paper AND (hasDecision -> Acceptance)` | CC-01 | |
| `ekaw:Rejected_Paper = cmt:Paper AND (hasDecision -> Rejection)` | CC-02 | |
| `conference:Rejected_contribution = cmt:Paper AND (hasDecision -> Rejection)` | CC-02 | |
| `ekaw:Early-Registered_Participant = confof:Participant AND (earlyRegistration = true)` | CC-03 | |
| — the same shape with `false` | CC-04 | **not in mapping.rdf**; see `../derived_mappings/` |
| `conference:Reviewer = confof:Member_PC AND (reviewes exists)` | CC-05 | |
| `conference:Presentation = edas:Paper AND (isWrittenBy -> Presenter)` | CC-06 | partial, see below |
| `confof:Camera_Ready_event = ekaw:Individual_Presentation AND (...)` | CC-07 | partial, see below |
| `ekaw:Assigned_Paper = edas:Paper AND (isReviewedBy exists)` | CC-08 | no CQA, hand-written |
| `ekaw:Event = edas:Conference OR ConferenceEvent OR ConferenceSession` | CC-09 | no CQA, hand-written |
| `conference:Written_contribution = edas:Document AND (relatesTo -> Author)` | — | not covered |
| `edas:ConferenceEvent = ekaw:Scientific_Event OR Social_Event` | — | not covered |
| `ekaw:Scientific_Event = edas:Conference OR AcademicEvent OR ConferenceSession` | — | not covered |

10 of 13 reached, 3 not. The three left out are abstract event-hierarchy unions
with no CQA and no question a person would actually ask; inventing one would
have meant inventing the gold answer too.

### 4. distractors (7)

Worded like an ordinary conference question. The surrounding concept really is
in all five ontologies — papers, reviews, people, sessions — but the
**distinction the question turns on** is modelled by exactly one:

| question | gold source | why only one |
|---|---|---|
| Which papers were withdrawn? | edas | papers everywhere; `edas:WithdrawnPaper` alone |
| Which reviews were positive? | ekaw | reviews in four; `ekaw:Positive_Review` alone |
| Who is registered as a user of the submission system? | cmt | people everywhere; `cmt:User` alone |
| Who volunteered to help out? | confof | roles everywhere; `confOf:Volunteer` alone |
| Which announcements were published about the conference? | conference | `conference:Conference_announcement` alone |
| Which poster sessions are in the programme? | ekaw | `ekaw:Poster_Session` alone |
| Which travel grants are on offer? | edas | `edas:TravelGrant` alone |

Each class was checked twice: its local name occurs in no other ontology, and it
appears in no cell of `mapping.rdf`. One per ontology, plus two extra for `edas`
and `ekaw`.

Without this category a system that always queries all five endpoints scores a
perfect source-selection F1, which is the single easiest way to make the headline
number meaningless.

---

## Judgement calls

**Ordinary English words that happen to be class names.** *"Which papers were
accepted?"* uses "accepted", and three ontologies have a class with that word in
its name. Avoiding every such word would produce questions nobody asks. The line
drawn: the question may use the plain English concept, never the ontology's
spelling of it — no `Accepted_Paper`, no `SubjectArea`, no `Regular_author`. Where
a plain word would have pointed at one ontology's label, it was replaced:
"outings" not "excursions" (`edas:Excursion`), "formal dinners" not "banquets"
(`confOf:Banquet`), "research areas" not "subject areas" (`cmt:SubjectArea`),
"higher-education institutions" not "academic organisations"
(`edas:AcademiaOrganization`).

**Joining across endpoints assumes shared instance identity.** A cross-source
join matches a `conference` paper URI against an `edas` paper URI. With no
instance data that is untestable; it is also exactly what the pipeline's own
prompt tells the model to do ("if the sources hold complementary data about the
same entity, join their `SERVICE` blocks on a shared variable"), so the gold and
the system are being judged on the same convention. If real data ever arrives,
this convention needs revisiting before any answer-based metric.

**Helper variables are renamed per block.** The CQAs all use `?s`, `?o`, `?o2`.
Two `SERVICE` blocks in a joined query share one variable scope, so two blocks
that both used `?o2` would be joined on it by accident — a join nobody wrote and
nobody can see. Every variable that is not a named answer variable becomes
`?<source>_<name>`. The named ones (`?paper`, `?person`, …) are deliberately
shared, which is also what lets `UNION` branches line up on the same columns.

**CC-06 and CC-07 are partial.** In CC-06 the alignment says
`conference:Presentation = edas:Paper AND (isWrittenBy -> Presenter)` while the
`edas` CQA answers with `edas:PaperPresentation`. The gold uses the CQA — it is
the published answer — and the correspondence is recorded in `exercises` as the
reason `edas` is reachable from `conference` at all. In CC-07 the `ekaw` half of
the correspondence has no CQA, so `ekaw` is **not** in the gold source set even
though the cell mentions it.

**CC-08, CC-09 and the seven distractors have no upstream id** (`cqa_ids: []`).
They are written directly against the ontologies and the alignment because no CQA
covers them. Every URI in them was checked against the OAEI 2022 files, same as
everything else.

**Why 42 and not 40.** No category was padded. Cross-source join got 14 because
it is the category the evaluation actually turns on and the CQAs support that
many honest compositions; conditional stopped at 9 because that is all the 13
cells support.

---

## Verification

`python eval/verify_gold_questions.py`, last run:

```
parse                     42/42      rdflib prepareQuery
URIs exist                42/42      112 SERVICE blocks, every class and property
                                     found in the OAEI 2022 file for its own source
execute (HTTP 200)        42/42      against http://192.168.100.44:3030
```

Nothing failed and nothing was dropped.

Empty result sets are expected — the endpoints hold no instance data and never
will. The execution check only separates *"well-formed and reachable"* from
*"broken"*; it says nothing about whether the answer would be right.

The ontology files are cached in `eval/.cache/ontologies/` after the first run.
Delete that folder to re-download. They are the same 2022 files `app/main.py`
loads, on purpose: a term that exists in one OAEI year and not another would pass
here and fail in the pipeline.
