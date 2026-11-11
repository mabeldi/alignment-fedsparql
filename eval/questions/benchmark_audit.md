# Benchmark audit — wording vs. gold projection

This exists so the 42 questions can be adjudicated once and then frozen, before
the final runs. The seven arity questions were decided on 2026-08-09 and applied
(see below); everything else is still open.

Adjudicate each row from the **question wording and the intended task only**.
Do not look at what any generated query produced — deciding gold after seeing
predictions is test-set leakage, and the whole point of freezing is to make that
impossible.

## How the "contains" column was produced

The RDF kind of each projected column was measured on the live endpoints, not
guessed. A variable's kind is decided by the patterns that *bind* it, so each
binding pattern was probed on its own dataset with a single-pattern query
(`isIRI` / `isBlank` / `DATATYPE` / `LANG` evaluated server-side). That sees
every `UNION` branch by construction, because every branch's patterns are
enumerated.

Two caveats on that number:

- It is an **upper bound**. The probe asks what kinds a predicate takes across
  the whole dataset, not what survives the question's other constraints. Where
  a column reads `identifier (or blank)`, the blank nodes may well be filtered
  out by the rest of the query. The identifier-versus-literal split — which is
  what actually needs deciding — is unaffected.
- The twelve single-source questions were additionally verified by executing the
  real gold query, and those came back `identifier` only.

## The decision that applies to almost every row

**30 of 42 questions return entity identifiers for wording that never says
whether it wants an identifier or a name.** "Who are the authors?", "Which
conferences are described?", "Which papers were accepted?" — all answered with
identifiers. That is a perfectly defensible convention, but right now it is
unstated, so a system returning names is being marked wrong for guessing a
different unstated convention.

Pick one rule, write it into the benchmark description, and the whole class
resolves at once:

> Fields naming an entity return its identifier. Fields explicitly asking for a
> name, title, surname, address or date return a literal. Each question states
> which fields it wants.

## Decisions taken — 2026-08-09

All seven arity questions were resolved by **rewording**; no variable was
dropped, so no `select` list, no gold query and no gold answer set changed.
Applied to both `eval/questions/gold_questions.json` and the `QUESTIONS` list in
`eval/build_gold_questions.py`.

| id | new wording |
|---|---|
| MC-05 | Which papers are there, and what is the title of each one? |
| CJ-03 | Which reviews were written, which paper does each one review, and to which conference was that paper submitted? |
| CJ-05 | Who ran a workshop, what is each one's surname, and which workshop did they run? |
| CJ-07 | Who is still studying, and what is each one's given name? |
| CJ-09 | Who sits on the organising committee, and what is each member's surname? |
| CJ-14 | Which papers were accepted, and what subject does each deal with? |
| CC-07 | For each conference, by when must the final version of a paper be handed in? |

**The wording is the prompt.** Any run recorded before this date used the old
wording for these seven and must be regenerated before it is compared with
anything produced after it.

## Rows that needed a decision (original state, kept as the record)

Seven questions ask for **fewer fields in the wording than gold projects**.
These are the ones that cannot be resolved by a global convention.

| id | question | gold projects | contains | issue |
|---|---|---|---|---|
| MC-05 | What is the title of each paper? | `paper`, `title` | identifier, literal `string` | wording asks only for titles |
| CJ-03 | Which reviews were written, and to which conference was the reviewed paper submitted? | `review`, `paper`, `conf` | identifier ×3 | the paper is named as context, not requested as output |
| CJ-05 | What are the surnames of the people who ran a workshop, and which workshop did each run? | `person`, `surname`, `workshop` | identifier, literal `string`, identifier | wording asks for surname + workshop |
| CJ-07 | What are the given names of the people who are still studying? | `person`, `firstname` | identifier, literal `string` | wording asks only for given names |
| CJ-09 | What are the surnames of the organising committee members? | `person`, `surname` | identifier, literal `string` | wording asks only for surnames |
| CJ-14 | Which subjects do the accepted papers deal with? | `paper`, `topic` | identifier ×2 | wording asks only for subjects |
| CC-07 | By when must the final version of a paper be handed in? | `conf`, `date` | identifier, literal **mixed** | wording asks only for the date |

Each resolves one of two ways: drop the extra variable from gold, or reword the
question to ask for it. Suggested rewordings for the two already discussed:

- **MC-05** → *For each paper, return the paper identifier and its title.*
- **CJ-03** → *For each review, return the review, the reviewed paper, and the conference to which that paper was submitted.*

## Data-representation problems, independent of wording

| id | column | finding | why it matters |
|---|---|---|---|
| CC-07 | `date` | three declared datatypes in one column: `xsd:string` (confof, 247 values), `xsd:date` (conference, 5), `xsd:dateTime` (edas, 5) | harmless today, a trap later — see below |
| CJ-13 | `deadline` | `xsd:string`, while CJ-10 / CJ-11 / CJ-12 use `xsd:dateTime` | same situation as CC-07 |
| CJ-01, CJ-02 | `abstract` | returned as an identifier, not text | a "summary" that is an identifier is a strange answer to give a user |

**Correction to an earlier version of this file:** the three date columns do NOT
write the same instant three different ways. The declared types disagree, but
the text is identical in shape everywhere — `2008-03-22T21:00:00`. The executor
keeps only the text, so today's scorer never sees the disagreement and nothing
is mis-scored.

It matters in one direction only: making the scorer type-aware, which one review
recommended, would turn a column that currently compares fine into three
mutually unequal columns. If that change is ever made, normalise these first.

Separately, `conference`'s values are declared `xsd:date` but carry a time
(`2012-07-11T18:00:00`), which is not a valid `xsd:date`. A data defect, with no
effect on text comparison.

## Full table

`contains` is the measured RDF kind. The seven arity rows are decided; the
`convention`, `representation` and `datatype` rows still need you.

| id | gold projects | contains | wording implies | flag | decision | reason |
|---|---|---|---|---|---|---|
| MC-01 | `paper` | identifier (or blank) | entity list | convention | | |
| MC-02 | `person` | identifier (or blank) | entity list | convention | | |
| MC-03 | `topic` | identifier (or blank) | entity list | convention | | |
| MC-04 | `conf` | identifier (or blank) | entity list | convention | | |
| MC-05 | `paper`, `title` | identifier, literal `string` | title only | arity **resolved: reword** | reword | wording now asks for every projected field |
| MC-06 | `person` | identifier (or blank) | entity list | convention | | |
| MC-07 | `person` | identifier (or blank) | entity list | convention | | |
| MC-08 | `outing` | identifier (or blank) | entity list | convention | | |
| MC-09 | `dinner` | identifier (or blank) | entity list | convention | | |
| MC-10 | `org` | identifier (or blank) | entity list | convention | | |
| MC-11 | `person` | identifier (or blank) | entity list | convention | | |
| MC-12 | `org` | identifier (or blank) | entity list | convention | | |
| CJ-01 | `paper`, `abstract`, `topic` | identifier ×3 | paper + summary + subjects | **representation** | | |
| CJ-02 | `paper`, `abstract`, `conf` | identifier ×3 | paper + summary + conference | **representation** | | |
| CJ-03 | `review`, `paper`, `conf` | identifier ×3 | review + conference | arity **resolved: reword** | reword | wording now asks for every projected field |
| CJ-04 | `author`, `paper`, `topic` | identifier ×3 | author + paper + subjects | ok | | |
| CJ-05 | `person`, `surname`, `workshop` | identifier, literal, identifier | surname + workshop | arity **resolved: reword** | reword | wording now asks for every projected field |
| CJ-06 | `person`, `conf` | identifier ×2 | staff + conference | ok | | |
| CJ-07 | `person`, `firstname` | identifier, literal | given name only | arity **resolved: reword** | reword | wording now asks for every projected field |
| CJ-08 | `person`, `firstname`, `invitee` | identifier, literal, identifier | inviter + invitee + given name | ok | | |
| CJ-09 | `person`, `surname` | identifier, literal | surname only | arity **resolved: reword** | reword | wording now asks for every projected field |
| CJ-10 | `conf`, `place`, `deadline` | identifier, literal, `dateTime` | conference + place + deadline | ok | | |
| CJ-11 | `conf`, `url`, `enddate` | identifier, `anyURI`, `dateTime` | conference + address + end date | ok | | |
| CJ-12 | `workshop`, `topic`, `startdate` | identifier ×2, `dateTime` | workshop + subject + start | ok | | |
| CJ-13 | `workshop`, `conf`, `deadline` | identifier ×2, literal `string` | workshop + conference + deadline | **datatype** | | |
| CJ-14 | `paper`, `topic` | identifier ×2 | subjects only | arity **resolved: reword** | reword | wording now asks for every projected field |
| CC-01 | `paper` | identifier (or blank) | entity list | convention | | |
| CC-02 | `paper` | identifier (or blank) | entity list | convention | | |
| CC-03 | `person` | identifier (or blank) | entity list | convention | | |
| CC-04 | `person` | identifier (or blank) | entity list | convention | | |
| CC-05 | `person` | identifier (or blank) | entity list | convention | | |
| CC-06 | `talk` | identifier (or blank) | entity list | convention | | |
| CC-07 | `conf`, `date` | identifier, literal **mixed** | date only | arity **resolved: reword**; datatype **open** | reword | wording now asks for every projected field; mixed date types still unresolved |
| CC-08 | `paper` | identifier (or blank) | entity list | convention | | |
| CC-09 | `event` | identifier (or blank) | entity list | convention | | |
| DS-01 | `paper` | identifier (or blank) | entity list | convention | | |
| DS-02 | `review` | identifier (or blank) | entity list | convention | | |
| DS-03 | `person` | identifier (or blank) | entity list | convention | | |
| DS-04 | `person` | identifier (or blank) | entity list | convention | | |
| DS-05 | `item` | identifier (or blank) | entity list | convention | | |
| DS-06 | `session` | identifier (or blank) | entity list | convention | | |
| DS-07 | `grant` | identifier (or blank) | entity list | convention | | |

**convention** — resolved by stating the identifier-versus-name rule once, globally.
**arity** — the wording asks for fewer fields than gold returns; needs a per-question decision.
**representation** — the column's kind is semantically odd for what the wording asks.
**datatype** — the literal's type is inconsistent, within a column or across questions.

## After the decisions

1. Apply them to `gold_questions.json` (and to `eval/build_gold_questions.py`, which regenerates it).
2. Re-check the five held-out questions in `answer_eval_exclusions.json` — DS-01 and DS-04 to DS-07 still return no rows.
3. Freeze the benchmark.
4. Only then run the three complete repetitions with `--execute`.


---

# Final state — every item closed except one

The sections above are the working record. This is the settled position.

## Projection: closed, 42 of 42

Rule applied: *the wording must explicitly request, as something returned, every
variable in `select`.* A variable appearing only as context ("the title of each
**paper**") is not a request.

- 26 questions project one column and cannot have the problem.
- All 16 multi-column questions were reviewed and now name every projected field.
- Resolved entirely by rewording. **No `select` list changed, no gold query
  changed, no gold answer set changed.**

Also tightened: every question that returns people now opens "Which people…"
rather than "Who…", because "who" is answerable by a name alone and that
collided with the identifier-versus-name question below.

## CJ-04 "lead writer": closed, no change needed

Measured, not assumed: cmt holds 4,300 papers, exactly 2,150 `hasAuthor` links
(one per paper that has one) and a separate co-author link carrying up to 9 per
paper. The single author link is the author of record, so the body does
implement "lead writer". Question and query agree.

## CJ-01 / CJ-02 summary: closed

The query returns a reference to the summary, not its text, so the wording now
asks *which summary belongs to each one* instead of *what is each one's
summary*. The question no longer promises something the answer does not deliver.

## Vocabulary leakage: closed

All 42 questions were swept for gold class and property names appearing in the
wording. 49 raw hits across 25 questions — but almost all are the domain's
ordinary English ("paper", "conference", "accepted paper"). A benchmark about
papers cannot avoid the word "paper", and the ontology naturally names that
class after it. Those are the task, not a leak.

One genuine case: CJ-04 said "deal with" against a gold property of that exact
name. Changed to "cover". CJ-01 was changed to match for consistency.

Judged and left alone: "administrative staff" against a class named for
administrators — the natural English for the concept, same category as "paper".

## Builder drift: closed

`eval/build_gold_questions.py` cannot run (its input CQA files are missing), so
`gold_questions.json` is authoritative. Two builder entries disagreed with it and
would have produced wrong queries if it were ever repaired and re-run:

- CJ-07 declared the wrong source for its second part.
- CJ-11 had subject and object swapped.

Both corrected to match the JSON.

## Date datatypes (CC-07, CJ-13): closed, no action

The declared types disagree across sources but the text is identical in shape, and
the scorer compares text. Nothing is mis-scored. Becomes a trap only if the
scorer is ever made type-aware — normalise first if that happens.

## Union duplication: closed, no action

No union question bridges sources with an identity link, so one real entity
appears once per source. Gold and prediction behave identically, so nothing is
mis-scored. The only risk is inverted: a system that merged them would be marked
wrong for being more correct than gold.

## Empty gold (DS-01, DS-04 to DS-07): closed

Held out by the frozen list in `answer_eval_exclusions.json`. Answer metrics rest
on 37 of 42 and say so.

## Identifier or name — CLOSED 2026-08-09

30 questions return entity identifiers for wording that never says which form is
wanted, and the pipeline was returning names instead. Naming the field in the
question cannot fix that: it is about the FORM of the answer, not which fields
come back.

**Two prompt attempts failed.** A sentence was added to the generator's OUTPUT
section stating the convention, then rewritten to forbid the behaviour outright.
Neither changed anything (both verified against a server running the edited
file). The reason is upstream: the schema handed to the generator advertised
`rdfs:label` on 224 of 302 classes, and the generator's first rule is to write
only what it was handed. An edge offered on three quarters of all classes is not
an option, it is a default.

**Applied instead — the offer was withdrawn at the source.** `value_edges_for`
in `app/tools/find_class_paths.py` no longer appends `rdfs:label` to classes
that declare no name of their own. Measured after the change: 0 of 302 classes
are offered it. The generator can no longer return a name where the ontology
declares none, so the form of the answer follows from the schema rather than
from an instruction it is free to ignore.

Two questions depended on that edge and were rewritten so nothing is broken by
its removal:

| id | before | after | why |
| --- | --- | --- | --- |
| MC-05 | 5-source union; conference, edas and ekaw read the title off `rdfs:label` | cmt + confof only, `?paper ?title` unchanged | those three declare no title property at all, so no schema-driven system can write those branches. Dropping `?title` instead was rejected: MC-01 is already the 5-source "list all the papers" and MC-05 would have duplicated it. Executes, 4,300 rows. |
| CJ-10 | `?conf ekaw:heldIn ?loc . ?loc rdfs:label ?place` | `?conf ekaw:heldIn ?place` | `ekaw:Location` declares no datatype property at all, so a label was the only text on it. Returning the place itself matches every other question. All three columns kept. Executes, 5 rows. |

Both deviate from the published CQA bodies; that is recorded here and belongs in
the paper's benchmark description. No other gold query referenced `rdfs:label` —
checked across all 42, in both `gold_query` and every per-source `body`.

The generator's OUTPUT section keeps only the general projection rule ("project
exactly what the question asks for, and no others"), which is what fixed the
arity mismatches. The label-specific wording was removed once the cause was gone,
so nothing benchmark-specific is baked into the prompt.

## Scoring: same thing, different source's URI — CLOSED 2026-08-09

Five sources store five URIs for one real paper. Which one an answer carries
depends only on which source the query anchored on, and that anchor is a free
choice the system is meant to make for itself. Comparing the strings scored that
choice instead of the answer.

Measured before the fix: CJ-14 returned **4,370 rows against gold's 4,370** and
scored 0.

`align_tables` in `eval/harness/score.py` now rewrites every URI to one
representative of its `owl:sameAs` class before comparing, on both sides. Only
`owl:sameAs` is followed — an identity claim the datasets themselves assert
(cmt 73,963, confof 42,918, ekaw 79,960, conference 68,522, edas 86,225, stored
symmetrically). A URI and its label are deliberately NOT collapsed: that is the
evaluator's opinion rather than the data's, and it would let a query return a
paper's ID where the question asked for its title and still pass. That leniency
stays in the per-column diagnostic, where it already was.

Effect on the 4-question probe: exec accuracy 0.000 -> 0.250, answer rows F1
0.174 -> 0.417.

It also exposed a double count. MC-05's gold union returns **4,300 rows for 2,150
papers** — each paper once under cmt's URI and once under confof's. Canonicalised
it is 2,150, the generated answer covers all of them (recall 1.000) and adds
2,150 more from a broader paper class (precision 0.500, F1 0.667). The previous
0.697 was mixing URI-choice noise into that. No gold query was changed; the
double count is a property of a UNION over sources and both sides are now
collapsed the same way.

Related guard, same commit: a generated query that fails with a CONNECTION error
is now unscorable rather than 0. Fuseki drops every connection when it dies, and
scoring a crash as a wrong answer is a silent one-directional bias against the
system under test. A timeout or an HTTP error still scores 0 — the endpoint
answered, the query is genuinely at fault.

## The instances are replicated, so answers alone cannot prove federation

Found on the 10-question pass, 2026-08-09. It is a property of the dataset, not
a fault in the scorer, and it has to be stated in the paper.

OAEI conference is an INSTANCE-MATCHING benchmark: the five ontologies describe
the same individuals and link them with `owl:sameAs`. So for most questions one
source already holds the whole answer.

| | gold | what the system did | scored |
| --- | --- | --- | --- |
| CC-01 | union over 4 sources, 3,484 rows | queried cmt alone, 871 rows | **exact match** |
| CJ-09 | join over 2 sources, 304 rows | queried conference alone, 304 rows | **exact match** |

3,484 / 871 = exactly 4. Gold returns the same 871 papers once per source.

Consequence: **execution accuracy and answer F1 cannot distinguish a federated
solution from a single-source one on this data.** Three of the five exact matches
in that pass were obtained by AVOIDING federation. Removing the `owl:sameAs`
canonicalisation would not fix this — it would only re-introduce the worse
problem of scoring a correct answer 0 for anchoring on a different source.

What was done instead: `source F1` and `join F1` were promoted out of the
diagnostics and into the primary table in `report.py`, alongside the two answer
metrics, and the report says why. Answer accuracy says the question was answered;
those two say whether it was answered by reaching across sources. On this
benchmark both claims are needed and neither implies the other.

Same pass, same commit: the per-column diagnostic `answer_f1` now canonicalises
too. It was contradicting the headline — CJ-14 read rows 1.000 and columns 0.000
on one table, because only the row comparison collapsed the two sources' URIs.
Diagnostic average over the ten moved 0.482 -> 0.756; no headline number changed.

## Next

Freeze. `verify_gold_questions.py` passes 42/42 on parse and URI existence; the
execute check needs Fuseki up and dies part-way through under the heavy federated
queries, so re-run it until it completes. Then regenerate every run (all previous
runs answered older wording and are void), then score with `--execute` and three
repeats.
