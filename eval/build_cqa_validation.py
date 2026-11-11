"""Build one federated SPARQL query for every folder in questions/CQAs.

Each CQA folder contains equivalent query patterns for the ontologies that can
answer that competency question.  The generated query places each pattern in
its ontology's SERVICE and UNIONs the alternatives, preserving the projected
variables from the source queries. Complementary queries are a different case:
they must join source-local individuals through owl:sameAs (and may make a
complementary SERVICE optional when its value is not required).

Run:
    python eval/build_cqa_validation.py
    python eval/build_cqa_validation.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from rdflib.plugins.sparql import prepareQuery


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "eval" / "questions" / "CQAs"
OUTPUT_DIR = ROOT / "eval" / "questions" / "CQAs-validation"
FUSEKI_BASE = os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030").rstrip("/")

SOURCE_NAMES = {
    "cmt": "cmt",
    "conference": "conference",
    "confOf": "confof",
    "edas": "edas",
    "ekaw": "ekaw",
}


# Complete natural-language wording for each compact CQA identifier.
QUESTIONS = {
    "abstract_of_invited_talk": "Which abstract belongs to each invited talk?",
    "abstract_of_paper": "Which abstract belongs to each paper?",
    "accepted_paper": "Which papers were accepted?",
    "administrator": "Who are the administrators?",
    "all_reviewer_author_of_review": "Which reviewers authored which reviews?",
    "assigned_reviewer": "Which reviewers have been assigned a paper to review?",
    "assigned_reviewer_author_of_review": "Which assigned reviewers authored which reviews?",
    "author": "Who are the authors?",
    "author_of_inv_talk_abstract": "Who authored each invited-talk abstract?",
    "chair_of_workshop": "Who chairs each workshop?",
    "company": "Which companies are represented?",
    "conference": "Which conferences are described?",
    "conference_camera_date": "What is the camera-ready submission deadline for each conference?",
    "conference_chair": "Who are the conference chairs?",
    "conference_end_date": "What is the end date of each conference?",
    "conference_location": "Where is each conference located?",
    "conference_name": "What is the name of each conference?",
    "conference_notif_date": "What is the acceptance-notification date for each conference?",
    "conference_paper_date": "What is the paper-submission deadline for each conference?",
    "conference_proceedings": "Which conference proceedings are available?",
    "conference_proceedings_name": "What is the name of each conference proceedings volume?",
    "conference_regis_date": "What is the registration deadline for each conference?",
    "conference_site": "Which conference websites are described?",
    "conference_site_url": "What are the conference website URLs?",
    "conference_start_date": "What is the start date of each conference?",
    "conference_topic": "Which topics does each conference cover?",
    "early_registered_participant": "Which participants registered early?",
    "early_registration": "Did each participant register early?",
    "external_reviewer": "Who are the external reviewers?",
    "gala_dinner": "Which gala dinners are scheduled?",
    "gala_dinner_end_date": "What is the end date and time of each gala dinner?",
    "gala_dinner_location": "Where is each gala dinner held?",
    "gala_dinner_start_date": "What is the start date and time of each gala dinner?",
    "invited_speaker": "Who are the invited speakers?",
    "invited_speaker_of_talk": "Which invited speaker gives each invited talk?",
    "invited_talk": "Which invited talks are scheduled?",
    "invited_talk_abstract": "Which abstracts are invited-talk abstracts?",
    "invite_reviewer": "Who invited which reviewer?",
    "is_assigned_reviewer_of": "Which reviewer is assigned to which paper?",
    "is_author_of_paper": "Which author wrote which paper?",
    "late_registered_participant": "Which participants registered late?",
    "member_of_conference": "Which person is a member of which conference?",
    "member_of_pc": "Which person belongs to which program committee?",
    "oc_chair": "Who are the organizing-committee chairs?",
    "oc_member": "Who are the organizing-committee members?",
    "paper": "Which papers are there?",
    "paper_1st_author": "Who are the first authors of papers?",
    "paper_abstract": "Which paper abstracts are there?",
    "paper_abstract_title": "What is the title of each paper abstract?",
    "paper_co_author": "Who are the paper co-authors?",
    "paper_presentation": "Which paper presentations are scheduled?",
    "paper_submitted_at_conference": "To which conference was each paper submitted?",
    "paper_title": "What is the title of each paper?",
    "participant": "Who are the conference participants?",
    "pc_chair": "Who are the program-committee chairs?",
    "pc_member": "Who are the program-committee members?",
    "person": "Which people are described?",
    "person_first_name": "What is each person's first name?",
    "person_last_name": "What is each person's last name?",
    "person_name": "What is each person's full name?",
    "presentation_in_conference": "At which conference is each presentation given?",
    "presentation_of_paper": "Which paper is presented by each presentation?",
    "presenter": "Who are the paper presenters?",
    "presenter_of_paper_presentation": "Who presents each paper presentation?",
    "proceedings_publisher": "Who publishes each conference proceedings volume?",
    "program_committee": "Which program committees are described?",
    "publisher": "Which proceedings publishers are described?",
    "reception": "Which receptions are scheduled?",
    "reception_end_date": "What is the end date and time of each reception?",
    "reception_location": "Where is each reception held?",
    "reception_start_date": "What is the start date and time of each reception?",
    "regular_paper": "Which papers are regular papers?",
    "rejected_paper": "Which papers were rejected?",
    "review": "Which reviews are there?",
    "reviewers_all": "Who are all the reviewers?",
    "review_of_paper": "Which paper does each review evaluate?",
    "sc_member": "Who are the steering-committee members?",
    "site_url_of_conference": "What website URL belongs to each conference?",
    "student": "Which people are students?",
    "topic": "Which research topics are described?",
    "topic_of_paper": "Which topics does each paper cover?",
    "track": "Which conference tracks are described?",
    "track_of_conference": "Which tracks belong to each conference?",
    "trip": "Which conference trips are scheduled?",
    "trip_end_date": "What is the end date and time of each conference trip?",
    "trip_location": "Where is each conference trip held?",
    "trip_start_date": "What is the start date and time of each conference trip?",
    "tutorial": "Which tutorials are scheduled?",
    "tutorial_of_conference": "Which tutorials belong to each conference?",
    "tutorial_topic": "Which topic does each tutorial cover?",
    "university": "Which universities are represented?",
    "workshop": "Which workshops are scheduled?",
    "workshop_camera_date": "What is the camera-ready submission deadline for each workshop?",
    "workshop_chair": "Who are the workshop chairs?",
    "workshop_end_date": "What is the end date and time of each workshop?",
    "workshop_notif_date": "What is the review-notification deadline for each workshop?",
    "workshop_of_conference": "Which workshops belong to each conference?",
    "workshop_start_date": "What is the start date and time of each workshop?",
    "workshop_submission_date": "What is the paper-submission deadline for each workshop?",
    "workshop_topic": "Which topic does each workshop cover?",
}


def strip_prefixes(text: str) -> tuple[str, dict[str, str]]:
    """Remove PREFIX declarations and return their namespace mapping."""
    prefixes = {
        (match.group(1) or ""): match.group(2)
        for match in re.finditer(
            r"(?i)PREFIX\s+([A-Za-z][\w.-]*)?\s*:\s*<([^>]+)>", text
        )
    }
    without = re.sub(
        r"(?i)PREFIX\s+([A-Za-z][\w.-]*)?\s*:\s*<[^>]+>", "", text
    )
    return without, prefixes


def expand_curies(text: str, prefixes: dict[str, str]) -> str:
    """Expand prefixed names while leaving IRIs and string literals untouched."""
    def expand_segment(segment: str) -> str:
        def replace(match: re.Match[str]) -> str:
            prefix, local = match.group(1) or "", match.group(2)
            namespace = prefixes.get(prefix)
            return f"<{namespace}{local}>" if namespace else match.group(0)

        return re.sub(
            r"(?<![?$\w])([A-Za-z][\w-]*)?:([A-Za-z_][\w-]*)",
            replace,
            segment,
        )

    result: list[str] = []
    position = 0
    for match in re.finditer(r'<[^>]*>|"(?:\\.|[^"\\])*"', text):
        result.append(expand_segment(text[position:match.start()]))
        result.append(match.group(0))
        position = match.end()
    result.append(expand_segment(text[position:]))
    return "".join(result)


def where_body(text: str) -> str:
    """Return the text inside the query's outer WHERE braces."""
    where = re.search(r"(?i)\bWHERE\b", text)
    start = text.find("{", where.end() if where else 0)
    if start < 0:
        raise ValueError("query has no WHERE group")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    raise ValueError("unbalanced WHERE braces")


def projection(text: str) -> tuple[str, ...]:
    match = re.search(
        r"(?is)\bSELECT\s+(?:DISTINCT\s+)?(.+?)\s+WHERE\s*\{", text
    )
    if not match:
        raise ValueError("query has no SELECT projection")
    variables = tuple(re.findall(r"[?$](\w+)", match.group(1)))
    if not variables:
        raise ValueError("query projection has no variables")
    return variables


def rename_helpers(body: str, selected: tuple[str, ...], source: str) -> str:
    """Keep answer variables shared and scope all helper variables by source."""
    selected_set = set(selected)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return f"?{name}" if name in selected_set else f"?{source}_{name}"

    return re.sub(r"[?$](\w+)", replace, body)


def clean_body(body: str) -> str:
    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    return "\n".join(lines)


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def build_query(folder: Path) -> str:
    question = QUESTIONS[folder.name]
    branches: list[str] = []
    expected_projection: tuple[str, ...] | None = None

    for path in sorted(folder.glob("*.sparql"), key=lambda item: item.stem.lower()):
        if path.stem not in SOURCE_NAMES:
            raise ValueError(f"unknown ontology filename: {path}")
        source = SOURCE_NAMES[path.stem]
        text = path.read_text(encoding="utf-8-sig")
        selected = projection(text)
        if expected_projection is None:
            expected_projection = selected
        elif selected != expected_projection:
            raise ValueError(
                f"projection mismatch in {folder.name}: "
                f"{expected_projection} != {selected} ({path.name})"
            )

        without_prefixes, prefixes = strip_prefixes(text)
        body = where_body(without_prefixes)
        body = expand_curies(body, prefixes)
        body = rename_helpers(body, selected, source)
        body = clean_body(body)
        endpoint = f"{FUSEKI_BASE}/{source}/sparql"
        branch = (
            f"  {{\n"
            f"    # source: {source}\n"
            f"    SERVICE <{endpoint}> {{\n"
            f"{indent(body, 6)}\n"
            f"    }}\n"
            f"  }}"
        )
        branches.append(branch)

    if not branches or expected_projection is None:
        raise ValueError(f"no SPARQL files in {folder}")

    variables = " ".join(f"?{name}" for name in expected_projection)
    union = "\n  UNION\n".join(branches)
    query = (
        f"#question: {question}\n"
        f"# CQA: {folder.name}\n\n"
        f"# federation: UNION of complete alternative answers from each source\n\n"
        f"SELECT DISTINCT {variables}\n"
        f"WHERE {{\n"
        f"{union}\n"
        f"}}\n"
    )
    prepareQuery(query)
    return query


def compile_queries() -> dict[str, str]:
    folders = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())
    folder_names = {path.name for path in folders}
    question_names = set(QUESTIONS)
    if folder_names != question_names:
        missing = sorted(folder_names - question_names)
        stale = sorted(question_names - folder_names)
        raise ValueError(f"question map mismatch; missing={missing}, stale={stale}")
    return {f"{folder.name}.sparql": build_query(folder) for folder in folders}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated files are current without writing them",
    )
    args = parser.parse_args()
    generated = compile_queries()

    if args.check:
        existing = {
            path.name: path.read_text(encoding="utf-8")
            for path in OUTPUT_DIR.glob("*.sparql")
        } if OUTPUT_DIR.exists() else {}
        if existing != generated:
            missing = sorted(set(generated) - set(existing))
            extra = sorted(set(existing) - set(generated))
            changed = sorted(
                name for name in set(existing) & set(generated)
                if existing[name] != generated[name]
            )
            print(f"generated files are stale: missing={missing}, extra={extra}, changed={changed}")
            return 1
        print(f"validated {len(generated)} federated SPARQL files")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in generated.items():
        (OUTPUT_DIR / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(generated)} federated SPARQL files to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
