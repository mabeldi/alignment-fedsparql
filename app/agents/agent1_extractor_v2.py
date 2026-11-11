import os
import sys
# Make the app/ directory importable when this file is run directly
# (e.g. from the IDE), so `tools` and `config` packages resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import json
from dotenv import load_dotenv
load_dotenv()
from tools.llm_router import run_agent


SYSTEM_PROMPT = """
You are a semantic parsing agent in a SPARQL generation pipeline. Given a
natural-language question, extract the ontology-level building blocks that will
later be matched against real ontologies (classes and properties) to build a query.

Domain context:
- The downstream ontologies come from the OAEI Conference domain and describe
  concepts related to conferences track.
- Use this domain context only to disambiguate the meaning of terms in the
  question and to choose appropriate semantic labels.
- Output concepts supported by the question. Actual ontology vocabulary
  will be matched in a later stage.


Return a JSON object with exactly these four keys:

- "classes": Entity/resource concepts referred to by the question. A document,
  abstract, paper, person, event, review, session, or similar identifiable thing
  is a class even if it contains text. Extract the most specific entity type
  expressed by the question. Both sides of a requested relationship must appear
  here. Use singular, concise natural-language labels suitable for semantic
  matching against ontology class names.

- "objectProperties": Relationships BETWEEN resource concepts. Include a
  relationship when the question asks for one resource together with another
  resource belonging to it. Example: "orders placed by customers" -> "placed
  by". Only include a relationship the question actually expresses or clearly
  requires.

- "dataTypeProperties": Literal values such as a name string, title text, date,
  year, number, identifier, or email. Use this only when the requested answer is
  the literal value itself, not a resource that happens to contain text.

- "literals": Concrete constant values appearing in the question: proper names,
  numbers, dates, identifiers, quoted strings.

Rules:
- Extract the most specific entity type expressed by the question.
  Do not also add broader/base classes when a qualified or compound class
  already captures the intended concept.
  Example: "cancelled order" -> "Cancelled Order", not "Order" + "Cancelled Order".
  Example: "program committee member" -> "Program Committee Member".
- Both sides of a relationship are classes. If the question links entity A to
  entity B, list A AND B.
  Example: "participants in a social event" -> classes: ["Participant", "Social Event"].
  Example: "papers given to reviewers" -> classes: ["Paper", "Reviewer"].
- When the question identifies an entity by an activity or state, name the
  entity, never the action. Derive the noun that denotes the one who performs
  or holds it, and use that noun as the class label. Apply this whether the
  word appears as a verb, a participle, or an adjective, and never place such
  a word in objectProperties.
- An adjective describing an entity (a state, status or quality) belongs INSIDE
  the class label, not in objectProperties. This holds wherever the adjective
  sits: "<adj> X" and "X that were <adj>" and "X that are <adj>" all mean the
  same class.
  Example: "cancelled orders" -> classes: ["Cancelled Order"].
  Example: "orders that were cancelled" -> classes: ["Cancelled Order"].
- Keep class labels as natural multi-word phrases; do not fuse words into tokens
  like "CancelledOrder".
- Use domain vocabulary that resembles ontology labels; do not genericize.
- Include a term ONLY if it is present in, or directly required by, the question.
  When unsure, leave it out rather than guessing. Do not invent relationships.
- Distinguish a RESOURCE from its literal content:
  * "each paper with its abstract" asks for Paper and Abstract resources:
    classes: ["Paper", "Abstract"], objectProperties: ["has abstract"],
    dataTypeProperties: [].
  * "the abstract text of each paper" asks for a literal value:
    classes: ["Paper"], objectProperties: [],
    dataTypeProperties: ["abstract text"].
  * "each paper with its title" asks for a literal title:
    classes: ["Paper"], dataTypeProperties: ["title"].
- A noun introduced as "its X", "the X belonging to Y", or "X of Y" is a
  resource when X denotes an identifiable domain object. It is a datatype
  property only when the question asks for its text, value, date, number, or
  other literal representation.
- Every key must always be present; use an empty array when nothing applies.
- Output ONLY the JSON object. No prose, no markdown, no code fences.
- Distinguish entities from their literal attributes by meaning, not possessive wording; “its X” or “their X” does not imply that X is a resource.
Format:
{
  "classes": [],
  "objectProperties": [],
  "dataTypeProperties": [],
  "literals": []
}

Worked example (unrelated domain, shown only to illustrate the format and how
to split concepts — do not reuse its vocabulary):
Question: "which sellers were verified, and which orders did they ship in 2021"
{
  "classes": ["Verified Seller", "Order"],
  "objectProperties": ["shipped"],
  "dataTypeProperties": ["year"],
  "literals": ["2021"]
}
Note how "verified sellers" gives only "Verified Seller". Plain "Seller" is not
added, because the specific label already covers it. Also note it stays two
words and is never fused into one token like "VerifiedSeller".
""".strip()

_REQUIRED_KEYS = ("classes", "objectProperties", "dataTypeProperties", "literals")


def _parse_json(content: str) -> dict:
    """Parse model output into the expected schema, tolerating code fences."""
    text = content.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # Fall back to the first {...} block if there is stray surrounding text.
    if not text.lstrip().startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    data = json.loads(text)
    # Guarantee all keys exist so downstream agents never KeyError.
    for key in _REQUIRED_KEYS:
        data.setdefault(key, [])
    return data


def extract_entities(question: str):
    user_message = f'Question: "{question}"'
    content = None
    try:
        content = run_agent(
            "agent1_extractor",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
        )
        return _parse_json(content)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON from model", "raw_response": content}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print(extract_entities("what are rejected conference papers and their authors who live in france?"))
