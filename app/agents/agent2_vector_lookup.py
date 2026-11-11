import os
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from core.ontology_loader import get_graph
from tools.find_class_paths import usable_property_uris

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    # The client's 5s default is too tight for this store: a cold query against
    # it intermittently timed out or dropped the connection, and a lost lookup
    # silently costs the question every term that keyword would have found.
    timeout=60
)

COLLECTION = "ontology_terms"
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")

def embed_text(text):
    res = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return res.data[0].embedding

def embed_query(text):
    return embed_text(text)

# Minimum similarity a term must reach to be offered to Agent 4, replacing the
# fixed top_k=5 this used to take.
#
# A fixed N cannot adapt to how many terms a keyword actually matches. Measured
# on "Who authored each invited-talk abstract?": the keyword "Invited Talk
# Abstract" spent all five global slots on ekaw classes whose names spell the
# question's words, so `conference#Abstract` (0.7039) and
# `conference#has_an_abstract` (0.7033) never reached Agent 4 and the conference
# branch of the query answered about the talk instead of the abstract. Meanwhile
# the easy keyword "Author" spent the same five slots on near-duplicates that
# scored 0.80+. The right cut is a score, not a count.
#
# 0.70 is the HIGHEST cut that still covers every term that question's gold
# query needs — measured, not guessed; at 0.71 those two terms are the first
# things lost. Fitted to ONE question so far, so treat it as provisional and
# re-measure across the validation set before trusting it generally.
MIN_SCORE = float(os.getenv("AGENT2_MIN_SCORE", "0.70"))

# Ceiling on what a single keyword may contribute, so a vague keyword matching
# half the collection cannot flood Agent 4's prompt on its own. At MIN_SCORE the
# worst keyword measured returned 29 terms, so this is headroom, not a working
# limit — if it ever binds, the threshold is the thing to fix, not this.
MAX_HITS = 60

# How many hits to pull for the empty-answer fallback below. A property dropped
# by `_is_dead_property` is dropped from the answer too, so spare hits let the
# fallback reach past a dead one to the next real candidate.
OVERFETCH = 4


def _is_dead_property(payload):
    """True for a property this ontology can never build a path step from.

    Its two ends are what a path is made of, and `usable_property_uris` reports
    which properties the index managed to give ends to. One with none is a name
    and nothing else — it cannot reach Agent 7, so the only thing it can do
    between here and there is occupy a retrieval slot and prompt lines.

    Fail-OPEN: anything that cannot be checked (a term with no source, a graph
    still loading, an ontology not in this store) is kept. Dropping a term
    because we could not look it up would be worse than the noise.
    """
    if payload.get("type") not in ("ObjectProperty", "DatatypeProperty"):
        return False
    source, uri = payload.get("source"), payload.get("uri")
    if not source or not uri:
        return False
    try:
        graph = get_graph().get_context("http://" + source)
    except Exception:
        return False
    if len(graph) == 0:
        return False
    return uri not in usable_property_uris(graph)


def _as_result(hit):
    payload = hit.payload or {}
    return {
        "label": payload.get("label") or payload.get("name"),
        "uri": payload.get("uri"),
        "type": payload.get("type"),
        "name": payload.get("name"),
        "payload": payload,
        "score": hit.score,
    }


def search_qdrant(query_text, min_score=None, type_filter=None):
    min_score = MIN_SCORE if min_score is None else min_score
    vector = embed_query(query_text)

    filters = None
    if type_filter:
        filters = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=type_filter))]
        )

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=MAX_HITS,
        score_threshold=min_score,
        query_filter=filters
    ).points

    results = []
    dropped = []
    for h in hits:
        payload = h.payload or {}
        if _is_dead_property(payload):
            dropped.append(f"{payload.get('source')}:{payload.get('name')}")
            continue
        results.append(_as_result(h))

    if dropped:
        print(f"[agent2] {query_text!r}: dropped {len(dropped)} property/ies with no "
              f"domain or range — {', '.join(dropped)}")

    # Fail-OPEN when the threshold admits nothing. A score cut has a failure the
    # old top_k did not: a keyword whose best match sits below MIN_SCORE returns
    # NOTHING, and a concept with no candidates at all cannot be recovered by any
    # later agent — it simply leaves the query. Handing over the single best term
    # and letting Agent 4 judge it is the same bargain the old top_k made.
    if not results:
        spare = qdrant.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=OVERFETCH,
            query_filter=filters
        ).points
        for h in spare:
            if _is_dead_property(h.payload or {}):
                continue
            print(f"[agent2] ⚠ {query_text!r}: nothing reached {min_score:.2f} — "
                  f"falling back to best match "
                  f"{(h.payload or {}).get('source')}:{(h.payload or {}).get('name')} "
                  f"({h.score:.4f})")
            results.append(_as_result(h))
            break

    print(f"[agent2] {query_text!r}: {len(results)} term(s) >= {min_score:.2f}"
          + (f" (top {results[0]['score']:.4f})" if results else ""))
    return results

def lookup_concepts(extracted):
    keywords = extracted.get("objectProperties", []) + extracted.get("literals", [])  + extracted.get("classes", []) + extracted.get("dataTypeProperties", [])
    
    results = {}
    for kw in keywords:
        results[kw] = search_qdrant(kw)

    return results
