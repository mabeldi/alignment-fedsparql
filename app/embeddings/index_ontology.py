from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
from rdflib import Graph, RDF, RDFS, OWL, URIRef
import os
import requests
load_dotenv()


import sys
import uuid

# Run as a standalone script (`python app/embeddings/index_ontology.py`), so the
# app package is not on the path yet. Added here rather than re-implementing
# source_of: a second copy of that rule drifting from the canonical one is what
# put the wrong source on two ontologies in the first place.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.alignment_store import source_of
# Same reasoning as source_of above: find_class_paths already resolves
# owl:unionOf domains/ranges on the graph side, and a second copy of that rule
# here would be free to drift from it.
from tools.find_class_paths import _expand_class_expression


# Local Ollama serves an OpenAI-compatible API, so we reuse the OpenAI client
# and just point it at the Ollama endpoint. No API key is required.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
print(os.getenv("QDRANT_URL"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL")
)
# ,
#     api_key=os.getenv("QDRANT_API_KEY")
COLLECTION = "ontology_terms"
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")

def embed_text(text):
    res = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return res.data[0].embedding

def init_collection():
    # Derive the vector size from the model itself so it stays correct if the
    # embedding model changes (qwen3-embedding:4b -> 2560 dims).
    dim = len(embed_text("dimension probe"))
    collections = qdrant.get_collections().collections
    existing = next((c for c in collections if c.name == COLLECTION), None)

    if existing is None:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION}' with dim {dim}.")
        return

    # get_collections() returns CollectionDescription objects, which carry the
    # name and nothing else; the config only comes from a per-collection call.
    # Reading existing.config here raised AttributeError and aborted every run
    # against an already-created collection.
    current_size = qdrant.get_collection(COLLECTION).config.params.vectors.size
    if current_size != dim:
        print(f"Collection '{COLLECTION}' dim {current_size} != {dim}; recreating.")
        qdrant.delete_collection(collection_name=COLLECTION)
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{COLLECTION}' already exists with dim {dim}. Skipping creation.")


# def index_ontology(ontology_path):
#     g = Graph()
#     g.parse(ontology_path)

#     points = []
#     idx = 0

#     for s, p, o in g.triples((None, RDFS.label, None)):
#         label = str(o)
#         comment = next(g.objects(s, RDFS.comment), None)
#         description = f"{label} {uri.split('/')[-1]} {type} {comment or ''}"
#         vector = embed_text(description)

#         points.append(
#             PointStruct(
#                 id=idx,
#                 vector=vector,
#                 payload={
#                     "label": label,
#                     "uri": str(s),
#                     "comment": str(comment) if comment else "",
#                     "type": "ClassOrProperty"
#                 }
#             )
#         )
#         idx += 1

#     qdrant.upsert(collection_name=COLLECTION, points=points)
#     print(f"Inserted {len(points)} ontology terms.")




def get_local_name(uri):
    return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]

def get_label(graph, subject):
    for _, _, label in graph.triples((subject, RDFS.label, None)):
        return str(label)
    return get_local_name(subject)  # fallback to URI name




def download_rdf(url, timeout=60):
    """Fetch RDF content from a URL and return the raw bytes."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def load_graph(source, fmt=None):
    """Build an rdflib Graph from either a URL or a local file path.

    OAEI conference ontologies are RDF/XML, so we default to 'xml' for URLs.
    """
    g = Graph()
    if str(source).startswith(("http://", "https://")):
        data = download_rdf(source)
        g.parse(data=data, format=fmt or "xml")
    else:
        g.parse(source, format=fmt)
    return g


def _named(node):
    """The node if it is a real URI, else None.

    OWL uses blank nodes for anonymous class expressions (restrictions, unions,
    intersections), and they are typed owl:Class like everything else. Indexing
    them produced terms whose 'uri' was a generated id such as
    'n7372e572dda54117...' — no namespace, so source_of read that id AS the
    ontology name and each one became its own source. 65 such nodes across the
    five ontologies; 27 were already in the index.
    """
    return node if isinstance(node, URIRef) else None


def _class_ends(graph, prop, predicate):
    """Every named class an rdfs:domain / rdfs:range denotes, as a list of IRIs.

    A domain or range is not always a plain IRI. OWL writes "one of several
    classes" as an anonymous union, and rdflib hands back the blank node:

        <owl:ObjectProperty rdf:about="#relatesTo">
          <rdfs:range>
            <owl:Class><owl:unionOf rdf:parseType="Collection">
              <owl:Class rdf:about="#Conference"/>
              <owl:Class rdf:about="#Person"/>
            </owl:unionOf></owl:Class>
          </rdfs:range>

    The previous code read the end with g.value + _named, so every such node
    became None. Across the five OAEI ontologies that is 40 ends dropped, 38 of
    them unions, discarding 94 named classes — confof:hasCity (Person,
    Organization), edas:relatesTo, ekaw:organisedBy and the rest. Agent 4 both
    shows these ends to the model and rebuilds Class terms from them, so a
    dropped end costs twice.

    Reading with graph.objects rather than graph.value also keeps every end when
    a property declares more than one rdfs:domain triple.

    The 2 remaining blank ends are owl:oneOf enumerations of individuals, not
    classes; they stay unresolved and yield an empty list.
    """
    out = []
    for node in graph.objects(prop, predicate):
        for uri in _expand_class_expression(graph, node):
            if uri not in out:
                out.append(uri)
    return out


def parse_rdf_to_json(source, fmt=None):
    g = load_graph(source, fmt=fmt)

    result = []

    # Extract Classes
    for s in g.subjects(RDF.type, OWL.Class):
        if _named(s) is None:
            continue
        result.append({
            "type": "Class",
            "uri": str(s),
            "name": get_local_name(s),
            "label": get_label(g, s)
        })

    # Extract ObjectProperties / DatatypeProperties. Ends go through
    # _class_ends, which resolves owl:unionOf instead of dropping it.
    #
    # BOTH SHAPES ARE STORED. `domains`/`ranges` (plus their *Uris) carry every
    # member of a union. The singular `domain`/`range`/`domainUri`/`rangeUri`
    # keep their old string-or-None type and hold the first member, because
    # agent4 renders them into a line and rebuilds Class terms from them by
    # name; handing those readers a list would break them. A union end that used
    # to be None is now at least its first class, with no change on their side.
    for prop_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in g.subjects(RDF.type, prop_type):
            if _named(s) is None:
                continue
            is_object = prop_type == OWL.ObjectProperty
            domain_uris = _class_ends(g, s, RDFS.domain)
            domain_names = [get_local_name(u) for u in domain_uris]

            if is_object:
                range_uris = _class_ends(g, s, RDFS.range)
                range_names = [get_local_name(u) for u in range_uris]
            else:
                # A datatype range is a datatype IRI (xsd:string), not a class:
                # never expanded, and shown in full rather than by local name.
                dt = _named(g.value(subject=s, predicate=RDFS.range))
                range_uris = [str(dt)] if dt else []
                range_names = list(range_uris)

            result.append({
                "type": "ObjectProperty" if is_object else "DatatypeProperty",
                "name": get_local_name(s),
                "uri": str(s),
                "label": get_label(g, s),
                "domain": domain_names[0] if domain_names else None,
                "range": range_names[0] if range_names else None,
                "domainUri": domain_uris[0] if domain_uris else None,
                "rangeUri": range_uris[0] if range_uris else None,
                "domains": domain_names,
                "ranges": range_names,
                "domainUris": domain_uris,
                "rangeUris": range_uris,
            })

    return result



def build_embedding_text(entry):
    """Text embedded for one ontology term: its NAME, and nothing else.

    This used to be padded — "title DatatypeProperty title domain Paper range
    http://www.w3.org/2001/XMLSchema#string" — on the reasoning that the query
    side wrapped its queries in an instruction template, so a bare document
    would under-match a rich instructed query. That template was later removed
    (see agent2_vector_lookup.embed_query, which measured the prefix as actively
    harmful) and the documents were never re-tuned. The two sides have disagreed
    since.

    Padding can only ever DILUTE an exact match, and it dilutes the exact match
    hardest — a term whose name already IS the query has nothing to gain from
    extra words and everything to lose. That inverts rankings. Measured over all
    544 indexed terms, query "title":

        padded      confOf:hasTitle 0.7365   confOf:has_title 0.7186   cmt:title 0.6932
        name only   cmt:title 1.0000         confOf:has_title 0.8525   confOf:hasTitle 0.7910

    `cmt:title` is an exact name match and the padding pushed it to third, under
    the 0.7 score cutoff in main.py — so the one real title property in cmt was
    dropped before any agent saw it.

    Name-only won on every keyword tested (title, Paper, author, review,
    decision, acceptance, reviewer, conference, "written by", "submission
    deadline"). A middle option — name + type, dropping only domain/range — was
    the WORST of the three: the word "ObjectProperty"/"DatatypeProperty" pulls
    properties above classes, so "review" returned ekaw:hasReview instead of
    ekaw:Review and "conference" returned cmt:hasConferenceMember instead of
    cmt:Conference. The type belongs in the payload, not in the vector.

    NOTHING IS LOST. type/domain/range/domainUri/rangeUri are all still stored
    in the Qdrant PAYLOAD (index_to_qdrant passes the whole entry), which is a
    separate argument from the vector. Agent 4 reads them from there, the type
    filter in search_qdrant filters on the payload field, and the value-edge
    extraction reads domain/range from the rdflib graph directly. This function
    decides only how a term RANKS, never what comes back with it.
    """
    return str(entry.get("label") or entry.get("name") or "")


# Stable namespace for point ids. Any fixed UUID works; it only has to stay the
# same across runs so a term keeps its id.
_POINT_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def index_to_qdrant(ontology_json):
    """Embed and upsert one ontology's terms.

    `source` comes from each term's OWN URI, never from a label passed in
    alongside the file. The list below used to carry the name by hand and had
    two entries mislabelled, so every term of two ontologies was stored under
    another ontology's name — which then decided how Agent 4 groups terms, which
    graph Agent 3 searches for paths, and how sources are ranked.

    Ids are derived from the URI, so re-indexing REPLACES a term. With random
    ids every run appended a second copy of the whole ontology instead.
    """
    points = []
    for entry in ontology_json:
        uri = entry.get("uri") or ""
        embedding_input = build_embedding_text(entry)
        vector = embed_text(embedding_input)

        points.append(PointStruct(
            id=str(uuid.uuid5(_POINT_NS, uri)),
            vector=vector,
            payload={**entry, "source": source_of(uri)}
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Inserted {len(points)} entries into Qdrant.")


# Ontologies to download and index. MUST stay in step with ONTOLOGIES in
# main.py: a term indexed here but not loaded there is retrievable yet has no
# graph, so it wins a top-k slot and then yields no path at all.
ONTOLOGY_URLS = [
    "http://oaei.ontologymatching.org/2022/conference/data/confof.owl",
    "http://oaei.ontologymatching.org/2022/conference/data/cmt.owl",
    "http://oaei.ontologymatching.org/2022/conference/data/ekaw.owl",
    "http://oaei.ontologymatching.org/2022/conference/data/conference.owl",
    "http://oaei.ontologymatching.org/2022/conference/data/edas.owl",
]

def recreate_collection():
    """Drop the collection and start clean. DESTRUCTIVE, opt-in only.

    Needed once after the id scheme changed: the points written with random ids
    cannot be overwritten by the URI-derived ones, so an ordinary re-index leaves
    both copies in place — including the ones carrying the wrong source.
    """
    dim = len(embed_text("dimension probe"))
    qdrant.delete_collection(collection_name=COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Recreated collection '{COLLECTION}' with dim {dim}.")


if __name__ == "__main__":
    if "--recreate" in sys.argv:
        recreate_collection()
    else:
        init_collection()

    for url in ONTOLOGY_URLS:
        print(f"\nProcessing {url}")
        ontology_items = parse_rdf_to_json(url)
        sources = {source_of(i.get("uri") or "") for i in ontology_items}
        print(f"  Parsed {len(ontology_items)} terms, source(s): {', '.join(sorted(sources))}")
        index_to_qdrant(ontology_items)
