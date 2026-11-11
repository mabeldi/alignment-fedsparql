import io
from typing import List, Dict
from urllib.request import Request, urlopen

from rdflib import ConjunctiveGraph

# Global conjunctive graph instance to store all ontologies
store = ConjunctiveGraph()

def load_ontologies(ontology_infos: List[Dict[str, str]]):
    """
    Load multiple OWL/RDF ontology files into named graphs inside a ConjunctiveGraph.
    """
    for info in ontology_infos:
        path = info.get("path")
        public_id = info.get("id")
        print(f"🔁 Loading ontology: {path} into graph: {public_id}")

        try:
            if path.startswith("http"):
                # ✅ Add browser-like headers
                req = Request(
                    path,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urlopen(req) as response:
                    data = response.read()
                    store.get_context(public_id).parse(io.BytesIO(data), format="xml")
            else:
                # ✅ Local file
                store.get_context(public_id).parse(path, format="xml")

            print(f"✅ Loaded {public_id}")

        except Exception as e:
            print(f"❌ Error loading {path}: {e}")

    

def get_graph() -> ConjunctiveGraph:
    """
    Returns the in-memory RDFLib ConjunctiveGraph with named graphs.
    """
    return store