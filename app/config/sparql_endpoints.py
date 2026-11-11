"""One Fuseki SPARQL endpoint per ontology.

Kept out of the agent modules so anything needing an endpoint URL can import it
without pulling in the LLM client stack.
"""

import os
from typing import Dict

FUSEKI_BASE_URL = os.getenv("FUSEKI_BASE_URL", "http://192.168.100.45:3030").rstrip("/")

# Add other sources here. Must match ONTOLOGIES in main.py.
SOURCES = ("cmt", "confof", "ekaw", "conference", "edas")

SPARQL_ENDPOINTS: Dict[str, str] = {
    source: f"{FUSEKI_BASE_URL}/{source}/sparql"
    for source in SOURCES
}
