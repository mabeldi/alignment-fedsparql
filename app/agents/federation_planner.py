"""
Federation planner (deterministic — no LLM).

Given the finalized class paths, look up the cross-ontology CLASS correspondences
that connect the classes involved to other *loaded* ontologies, and format them
for Agent 7. Agent 7 then decides how to weave them in (UNION for the same
concept expressed differently, JOIN/OPTIONAL for complementary data).

We only retrieve here; we do not decide union-vs-join — that is the LLM's job.
"""

from typing import Dict, List, Any, Optional, Set
from tools.alignment_store import get_alignment_store, Mapping, source_of, local_name
from tools.prompt_format import format_alignment_grouped

# How many aligned counterparts a single retrieved class may pull in. Caps the
# extra graph traversals (Agent 3) and LLM path-filter calls (Agent 6).
MAX_ALIGNED_PER_CLASS = 5

# Alignment rendering strategy for the prompt.
#   True  = grouped by ontology pair, deduplicated, endpoints stated once per pair
#           (tools/prompt_format.format_alignment_grouped).
#   False = original flat "* one bullet per mapping" list, which repeated both
#           full URIs and both endpoint URLs on every single correspondence.
# Same switch pattern as USE_CONNECTOR_PATHS in main.py, so the previous
# rendering can be restored instantly.
USE_GROUPED_ALIGNMENT = True

# What to do with a source that only ever appeared as the far side of an
# alignment mapping — it never held a path, so nothing scored it.
#   "service"  = it is a participant and gets its own SERVICE block, built from
#                the alignment triples (current behaviour).
#   "metadata" = the mapping still informs Agent 7's joins, but the source is not
#                offered as a queryable endpoint.
ALIGNMENT_ONLY_SOURCES = "service"


def expand_matches_with_alignment(matches: Dict[str, List[Dict[str, Any]]],
                                  loaded_sources: Set[str],
                                  max_per_class: int = MAX_ALIGNED_PER_CLASS
                                  ) -> Dict[str, List[Dict[str, Any]]]:
    """Add the aligned counterpart classes of the retrieved classes as candidates.

    Retrieval (Agent 1 + vector search) is lexical: a class is only found when the
    question happens to use words close to that ontology's own vocabulary. When two
    ontologies name the same concept differently the counterpart is never retrieved,
    so Agent 3 builds no paths for it and Agent 7 gets a correspondence pointing at
    a class whose relationships it knows nothing about. Worse, surface similarity
    can retrieve a same-named class that the alignment says is a DIFFERENT concept.

    The alignment already states these cross-ontology identities, so we apply it
    here as a retrieval operator instead of only as a hint at generation time. The
    injected classes go through normal path extraction, and the LLM path filter
    (Agent 6) still prunes whatever turns out to be irrelevant.
    """
    store = get_alignment_store()

    # Every class already on the table — expansion must never duplicate these.
    known = {
        (t.get("payload") or {}).get("uri") or t.get("uri")
        for terms in matches.values() for t in terms
        if ((t.get("payload") or {}).get("type") or t.get("type")) == "Class"
    }
    known.discard(None)

    added: List[str] = []
    for keyword, terms in matches.items():
        additions = []
        for t in list(terms):
            payload = t.get("payload") or {}
            if payload.get("type") != "Class":
                continue
            origin = payload.get("uri")
            if not origin:
                continue

            budget = max_per_class
            for uri, mapping in store.counterparts(origin, loaded_sources):
                if budget <= 0:
                    break
                if uri in known:
                    continue
                known.add(uri)
                budget -= 1
                label = local_name(uri)
                additions.append({
                    "label": label,
                    "uri": uri,
                    "type": "Class",
                    # 0.0 so an injected class can never win base-source selection,
                    # which must stay anchored on a genuinely retrieved match.
                    "score": 0.0,
                    "payload": {"type": "Class", "uri": uri,
                                "label": label, "source": source_of(uri)},
                    "expanded_from": origin,          # provenance: alignment, not retrieval
                    "via_alignment": mapping.render(),
                })
                added.append(f"{local_name(origin)} -> {label}")
        terms.extend(additions)

    if added:
        print(f"[federation] alignment-expanded {len(added)} class(es): {', '.join(added)}")
    return matches


def plan_federation(finalized_paths: List[Dict[str, Any]],
                    loaded_sources: Set[str]) -> List[Mapping]:
    """Return the class correspondences relevant to the finalized paths whose
    other side lives in a loaded ontology. `finalized_paths` items carry a
    "class" URI (the anchor class of each path block)."""
    store = get_alignment_store()
    class_uris = {p["class"] for p in finalized_paths if p.get("class")}
    return store.find_for_classes(class_uris, loaded_sources)


def participating_sources(finalized_paths: List[Dict[str, Any]],
                          mappings: List[Mapping],
                          cut_sources: Optional[Set[str]] = None) -> Set[str]:
    """The ontologies the generated query may actually mention.

    Two ways in, and only these two:
      * the source owns a path that survived the filters, so it has a schema
        block in the prompt;
      * a surviving class corresponds to one of its classes, so the alignment
        gives ready-made triples for it even without a schema block — dropping
        it would lose the instances only that endpoint holds. Whether that
        second way counts at all is `ALIGNMENT_ONLY_SOURCES`.

    A source with neither is noise: naming it as a participant invites a SERVICE
    block the model has no vocabulary for and has to invent.

    `cut_sources` are sources that held paths and lost every one. They are
    excluded whatever the alignment says: the alignment here is dense enough
    that it would otherwise re-admit almost everything just removed.
    """
    sources = {p["source"] for p in finalized_paths if p.get("source")}
    if ALIGNMENT_ONLY_SOURCES == "service":
        for m in mappings:
            sources.update(s for s in (m.entity1_source, m.target_source) if s)
    return sources - (cut_sources or set())


def format_alignment_block(mappings: List[Mapping], endpoints: Dict[str, str] = None,
                           base_source: str = None) -> str:
    """Prompt-ready alignment section (dispatches on USE_GROUPED_ALIGNMENT).

    In both renderings we give BOTH sides as ready-made SPARQL triple patterns
    (rendered deterministically, not by the LLM) plus the endpoint each side must
    be queried at. The generator only has to place them into SERVICE blocks and
    decide UNION vs JOIN — it never re-derives the triples itself.

    `base_source` only orders the output (anchor's pairs first); the flat
    rendering ignores it.
    """
    if USE_GROUPED_ALIGNMENT:
        return format_alignment_grouped(mappings, endpoints, base_source)
    return _format_alignment_flat(mappings, endpoints)


def _format_alignment_flat(mappings: List[Mapping], endpoints: Dict[str, str] = None) -> str:
    """ORIGINAL rendering — one flat `*` bullet per mapping, both sides inline with
    full URIs and both endpoints repeated. Kept behind USE_GROUPED_ALIGNMENT for
    an instant revert / A-B comparison."""
    if not mappings:
        return "No cross-ontology class correspondences found for these classes."

    endpoints = endpoints or {}
    blocks = []
    for m in mappings:
        e1_src, t_src = m.entity1_source, m.target_source
        e1_ep = endpoints.get(e1_src, e1_src)
        t_ep = endpoints.get(t_src, t_src)
        blocks.append(
            f"* {m.render(uris=True)}\n"
            f"    side A [{e1_src}] endpoint <{e1_ep}>:\n"
            f"        ?x a <{m.entity1}> .\n"
            f"    side B [{t_src}] endpoint <{t_ep}>:\n"
            + "\n".join(f"        {ln}" for ln in m.to_sparql_fragment().splitlines())
        )
    return "\n".join(blocks)
