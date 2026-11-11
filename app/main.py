from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from agents.match_filter_agent import filter_matches_by_score
from agents.agent1_extractor_v2 import extract_entities
from agents.agent2_vector_lookup import lookup_concepts
from core.ontology_loader import load_ontologies
from contextlib import asynccontextmanager
from agents.ontology_ranking_agent import rank_ontology_sources, select_base_source, reanchor_base_source
from agents.agent3_path_connector import build_connector_paths_for_matches
from core.ontology_loader import get_graph
from tools.find_class_paths import build_index, index_stats, build_label_index
from agents.agent4_contextuel_match_relevance import filter_relevant_matches_with_llm
from agents.agent5_path_summarizer import summarize_class_paths
from agents.agent6_contextual_path_filter import contextual_class_path_filter
from agents.agent7_sparql_query_generator import generate_sparql_query, SPARQL_ENDPOINTS
from agents.federation_planner import plan_federation, format_alignment_block, expand_matches_with_alignment, participating_sources
from agents.path_selector import select_paths, stamp_source_completeness
from agents.agent8_source_selector import (apply_source_selection,
                                            build_relation_provider_plan)
from tools.alignment_store import get_alignment_store
from tools.path_validator import validate_paths_async
from tools.output_writer import (write_class_path_files, write_alignment_file,
                                 reset_prompt_files)
from tools.token_usage import reset as reset_token_usage, format_report as token_report, snapshot as token_snapshot
from fastapi.middleware.cors import CORSMiddleware
from config.pipeline_config import alignments_enabled
import time
import asyncio

# Ontologies participating in the pipeline. Sources are derived from the ids so
# indexing and federation stay dynamic if this list grows or shrinks.
ONTOLOGIES = [
    {"path": "http://oaei.ontologymatching.org/2022/conference/data/cmt.owl", "id": "http://cmt"},
    {"path": "http://oaei.ontologymatching.org/2022/conference/data/confof.owl", "id": "http://confof"},
    {"path": "http://oaei.ontologymatching.org/2022/conference/data/ekaw.owl", "id": "http://ekaw"},
    {"path": "http://oaei.ontologymatching.org/2022/conference/data/conference.owl", "id": "http://conference"},
    {"path": "http://oaei.ontologymatching.org/2022/conference/data/edas.owl", "id": "http://edas"},
]
LOADED_SOURCES = {o["id"].split("//")[-1].lower() for o in ONTOLOGIES}


class QuestionRequest(BaseModel):
    question: str

# Response model
class MatchedTerm(BaseModel):
    score: float
    payload: Dict[str, Any]


async def load_ontologies_async():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_ontologiesx)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ontologies...")
    asyncio.create_task(load_ontologies_async())  # Now it's truly async
    # IMPORTANT: This yields control back to FastAPI
    yield

    # Optionally do shutdown cleanup here
    print("Shutting down...")


def load_ontologiesx():
        load_ontologies(ONTOLOGIES)

        # Pre-build the O(1) property index for each participating ontology so
        # the first /query request is already fast (Agent 3 path extraction).
        for onto in ONTOLOGIES:
            start = time.perf_counter()
            graph = get_graph().get_context(onto["id"])
            build_index(graph)
            # label -> URI, so finalized paths can be rendered back into SPARQL
            labels = build_label_index(graph)
            out_classes, in_classes = index_stats(graph)
            elapsed = time.perf_counter() - start
            print(
                f"🗂️  Indexed {onto['id']} in {elapsed:.3f}s "
                f"(outgoing classes: {out_classes}, incoming classes: {in_classes}, "
                f"labels: {len(labels)})"
            )

        # Parse + index the cross-ontology alignment once (federation planner).
        store = get_alignment_store()
        print(f"🔗 Alignment loaded: {store.stats()}")


def print_elapsed_time(elapsed: float):
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"{minutes:02}:{seconds:04.1f}")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:4200",  # Angular dev server
    "http://127.0.0.1:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],              # allow all HTTP methods
    allow_headers=["*"],              # allow all headers
)


@app.post("/query", response_model=Any)
async def handle_question(req: QuestionRequest):
    alignment_mode = alignments_enabled()
    print(f"[pipeline] schema alignments: {'enabled' if alignment_mode else 'disabled'}")
    reset_token_usage()
    reset_prompt_files(req.question)

    # 🔍 Agent 1: extract entities
    start = time.perf_counter()
    extracted = extract_entities(req.question)
    print("#### AGENT 1:EXTRACTION Entities ######")
    print_elapsed_time(time.perf_counter() - start)
    #####################################################################


    # 🔍 Agent 2: Search for ontology terms
    start = time.perf_counter()
    matches = lookup_concepts(extracted)
    print("#### AGENT 2: lookup_concepts: #####")
    print_elapsed_time(time.perf_counter() - start)
    ######################################################################


    # Agent 3: Filter matches by score and context relevance (LLM)
    start = time.perf_counter()
    filtered_matches = filter_matches_by_score(matches, threshold=0.7)
    print("#### filtered_matches: #####")
    print_elapsed_time(time.perf_counter() - start)
    ######################################################################


    # Agent 5: context filtering of matches with LLM
    start = time.perf_counter()
    context_filtered_matches = await filter_relevant_matches_with_llm(question=req.question, matches=filtered_matches)
    print("#### context_filtered_matches: #####")
    context_removed_matches = get_removed_matches(filtered_matches, context_filtered_matches)
    print_elapsed_time(time.perf_counter() - start)
    ######################################################################


    # RANK Onotlogy 
    start = time.perf_counter()
    top_sources = rank_ontology_sources(context_filtered_matches, extracted)
    print("#### top_sources: #####")
    print_elapsed_time(time.perf_counter() - start)
    ######################################################################


    # ALignment Expander
    # Alignment-driven expansion: pull in the counterpart classes
    start = time.perf_counter()
    context_filtered_matches = expand_matches_with_alignment(context_filtered_matches, LOADED_SOURCES)
    print("#### alignment_expansion: #####")
    print_elapsed_time(time.perf_counter() - start)
    #######################################################################


    # PATH BUILDING
    start = time.perf_counter()
    participated_classes = {
        (item.get("payload") or {}).get("uri") or item.get("uri")
        for values in context_filtered_matches.values() for item in values
        if (item.get("payload") or {}).get("type") == "Class"
    }
    participated_classes.discard(None)
    participated_properties = collect_participated_properties(context_filtered_matches)
    print(f"#### participated_properties: {len(participated_properties)} "
          f"({sum(1 for p in participated_properties if p['kind'] == 'object')} object) ####")
    class_paths = build_connector_paths_for_matches(context_filtered_matches, participated_classes,
                                                        participated_properties)
    print_elapsed_time(time.perf_counter() - start)
    #######################################################################

    # SUMMARIZE Class Paths
    start = time.perf_counter()
    summarize_paths = summarize_class_paths(class_paths=class_paths, max_per_class=1000)
    print("#### summarize_paths: #####")
    print_elapsed_time(time.perf_counter() - start)
    ###########################################################################


    # CONTEXTUAL filtering of summarized class paths
    start = time.perf_counter()
    summarize_contextual_class_paths = await contextual_class_path_filter(req.question, summarize_paths)
    print("#### summarize_contextual_class_paths: #####")
    print_elapsed_time(time.perf_counter() - start)
    ############################################################################


    # Score every path against the question
    # cut the ones that do not answer it, note the sources left with nothing.
    start = time.perf_counter()
    selection = select_paths(req.question, summarize_contextual_class_paths,
                             extracted)
    selected_class_paths = selection.blocks
    print("#### selected_class_paths: #####")
    print_elapsed_time(time.perf_counter() - start)
    ############################################################################





    # Drop the paths no instance in the store follows (one ASK per chain).
    start = time.perf_counter()
    validated_class_paths = await validate_paths_async(selected_class_paths)
    # GTX Build candidate mappings before Agent 8, so it can retain complementary
    # GTX sources whose partial branches become complete through an alignment.
    # GTX Built here rather than after the stamp because completeness reads them:
    # GTX plan_federation only needs each block's anchor class URI, not the flag.
    candidate_alignment_mappings = plan_federation(
        validated_class_paths, LOADED_SOURCES)
    # GTX Validation can remove one link from a multi-chain answer graph, so
    # GTX calculate source completeness again from the paths that really exist.
    validated_class_paths = stamp_source_completeness(
        validated_class_paths, extracted, candidate_alignment_mappings)
    print("#### validated_class_paths: #####")
    print_elapsed_time(time.perf_counter() - start)
    ##################################################################################

    # SOURCE SELECTION
    # drops a source because another one covers the same thing more precisely.
    start = time.perf_counter()
    # GTX Select the best retained concrete property per requested relation;
    # GTX this preserves the Conference/EDAS clause split through generation.
    provider_plan = build_relation_provider_plan(
        extracted, context_filtered_matches, validated_class_paths,
        candidate_alignment_mappings)
    selected_class_paths, dropped_sources = await apply_source_selection(
        req.question, validated_class_paths, extracted,
        candidate_alignment_mappings, provider_plan)
    selection.cut_sources |= dropped_sources
    print("#### cross_source_selected_paths: #####")
    print_elapsed_time(time.perf_counter() - start)
    ###############################################################################

    


    # Pick the federation starting point 
    base_source = select_base_source(extracted, context_filtered_matches, top_sources)
    base_source = reanchor_base_source(base_source, selected_class_paths, top_sources,
                                       scores=selection.source_scores)
    print(f"#### base_source: {base_source} #####")
    ######################################################################################


    # Mapping: retrieve the cross-ontology class for Agent 7 to reason over.
    alignment_mappings = plan_federation(selected_class_paths, LOADED_SOURCES)
    alignment_block = format_alignment_block(alignment_mappings, SPARQL_ENDPOINTS, base_source)
    print(f"#### alignment mappings: {len(alignment_mappings)} #####")
    ############################################################################################
   
   
    # Which ontologies Agent 7 is told about: those holding a surviving path, plus
    # those a surviving class maps onto, minus the ones the cut emptied. A source
    # with neither contributes nothing the query can use, and listing it only
    # invites an invented SERVICE block.
    query_sources = participating_sources(selected_class_paths, alignment_mappings,
                                          selection.cut_sources)
    print(f"#### query sources: {', '.join(sorted(query_sources)) or '(none)'} "
          f"(of {', '.join(sorted(LOADED_SOURCES))}) #####")
    ############################################################################################


    # Dump the path payload (agents/output/) 
    write_class_path_files(summarize_paths, selected_class_paths,
                           base_source, SPARQL_ENDPOINTS, question=req.question)
    write_alignment_file(alignment_mappings, SPARQL_ENDPOINTS, question=req.question,
                         base_source=base_source)
    #########################################################################


    # QUERY GENERATION
    start = time.perf_counter()
    query = await generate_sparql_query(req.question, selected_class_paths, base_source,
                                        alignment_block, query_sources, extracted,
                                        provider_plan)
    print("#### query: #####")
    print_elapsed_time(time.perf_counter() - start)
    ######################################################################################


    # Where the prompt budget went, biggest spender first.
    print(token_report())

    return {
        "question": req.question,
        "alignments_enabled": alignment_mode,
        "extracted": extracted,
        "matches": filtered_matches,
        "context_filtered_matches": context_filtered_matches,
        "context_removed_matches": context_removed_matches,
        "top_sources": top_sources,
        "base_source": base_source,
        "class_paths": class_paths,
        "summarize_class_paths": summarize_paths,
        "summarize_contextual_class_paths": summarize_contextual_class_paths,
        "selected_class_paths": selected_class_paths,
        "validated_class_paths": validated_class_paths,
        "path_scores": [vars(r) for r in selection.rows],
        "source_scores": selection.source_scores,
        "provider_plan": provider_plan,
        "cut_sources": sorted(selection.cut_sources),
        "alignment": [m.render(uris=True) for m in alignment_mappings],
        "query": query,
        "token_usage": token_snapshot()
    }


def collect_participated_properties(matches: Dict[str, List[dict]]) -> List[Dict[str, Any]]:
    """The matched properties, flattened to one (domain, property, range) row each.

    WHY THIS EXISTS
    ---------------
    Agent 3 drops every non-Class term and rebuilds the relations from the GRAPH:
    for each pair of matched classes it BFS-searches the ontology for what
    connects them. So a property that retrieval matched and Agent 4 kept reaches
    the prompt only if the blind graph search happens to rediscover it — and
    `MAX_CONNECTORS_PER_PAIR` then truncates that search's output in discovery
    order, with no ranking at all. Measured: cmt:Conference -> cmt:Administrator
    has 14 connectors of length 1 and the cap forwards 8, so six are dropped by
    position. The matched property can be one of the six. "The property we
    matched is in the paths" was never a guarantee, only a coincidence.

    These rows are what makes it a guarantee: Agent 3 puts the seeded edge FIRST,
    so the truncation can no longer reach it.

    WHERE THE ENDS COME FROM
    ------------------------
    From `selected_ends`, stamped on the term by Agent 4 when it resolved the
    property's domain/range down to the class closest to the question (a union
    end names several classes, and only one of them is what the question asked
    about). A term without that key contributes nothing, so a run where Agent 4
    stamped nothing yields an empty list and Agent 3 behaves exactly as before.

    `kind` separates the two uses, because they are not seeded the same way:
      * "object" — a relation between two classes. Seeded as a connector.
      * "value"  — an attribute, whose `range` is a datatype and not a class, so
                   there is no pair to connect and nothing to walk. It needs no
                   seeding: `value_edges_for` lists every attribute declared on a
                   class as soon as that class is matched, and Agent 4 recovering
                   the attribute's DOMAIN is what puts it there. The row is still
                   emitted so the attribute's provenance is inspectable.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()
    for terms in matches.values():
        for term in terms:
            ends = term.get("selected_ends")
            if not ends:
                continue
            payload = term.get("payload") or {}
            ptype = payload.get("type")
            uri = payload.get("uri") or term.get("uri")
            source = payload.get("source")
            if not uri or not source:
                continue

            domains = [d for d in (ends.get("domain") or []) if d]
            if ptype == "ObjectProperty":
                # One row per (domain, range) combination. After Agent 4's
                # selection that is normally 1x1; it stays a product so the
                # `select_best_end=False` mode, which keeps every union member,
                # still produces usable rows instead of silently dropping them.
                pairs = [(d, r, "object")
                         for d in domains
                         for r in (ends.get("range") or []) if r]
            elif ptype == "DatatypeProperty":
                # The DECLARED datatype (xsd:string), carried as-is and never
                # traversed. None when the ontology declares no range, which
                # costs nothing here — nothing reads this field to walk it.
                datatype = payload.get("rangeUri") or payload.get("range")
                pairs = [(d, datatype, "value") for d in domains]
            else:
                continue

            for domain, range_, kind in pairs:
                key = (source, domain, uri, range_, kind)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"source": source, "domain": domain, "uri": uri,
                             "range": range_, "kind": kind})
    return rows


def get_removed_matches(original: Dict[str, List[dict]], filtered: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    removed = {}

    for keyword, original_terms in original.items():
        filtered_terms = filtered.get(keyword, [])

        # Extract URIs of filtered terms for comparison
        filtered_uris = {term["uri"] for term in filtered_terms if "uri" in term}

        # Keep terms from the original that were not in the filtered
        removed_terms = [term for term in original_terms if term.get("uri") not in filtered_uris]

        if removed_terms:
            removed[keyword] = removed_terms

    return removed
