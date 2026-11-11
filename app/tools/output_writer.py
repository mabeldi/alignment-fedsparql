"""
Debug dumps of what the pipeline actually hands to Agent 7.

The path payload and the alignment payload are the two things that decide whether
the generated SPARQL is right, and both are far too large to read off the console
(the old path rendering was ~440k characters). So we write them to disk, in the
same shape the prompt sees:

    paths_agent5_all_<src>.txt      Agent 5 — every path the graph walk found
    paths_agent6_kept_<src>.txt     Agent 6 — the filtered subset Agent 7 is given
    paths_agent6_dropped_<src>.txt  what Agent 6 removed (and which relations that
                                    cost us entirely)
    alignment.txt                   the cross-ontology correspondences
    prompt_<agent>_<n>.txt          every message list actually sent to an LLM

The prompt dumps answer a different question from the path dumps: the path files
show what the pipeline SELECTED, the prompt files show what the model was
literally asked. The two differ, because between them sit the rendering and the
instructions — and a path that is present but unwalkable as rendered looks fine
in the path dump and wrong in the prompt.

Both path stages are dumped, not just the one that reaches the prompt: Agent 6 is
an LLM pruning the walk, and without the before/after there is no way to tell a
path the filter discarded from a path the walk never produced.

Deliberately a separate module rather than a few `open()` calls inside the agents:
the agents stay pure functions of their inputs, and file writing never becomes a
step the request path depends on. Every failure here is swallowed — a debug dump
must not be able to fail a query.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.alignment_store import Mapping, local_name
from tools.prompt_format import (build_prefix_map, collect_edges, curie,
                                 format_alignment_grouped, render_edge_graph)

# Default target: the folder next to the agents, so the dumps sit beside the code
# that produced them. Created on demand.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "agents" / "output"

# Set False to stop writing the dumps entirely (they are pure debug output).
WRITE_DEBUG_OUTPUT = True


def _write(path: Path, text: str) -> Optional[Path]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # utf-8 explicitly: the default console codepage on Windows is cp1252 and
        # would refuse anything outside it.
        path.write_text(text, encoding="utf-8")
        return path
    except OSError as exc:
        print(f"[output_writer] could not write {path}: {exc}")
        return None


# --- prompt dumps -----------------------------------------------------------
#
# Written from llm_router rather than from the four agents, so the file holds the
# message list as SENT: nothing is re-rendered for the dump, and an agent added
# later is covered without being told to opt in.

_PROMPT_PREFIX = "prompt_"

# Agents 4 and 6 call once per keyword / per ontology, so one file per agent
# would keep only the last call. Numbered per agent instead, in call order.
_prompt_counts: Dict[str, int] = {}
_prompt_question = ""
# Agent 6 issues its calls concurrently; two of them incrementing the same
# counter must not land on the same file name.
_prompt_lock = threading.Lock()


def reset_prompt_files(question: str = "", out_dir: Path = None) -> None:
    """Start a new question: delete the previous dumps, restart numbering.
    The delete matters.
    """
    global _prompt_question
    with _prompt_lock:
        _prompt_counts.clear()
        _prompt_question = question or ""
    if not WRITE_DEBUG_OUTPUT:
        return
    folder = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    try:
        for stale in folder.glob(f"{_PROMPT_PREFIX}*.txt"):
            stale.unlink()
    except OSError as exc:
        print(f"[output_writer] could not clear old prompt dumps: {exc}")


def write_prompt_file(agent_key: str,
                      messages: List[Dict[str, str]],
                      provider: str = "",
                      model: str = "",
                      out_dir: Path = None) -> Optional[Path]:
    """Dump one LLM call's messages, verbatim, to prompt_<agent>_<n>.txt."""
    if not WRITE_DEBUG_OUTPUT:
        return None

    with _prompt_lock:
        index = _prompt_counts.get(agent_key, 0) + 1
        _prompt_counts[agent_key] = index
        question = _prompt_question

    body: List[str] = []
    chars = 0
    for message in messages or []:
        role = message.get("role", "?")
        content = message.get("content", "") or ""
        chars += len(content)
        body.append(f"# ----- {role} message ({len(content)} chars) -----")
        body.append(content)
        body.append("")

    header = [
        f"# AGENT         : {agent_key}",
        f"# provider      : {provider or '(unknown)'}",
        f"# model         : {model or '(unknown)'}",
        f"# call          : #{index} for this agent, this question",
        f"# question      : {question or '(not recorded)'}",
        # Roughly 4 characters per token for English prose; the exact count is in
        # the token report. This is here to spot a prompt that has grown, not to
        # bill anyone.
        f"# size          : {chars} chars across {len(messages or [])} message(s)"
        f"  (~{chars // 4} tokens)",
        "",
    ]
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    return _write(out_dir / f"{_PROMPT_PREFIX}{agent_key}_{index}.txt",
                  "\n".join(header + body))


def write_path_stage_files(paths: List[Dict[str, Any]],
                           file_prefix: str,
                           stage_label: str,
                           base_source: Optional[str] = None,
                           endpoints: Optional[Dict[str, str]] = None,
                           out_dir: Path = None,
                           question: str = "") -> List[Path]:
    """Write ONE pipeline stage's class paths as an adjacency list, per ontology.

    `stage_label` goes in the header because the two stages look identical
    otherwise, and confusing them is the whole risk: stage 5 is every path the
    graph walk found, stage 6 is the subset that actually reaches Agent 7.

    For stage 6 the file body is exactly the section `format_edge_graph` puts in
    the prompt, so what you read on disk is what the model read — the point of the
    dump is lost if it is a different rendering.
    """
    if not WRITE_DEBUG_OUTPUT:
        return []
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    endpoints = endpoints or {}

    edges_by_source, entry_by_source = collect_edges(paths)
    # One render for all sources, then one file per section. Rendering per source
    # would restart the (global, anchor-seeded) variable naming and the files
    # would no longer match the prompt.
    prefix_header, sections = render_edge_graph(paths, base_source, endpoints)
    written: List[Path] = []

    for source, section in sections.items():
        subset = [p for p in paths if p.get("source") == source]
        # Each file repeats the shared PREFIX header so it is readable alone.
        body = "\n".join(prefix_header) + "\n\n" + section + "\n"

        n_edges = len(edges_by_source.get(source, ()))
        n_classes = len({c for e in edges_by_source.get(source, ()) for c in (e[0], e[2])})
        n_chains = sum(len(set(p.get("outgoing", []) + p.get("incoming", [])))
                       for p in subset)
        header = [
            f"# STAGE         : {stage_label}",
            f"# ontology      : {source}",
            f"# endpoint      : <{endpoints.get(source, source)}>",
            f"# question      : {question or '(not recorded)'}",
            f"# entry classes : {len(entry_by_source.get(source, ()))}",
            f"# flattened chains at this stage : {n_chains}",
            f"# distinct edges they describe    : {n_edges}  over {n_classes} classes",
            "",
        ]
        target = _write(out_dir / f"{file_prefix}_{source}.txt", "\n".join(header) + body)
        if target:
            written.append(target)
    return written


# --- Agent 6 filter diff ----------------------------------------------------

def _index_by_class(summaries: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, List[str]]]:
    """(source, class URI) -> {"outgoing": [chains], "incoming": [chains]}.

    Deduplicated, order preserved. Both stages carry the same shape, so the two
    can be diffed key by key.
    """
    out: Dict[tuple, Dict[str, List[str]]] = {}
    for item in summaries or []:
        key = (item.get("source"), item.get("class"))
        bucket = out.setdefault(key, {"outgoing": [], "incoming": []})
        for direction in ("outgoing", "incoming"):
            for chain in item.get(direction, []) or []:
                if chain not in bucket[direction]:
                    bucket[direction].append(chain)
    return out


def write_filter_diff_files(before: List[Dict[str, Any]],
                            after: List[Dict[str, Any]],
                            endpoints: Optional[Dict[str, str]] = None,
                            out_dir: Path = None,
                            question: str = "") -> List[Path]:
    """Report what the Agent 6 contextual filter REMOVED, one file per ontology.

    Agent 6 is an LLM deciding which paths are relevant, and today that decision
    is invisible: a good path silently disappearing looks exactly like a path that
    never existed. Same reasoning as `get_removed_matches` in main.py, which
    already exposes the Agent 4 match filter's removals in the API response.

    Two things beyond the raw per-class diff, both diagnostics you cannot get by
    eyeballing the kept set:

    * EDGES LOST ENTIRELY — a relation that survives in NO kept chain. Pruning a
      redundant longer walk is the filter working; removing the last chain that
      mentioned a relation removes that relation from the schema Agent 7 sees, and
      no amount of prompting can recover it.
    * CHAINS ONLY IN THE KEPT SET — Agent 6 returns chains as text, so it can
      reword one instead of selecting it. A rewritten chain no longer resolves
      (`path_edges` / `render_path` return None on an unknown label) and is
      dropped downstream in silence, so it must be visible here.
    """
    if not WRITE_DEBUG_OUTPUT:
        return []
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    endpoints = endpoints or {}

    idx_before, idx_after = _index_by_class(before), _index_by_class(after)
    edges_before, _ = collect_edges(before)
    edges_after, _ = collect_edges(after)

    sources = sorted({s for s, _ in idx_before} | {s for s, _ in idx_after})
    prefixes = build_prefix_map([c for _, c in idx_before] + [c for _, c in idx_after])
    written: List[Path] = []

    for source in sources:
        if not source:
            continue
        keys = sorted({k for k in idx_before if k[0] == source}
                      | {k for k in idx_after if k[0] == source},
                      key=lambda k: local_name(k[1] or ""))

        n_in = n_kept = n_dropped = n_new = 0
        body: List[str] = []
        for key in keys:
            b = idx_before.get(key, {"outgoing": [], "incoming": []})
            a = idx_after.get(key, {"outgoing": [], "incoming": []})
            for direction in ("outgoing", "incoming"):
                before_set, after_set = b[direction], a[direction]
                dropped = [c for c in before_set if c not in after_set]
                added = [c for c in after_set if c not in before_set]
                n_in += len(before_set)
                n_kept += len(before_set) - len(dropped)
                n_dropped += len(dropped)
                n_new += len(added)
                if not before_set and not after_set:
                    continue
                body.append(f"## {curie(key[1], prefixes)}  [{direction}]  "
                            f"in {len(before_set)} -> kept {len(after_set) - len(added)}"
                            f", dropped {len(dropped)}"
                            + (f", NOT IN INPUT {len(added)}" if added else ""))
                for chain in dropped:
                    body.append(f"  - {chain}")
                for chain in added:
                    body.append(f"  ?? only in kept set (reworded by the filter?): {chain}")
                body.append("")

        lost = sorted(set(edges_before.get(source, ())) - set(edges_after.get(source, ())),
                      key=lambda e: (local_name(e[0]), local_name(e[1])))
        lost_block: List[str] = []
        if lost:
            lost_block.append(f"# EDGES LOST ENTIRELY ({len(lost)}) — these relations survive in")
            lost_block.append("# no kept chain, so Agent 7 never sees them at all:")
            for s, p, o in lost:
                lost_block.append(f"    {curie(s, prefixes)}  {curie(p, prefixes)}  "
                                  f"{curie(o, prefixes)}")
        else:
            lost_block.append("# EDGES LOST ENTIRELY: none — every relation Agent 5 found still")
            lost_block.append("# appears in at least one kept chain.")
        lost_block.append("")

        header = [
            "# STAGE         : Agent 6 filter DIFF (what the contextual filter removed)",
            f"# ontology      : {source}",
            f"# endpoint      : <{endpoints.get(source, source)}>",
            f"# question      : {question or '(not recorded)'}",
            f"# chains in (Agent 5)   : {n_in}",
            f"# chains kept (Agent 6) : {n_kept}",
            f"# chains dropped        : {n_dropped}",
            f"# chains only in output : {n_new}   (should be 0 — see note below)",
            f"# distinct edges  {len(edges_before.get(source, ()))} -> "
            f"{len(edges_after.get(source, ()))}",
            "#",
            "# NOTATION. This file shows paths as Agent 5 FLATTENED them, which is the",
            "# form Agent 6 sees, so the arrows appear here (they do NOT appear in the",
            "# Agent 7 payload — that one is plain triples):",
            "#     A -[p]-> B    the triple is (A p B): p goes FROM A TO B",
            "#     A <-[p]- B    the edge is traversed BACKWARDS: the triple is (B p A),",
            "#                   p goes from B to A. It is not a typo and not the same",
            "#                   as -[p]->; reading it as a forward arrow inverts the",
            "#                   relationship. tools/path_sparql._parse_chain rejects any",
            "#                   other arrow shape, so a reworded arrow drops the path.",
            "# 'chains only in output' should be 0. Anything above 0 means Agent 6",
            "# returned a chain that was not in its input — it reworded one instead of",
            "# selecting it — and a reworded chain no longer resolves to URIs, so it is",
            "# silently discarded downstream. Those lines are marked '??' below.",
            "",
        ]
        target = _write(out_dir / f"paths_agent6_dropped_{source}.txt",
                        "\n".join(header + lost_block + body))
        if target:
            written.append(target)
    return written


def write_class_path_files(all_paths: List[Dict[str, Any]],
                           filtered_paths: Optional[List[Dict[str, Any]]] = None,
                           base_source: Optional[str] = None,
                           endpoints: Optional[Dict[str, str]] = None,
                           out_dir: Path = None,
                           question: str = "") -> List[Path]:
    """Dump BOTH path stages plus the filter diff, one set of files per ontology.

        paths_agent5_all_<src>.txt      every path the graph walk produced
        paths_agent6_kept_<src>.txt     the filtered subset Agent 7 is given
        paths_agent6_dropped_<src>.txt  what the filter removed, and why it matters

    `filtered_paths` may be omitted (or be the same object) when only one stage is
    available — then only the stage-5 files are written, rather than three files
    claiming a filter ran.
    """
    if not WRITE_DEBUG_OUTPUT:
        return []
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    written = write_path_stage_files(
        all_paths, "paths_agent5_all",
        "Agent 5 — summarize_class_paths (ALL paths, before the LLM filter)",
        base_source, endpoints, out_dir, question)

    if filtered_paths is not None and filtered_paths is not all_paths:
        written += write_path_stage_files(
            filtered_paths, "paths_agent6_kept",
            "Agent 6 — contextual_class_path_filter (KEPT; this is what Agent 7 sees)",
            base_source, endpoints, out_dir, question)
        written += write_filter_diff_files(all_paths, filtered_paths,
                                          endpoints, out_dir, question)

    if written:
        print(f"[output_writer] wrote {len(written)} path file(s) to {out_dir}")
    return written


def write_alignment_file(mappings: List[Mapping],
                         endpoints: Optional[Dict[str, str]] = None,
                         out_dir: Path = None,
                         question: str = "",
                         base_source: Optional[str] = None) -> Optional[Path]:
    """Write the cross-ontology alignment block to its own file."""
    if not WRITE_DEBUG_OUTPUT:
        return None
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    endpoints = endpoints or {}

    by_kind: Dict[str, int] = {}
    for m in mappings:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
    kinds = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "none"

    # The prefix header is not part of `format_alignment_grouped` (in the prompt it
    # is already declared by the path section), so the file states it itself —
    # otherwise the CURIEs in the file would be unreadable on their own.
    uris = [u for m in mappings for u in (m.entity1, *m.targets)]
    prefixes = build_prefix_map(uris)

    header = [
        f"# question       : {question or '(not recorded)'}",
        f"# correspondences: {len(mappings)}  ({kinds})",
        "#",
        "# Prefixes used below:",
        *[f"#   {p}: <{ns}>" for p, ns in sorted(prefixes.items())],
        "",
    ]
    body = format_alignment_grouped(mappings, endpoints, base_source)
    return _write(out_dir / "alignment.txt", "\n".join(header) + body)
