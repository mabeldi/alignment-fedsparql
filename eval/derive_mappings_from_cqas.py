"""Extract the correspondences the CQA dataset implies but app/resources/mapping.rdf
does not contain.

THE IDEA
--------
A CQA is one question written once per ontology that can answer it. Two of those
SPARQL queries return the same thing from two different vocabularies, so the pair
IS a correspondence — stated as a query rewriting rather than as an EDOAL cell.
100 CQAs over 278 files therefore encode several hundred correspondences, and the
hand-built alignment has only some of them.

This reads every pair of files inside every CQA folder, turns the pair into a
correspondence, and asks whether mapping.rdf already has it.

WHAT "ALREADY HAS IT" MEANS
---------------------------
A cell counts as covering an implied correspondence when EVERY URI the cell names
appears on the matching side of the pair. That is stricter than "the two classes
are related somehow" and looser than string equality of the whole pattern — the
loose middle ground is reported separately as `partial`, because those are the
cases where we have a correspondence between the same two concepts but written in
a different shape, and a reviewer should look at them.

WHAT COMES OUT
--------------
  derived_mappings/derived_mappings.rdf     new cells, EDOAL, mergeable as-is
  derived_mappings/complex_unexpressed.json new ones EDOAL cannot state in this
                                            file's style, kept as the two queries
  derived_mappings/report.md                the counts

app/resources/mapping.rdf is never touched.

Run:  python eval/derive_mappings_from_cqas.py
      python eval/derive_mappings_from_cqas.py --check   # report only, write nothing
"""

import argparse
import itertools
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib.plugins.sparql import prepareQuery

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from tools.alignment_store import get_alignment_store, source_of   # noqa: E402

CQA_DIR = ROOT / "eval" / "cqa_source" / "CQAs"
OUT_DIR = ROOT / "eval" / "derived_mappings"

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
ONTO_NS = ("http://cmt#", "http://confOf#", "http://conference#",
           "http://edas#", "http://ekaw#")

# The CQA convention: ?s is the thing asked for, ?o the value asked about it.
ANSWER_VARS = ("s", "o")


# --- reading one CQA file ---------------------------------------------------

def _prefixes(text: str):
    return {m.group(1) or "": m.group(2)
            for m in re.finditer(r"PREFIX\s+([A-Za-z][\w.-]*)?\s*:\s*<([^>]+)>", text)}


class Side:
    """One ontology's answer to one CQA, reduced to what a correspondence needs."""

    def __init__(self, cqa: str, source: str, path: Path):
        self.cqa, self.source, self.path = cqa, source, path
        text = path.read_text(encoding="utf-8")
        self.query_text = " ".join(text.split())
        prepared = prepareQuery(text, initNs=_prefixes(text))
        self.algebra = prepared.algebra
        self.triples = []
        self.operators = set()
        _walk(prepared.algebra, self._visit)
        self.uris = {u for triple in self.triples for u in triple
                     if u.startswith(ONTO_NS)}

    def _visit(self, node):
        self.operators.add(node.name)
        if node.name == "BGP":
            for triple in node.triples:
                self.triples.append(tuple(_term(t) for t in triple))

    # --- shape tests --------------------------------------------------------

    def types_of(self, var: str):
        return [o for s, p, o in self.triples if s == var and p == RDF_TYPE]

    @property
    def binds_object(self) -> bool:
        """The CQA asks for a pair (?s, ?o), so the correspondence is about a
        relation rather than about a class."""
        return any("?o" in (s, o) for s, _, o in self.triples)

    @property
    def is_bare_class(self) -> bool:
        """Exactly `?s a <C>` and nothing else."""
        return (len(self.triples) == 1 and self.triples[0][0] == "?s"
                and self.triples[0][1] == RDF_TYPE
                and self.triples[0][2].startswith(ONTO_NS)
                and not self._extra_operators)

    @property
    def single_property(self):
        """(predicate, inverted) when the side is exactly one property triple
        between the two answer variables, else None."""
        if len(self.triples) != 1 or self._extra_operators:
            return None
        s, p, o = self.triples[0]
        if p == RDF_TYPE or not p.startswith(ONTO_NS):
            return None
        if {s, o} != {"?s", "?o"}:
            return None
        return p, s == "?o"        # inverted when ?o is the subject

    @property
    def _extra_operators(self) -> bool:
        return bool(self.operators - {"SelectQuery", "Project", "BGP",
                                      "ToMultiSet", "Join", "Distinct"})

    @property
    def class_union(self):
        """The classes of `{?s a A} UNION {?s a B} ...` when that is the WHOLE
        pattern, else None.

        Read off the algebra tree rather than the flattened triple list: `?s a C .
        {?s a D} UNION {?s a E}` has the same three type triples as a plain
        three-way union and means something else entirely (C AND (D OR E)).
        Flattening turned that intersection into a union until this was fixed.
        """
        return _class_union(_root_pattern(self.algebra))

    def canonical(self) -> str:
        """Stable text for deduplication: helper variables collapsed, ?s/?o kept."""
        def var(t):
            return t if t in ("?s", "?o") or not t.startswith("?") else "?v"
        rows = sorted(f"{var(s)} {p} {var(o)}" for s, p, o in self.triples)
        return f"{self.source}|" + " . ".join(rows) + "|" + ",".join(sorted(self._ops()))

    def _ops(self):
        return self.operators - {"SelectQuery", "Project", "BGP", "ToMultiSet", "Join"}


def _term(t) -> str:
    n3 = t.n3()
    return n3[1:-1] if n3.startswith("<") else n3


# Algebra nodes that only wrap the graph pattern; the pattern itself is under .p
_WRAPPERS = ("SelectQuery", "Project", "Distinct", "Reduced", "Slice",
             "OrderBy", "ToMultiSet")


def _root_pattern(node):
    while hasattr(node, "name") and node.name in _WRAPPERS and node.get("p") is not None:
        node = node["p"]
    return node


def _class_union(node):
    """[classes] if `node` is a union tree of single `?s a <C>` patterns, else None."""
    if not hasattr(node, "name"):
        return None
    if node.name == "Union":
        left, right = _class_union(node["p1"]), _class_union(node["p2"])
        return None if left is None or right is None else left + right
    if node.name == "BGP" and len(node.triples) == 1:
        s, p, o = (_term(t) for t in node.triples[0])
        if s == "?s" and p == RDF_TYPE and o.startswith(ONTO_NS):
            return [o]
    return None


def _walk(node, hit):
    if not hasattr(node, "name"):
        return
    hit(node)
    for value in node.values():
        if hasattr(value, "name"):
            _walk(value, hit)
        elif isinstance(value, list):
            for item in value:
                if hasattr(item, "name"):
                    _walk(item, hit)


# --- what the existing alignment covers -------------------------------------

def cell_sides(mapping):
    """The two URI sets a cell commits to, entity1 side and entity2 side."""
    right = set(mapping.targets)
    restriction = mapping.restriction or {}
    for key in ("property", "value"):
        value = restriction.get(key)
        if value and str(value).startswith(ONTO_NS):
            right.add(value)
    return {mapping.entity1}, right


def coverage(a_uris, b_uris, store):
    """'covered' / 'partial' / 'new' for a pair of query patterns."""
    best = "new"
    for mapping in store.mappings:
        left, right = cell_sides(mapping)
        for x, y in ((a_uris, b_uris), (b_uris, a_uris)):
            if left <= x and right <= y:
                return "covered"
            if (left & x) and (right & y):
                best = "partial"
    return best


# --- EDOAL rendering --------------------------------------------------------

def plain_cell(a: str, b: str) -> str:
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{a}"/>\n'
            f'\t\t<entity2 rdf:resource="{b}"/>\n'
            "\t\t<measure rdf:datatype='xsd:float'>1.0</measure>\n"
            "\t\t<relation>=</relation>\n"
            "\t</Cell>\n</map>\n")


def occurrence_cell(entity1: str, base: str, prop: str) -> str:
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{entity1}"/>\n'
            "\t\t<entity2>\n\t\t\t<Class>\n"
            f'\t\t\t\t<and><class rdf:resource="{base}"/></and>\n'
            "\t\t\t\t<and>\n\t\t\t\t\t<AttributeOccurenceCondition>\n"
            f'\t\t\t\t\t\t<onAttribute><Property rdf:resource="{prop}"/></onAttribute>\n'
            "\t\t\t\t\t</AttributeOccurenceCondition>\n\t\t\t\t</and>\n"
            "\t\t\t</Class>\n\t\t</entity2>\n"
            "\t\t<relation>=</relation>\n\t</Cell>\n</map>\n")


def type_cell(entity1: str, base: str, prop: str, value_class: str) -> str:
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{entity1}"/>\n'
            "\t\t<entity2>\n\t\t\t<Class>\n"
            f'\t\t\t\t<and><class rdf:resource="{base}"/></and>\n'
            "\t\t\t\t<and>\n\t\t\t\t\t<AttributeTypeCondition>\n"
            f'\t\t\t\t\t\t<onAttribute><Relation rdf:resource="{prop}"/></onAttribute>\n'
            f'\t\t\t\t\t\t<value><Instance rdf:resource="{value_class}"/></value>\n'
            "\t\t\t\t\t</AttributeTypeCondition>\n\t\t\t\t</and>\n"
            "\t\t\t</Class>\n\t\t</entity2>\n"
            "\t\t<relation>=</relation>\n\t</Cell>\n</map>\n")


def value_cell(entity1: str, base: str, prop: str, literal: str) -> str:
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{entity1}"/>\n'
            "\t\t<entity2>\n\t\t\t<class>\n"
            f'\t\t\t\t<and><class rdf:resource="{base}"/></and>\n'
            "\t\t\t\t<and>\n\t\t\t\t\t<AttributeValueCondition>\n"
            f'\t\t\t\t\t\t<onAttribute><Property rdf:resource="{prop}"/></onAttribute>\n'
            "\t\t\t\t\t\t<comparator>xsd:Equal</comparator>\n"
            f"\t\t\t\t\t\t<value>{literal}</value>\n"
            "\t\t\t\t\t</AttributeValueCondition>\n\t\t\t\t</and>\n"
            "\t\t\t</class>\n\t\t</entity2>\n"
            "\t\t<relation>=</relation>\n\t</Cell>\n</map>\n")


def union_cell(entity1: str, classes) -> str:
    members = "".join(f'\t\t\t\t<class rdf:resource="{c}"/>\n' for c in classes)
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{entity1}"/>\n'
            "\t\t<entity2>\n\t\t\t<or rdf:ParseType=\"Collection\">\n"
            f"{members}"
            "\t\t\t</or>\n\t\t</entity2>\n"
            "\t\t<measure rdf:datatype='xsd:float'>1.0</measure>\n"
            "\t\t<relation>=</relation>\n\t</Cell>\n</map>\n")


def intersection_cell(entity1: str, classes) -> str:
    members = "".join(f'\t\t\t\t<and><class rdf:resource="{c}"/></and>\n' for c in classes)
    return ("<map>\n\t<Cell>\n"
            f'\t\t<entity1 rdf:resource="{entity1}"/>\n'
            "\t\t<entity2>\n\t\t\t<Class>\n"
            f"{members}"
            "\t\t\t</Class>\n\t\t</entity2>\n"
            "\t\t<relation>=</relation>\n\t</Cell>\n</map>\n")


# --- turning one complex side into an EDOAL expression ----------------------

def describe_complex(side: Side):
    """Recognise the handful of query shapes EDOAL can state, or return None.

    Only shapes whose translation is unambiguous are accepted. Anything with a
    FILTER, MINUS, BIND, a property chain, or a second entity in the pattern is
    left alone rather than approximated — an approximate correspondence is worse
    than a missing one, because it silently changes what a rewritten query means.
    """
    if side.is_bare_class:
        return ("simple", {"class": side.triples[0][2]})

    if "Union" in side.operators:
        branches = side.class_union
        if branches and len(set(branches)) > 1:
            return ("union", {"classes": sorted(set(branches))})
        return None

    if side.operators & {"Filter", "Minus", "Extend", "LeftJoin"}:
        return None

    types = side.types_of("?s")
    others = [t for t in side.triples if not (t[0] == "?s" and t[1] == RDF_TYPE)]

    if len(types) > 1 and not others:
        return ("intersection", {"classes": sorted(set(types))})

    if len(types) == 1 and len(others) == 1:
        s, p, o = others[0]
        if s != "?s" or p == RDF_TYPE:
            return None
        if o.startswith("?"):
            # `?s a C . ?s p ?v` — the property only has to exist, unless the
            # value is itself typed (handled below)
            if any(t[0] == o for t in side.triples):
                return None
            return ("occurrence", {"class": types[0], "property": p})
        if o.startswith('"'):
            return ("value", {"class": types[0], "property": p,
                              "literal": o.split("^^")[0].strip('"')})
        return None

    if len(types) == 1 and len(others) == 2:
        # `?s a C . ?s p ?v . ?v a D`
        link = [t for t in others if t[0] == "?s" and t[1] != RDF_TYPE]
        typed = [t for t in others if t[1] == RDF_TYPE and t[0] != "?s"]
        if len(link) == 1 and len(typed) == 1 and link[0][2] == typed[0][0]:
            return ("type", {"class": types[0], "property": link[0][1],
                             "value": typed[0][2]})
    return None


def shape_label(side: Side) -> str:
    """A short name for why a side could not be written as EDOAL.

    The report groups the leftovers by this, because the groups are not noise —
    each one names a class of correspondence the alignment format as parsed by
    tools/alignment_store.py simply cannot hold, which is a finding in itself.
    """
    if side.is_bare_class:
        return "bare-class"
    if side.single_property:
        return "single-property"
    if side.binds_object and not (side.operators & {"Minus", "Extend", "Filter", "Union"}):
        # Both answer variables are bound, so the pair states a RELATION
        # equivalence, not a class one. Two ways to write a relation that a bare
        # property URI cannot: restrict its ends, or walk through an intermediate.
        direct = [t for t in side.triples
                  if {t[0], t[2]} == {"?s", "?o"} and t[1] != RDF_TYPE]
        return "relation with class restriction" if direct else "relation via a path"
    if "Minus" in side.operators:
        return "negation (MINUS)"
    if "Extend" in side.operators:
        return "computed value (BIND)"
    if "Filter" in side.operators:
        return "filter"
    if "Union" in side.operators:
        return "union of expressions"

    types = side.types_of("?s")
    others = [t for t in side.triples if not (t[0] == "?s" and t[1] == RDF_TYPE)]
    if not types and others:
        typed_object = any(t[1] == RDF_TYPE and t[0] != "?s" for t in side.triples)
        if len(others) == 2 and typed_object:
            return "type condition without a base class"
        if len(others) == 1:
            s, p, o = others[0]
            if o.startswith('"'):
                return "value condition without a base class"
            if s != "?s":
                return "inverse property"
            return "occurrence condition without a base class"
    if len(others) > 2:
        return "property chain"
    return "other"


def render(entity1: str, kind: str, data: dict) -> str:
    if kind == "simple":
        return plain_cell(entity1, data["class"])
    if kind == "union":
        return union_cell(entity1, data["classes"])
    if kind == "intersection":
        return intersection_cell(entity1, data["classes"])
    if kind == "occurrence":
        return occurrence_cell(entity1, data["class"], data["property"])
    if kind == "type":
        return type_cell(entity1, data["class"], data["property"], data["value"])
    if kind == "value":
        return value_cell(entity1, data["class"], data["property"], data["literal"])
    raise ValueError(kind)


# --- main -------------------------------------------------------------------

def collect():
    """Every implied correspondence, deduplicated, with its coverage verdict."""
    store = get_alignment_store()
    sides_by_cqa = {}
    for folder in sorted(p for p in CQA_DIR.iterdir() if p.is_dir()):
        sides = {}
        for path in sorted(folder.glob("*.sparql")):
            source = path.stem.lower()
            sides[source] = Side(folder.name, source, path)
        sides_by_cqa[folder.name] = sides

    seen, records = set(), []
    for cqa, sides in sides_by_cqa.items():
        for a_name, b_name in itertools.combinations(sorted(sides), 2):
            a, b = sides[a_name], sides[b_name]
            key = tuple(sorted((a.canonical(), b.canonical())))
            if key in seen:
                continue
            seen.add(key)

            if a.is_bare_class and b.is_bare_class:
                level = "simple"
            elif a.single_property and b.single_property:
                level = "property-level"
            else:
                level = "complex"

            records.append({
                "cqa": cqa,
                "pair": [a_name, b_name],
                "level": level,
                # class vs relation is orthogonal to simple/complex: a relation
                # correspondence can be a bare property pair or a whole path.
                "scope": "relation" if (a.binds_object or b.binds_object) else "class",
                "coverage": coverage(a.uris, b.uris, store),
                "sides": {a_name: a, b_name: b},
            })
    return records


def build_outputs(records):
    """(edoal cell texts, unexpressed entries) for the records that are new."""
    cells, unexpressed = [], []
    for rec in records:
        if rec["coverage"] == "covered":
            continue
        a_name, b_name = rec["pair"]
        a, b = rec["sides"][a_name], rec["sides"][b_name]

        if rec["level"] == "property-level":
            (pa, inv_a), (pb, inv_b) = a.single_property, b.single_property
            if inv_a != inv_b:
                # `?s p ?o` on one side and `?o q ?s` on the other is an inverse
                # correspondence; writing it as `=` would state something false.
                unexpressed.append(_unexpressed(rec, a, b, "inverse property pair"))
                continue
            cells.append((rec, plain_cell(pa, pb) + plain_cell(pb, pa)))
            continue

        # Class-level: EDOAL needs a NAMED class on the entity1 side, so at least
        # one of the two sides has to be a bare `?s a <C>`.
        for named, other in ((a, b), (b, a)):
            if not named.is_bare_class:
                continue
            shape = describe_complex(other)
            if shape is None:
                continue
            entity1 = named.triples[0][2]
            kind, data = shape
            text = render(entity1, kind, data)
            if kind == "simple":                 # symmetric, store both ways
                text += plain_cell(data["class"], entity1)
            cells.append((rec, text))
            break
        else:
            unexpressed.append(_unexpressed(rec, a, b, "no EDOAL shape fits"))
    return cells, unexpressed


def _unexpressed(rec, a, b, why):
    return {
        "cqa": rec["cqa"],
        "pair": rec["pair"],
        "level": rec["level"],
        "scope": rec["scope"],
        "coverage": rec["coverage"],
        "reason": why,
        "blocking_shape": sorted({shape_label(s) for s in (a, b)} - {"bare-class"})
                          or ["bare-class"],
        "sides": {s.source: {"file": str(s.path.relative_to(ROOT)).replace("\\", "/"),
                             "query": s.query_text,
                             "shape": shape_label(s),
                             "uris": sorted(s.uris)} for s in (a, b)},
    }


HEADER = """<?xml version='1.0' encoding='utf-8'?>

<!-- Correspondences implied by the IRIT CQA dataset that app/resources/mapping.rdf
     does not contain. Generated by eval/derive_mappings_from_cqas.py - do not edit
     by hand, and do not merge blindly: see eval/derived_mappings/report.md for
     which of these are worth trusting.

     Same <Cell> style as mapping.rdf, so merging is a copy of the <map> blocks
     into that file. Each block is preceded by the CQA it came from. -->

<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:align="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'>
<Alignment>
  <xml>yes</xml>
  <level>2EDOAL</level>
  <type>11</type>
"""


def write_if_changed(path: Path, text: str):
    if path.exists() and path.read_text(encoding="utf-8") == text:
        print(f"  unchanged {path.name}")
        return
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        print(f"  previous kept as {path.name}.bak")
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    records = collect()
    cells, unexpressed = build_outputs(records)

    counts = Counter((r["level"], r["coverage"]) for r in records)
    scopes = Counter((r["scope"], r["coverage"]) for r in records)
    print(f"implied correspondences (deduplicated): {len(records)}")
    for level in ("simple", "property-level", "complex"):
        row = {c: counts[(level, c)] for c in ("covered", "partial", "new")}
        print(f"  {level:16} covered {row['covered']:4}   partial {row['partial']:4}   "
              f"new {row['new']:4}")
    print("  " + "-" * 56)
    for scope in ("class", "relation"):
        row = {c: scopes[(scope, c)] for c in ("covered", "partial", "new")}
        print(f"  about a {scope:8} covered {row['covered']:4}   partial {row['partial']:4}   "
              f"new {row['new']:4}")
    print(f"\nEDOAL cells to write: {len(cells)}")
    print(f"kept as raw query pairs: {len(unexpressed)}")

    if args.check:
        print("\n--check: nothing written")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    body = []
    for rec, text in cells:
        body.append(f"\n<!-- CQA {rec['cqa']}  ({rec['pair'][0]} <-> {rec['pair'][1]}) "
                    f"[{rec['level']}, was: {rec['coverage']}] -->\n")
        body.append(text)
    write_if_changed(OUT_DIR / "derived_mappings.rdf",
                     HEADER + "".join(body) + "\n</Alignment>\n</rdf:RDF>\n")

    write_if_changed(OUT_DIR / "complex_unexpressed.json",
                     json.dumps({
                         "description": ("Implied correspondences that mapping.rdf does "
                                         "not have and EDOAL cannot state in this file's "
                                         "style. Kept as the two SPARQL queries so nothing "
                                         "is lost."),
                         "count": len(unexpressed),
                         "entries": unexpressed,
                     }, indent=2, ensure_ascii=False) + "\n")

    # Numbers the report quotes, dumped so the report can be re-checked without
    # re-reading 278 files by hand.
    pair_counts = Counter(" - ".join(sorted(r["pair"]))
                          for r in records if r["coverage"] == "new")
    blocking = Counter(s for e in unexpressed for s in e["blocking_shape"])
    write_if_changed(OUT_DIR / "summary.json", json.dumps({
        "implied_total": len(records),
        "by_level_and_coverage": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "by_scope_and_coverage": {f"{k[0]}/{k[1]}": v for k, v in sorted(scopes.items())},
        "new_by_ontology_pair": dict(pair_counts.most_common()),
        "edoal_cells_written": len(cells),
        "unexpressed": len(unexpressed),
        "unexpressed_by_shape": dict(blocking.most_common()),
    }, indent=2) + "\n")
    print("\nleftovers by blocking shape:")
    for shape, n in blocking.most_common():
        print(f"  {shape:40} {n:4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
