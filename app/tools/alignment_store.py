"""
AlignmentStore — parse the EDOAL-style alignment (`resources/mapping.rdf`) once
at bootstrap and index CLASS correspondences for O(1) lookup by class URI.

We only care about class-level mappings (no property mappings). Each mapping is
kept in its original EDOAL shape so Agent 7 can later reason about whether to
UNION (same concept expressed differently) or JOIN/OPTIONAL (complementary data)
— that decision is the LLM's, not ours.

The `<relation>` in EDOAL is always "="; the real semantics live in the entity2
structure, which is one of:
  - simple          : entity2 is a plain class URI
  - restriction_type: class AND (property -> some Instance)      (AttributeTypeCondition)
  - restriction_value: class AND (property comparator value)     (AttributeValueCondition)
  - restriction_occurrence: class AND (property exists)          (AttributeOccurenceCondition)
  - union           : classA OR classB OR ...                    (<or>)
  - intersection    : classA AND classB ...                      (<and> of classes, no condition)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from config.pipeline_config import alignments_enabled

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "resources" / "mapping.rdf"


# --- tiny namespace-agnostic XML helpers ------------------------------------
# The file mixes tag casing (<Class>/<class>, <Property>/<Relation>) and
# rdf:resource vs rdf:about, so we match on lower-cased LOCAL names only.

def _local(tag: str) -> str:
    return tag.split("}")[-1]

def _lname(el: ET.Element) -> str:
    return _local(el.tag).lower()

def _uri(el: ET.Element) -> Optional[str]:
    """Return rdf:resource or rdf:about of an element, whichever is present."""
    for key, val in el.attrib.items():
        if _local(key).lower() in ("resource", "about"):
            return val
    return None

def _child(el: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if el is None:
        return None
    for c in el:
        if _lname(c) == name.lower():
            return c
    return None

def _children(el: Optional[ET.Element], name: str) -> List[ET.Element]:
    if el is None:
        return []
    return [c for c in el if _lname(c) == name.lower()]


def source_of(uri: str) -> str:
    """Ontology short name from a class URI: 'http://confOf#X' -> 'confof'."""
    if not uri:
        return ""
    body = uri.split("//", 1)[-1]        # confOf#X
    return body.split("#", 1)[0].split("/", 1)[0].lower()  # confof

def local_name(uri: str) -> str:
    """'http://confOf#Contribution' -> 'Contribution'."""
    return uri.split("#")[-1].split("/")[-1] if uri else uri


def _curie(uri: str) -> str:
    """Readable 'source:LocalName' for prompt rendering."""
    return f"{source_of(uri)}:{local_name(uri)}" if uri else uri


# Plain equivalences are the most reliable bridges, so when a caller caps the
# number of counterparts it keeps those first; conditional shapes come last.
_KIND_PRIORITY = {"simple": 0, "union": 1, "intersection": 2}


# --- mapping model ----------------------------------------------------------

@dataclass
class Mapping:
    entity1: str                       # the named class (e.g. ekaw:Rejected_Paper)
    kind: str                          # simple | restriction_* | union | intersection
    targets: List[str] = field(default_factory=list)   # class URIs on entity2 side
    restriction: Optional[dict] = None # {"property":..., "value":..., "comparator":...}
    relation: str = "="

    @property
    def entity1_source(self) -> str:
        return source_of(self.entity1)

    @property
    def target_source(self) -> str:
        return source_of(self.targets[0]) if self.targets else ""

    def render(self, uris: bool = False) -> str:
        """Human/LLM-readable equivalence. With uris=False (default) uses short
        CURIEs for logging; with uris=True wraps full <URI>s so the SPARQL
        generator can copy them verbatim, e.g.
        '<http://ekaw#Rejected_Paper> = <http://cmt#Paper> AND (<http://cmt#hasDecision> -> <http://cmt#Rejection>)'.
        """
        fmt = (lambda u: f"<{u}>") if uris else _curie
        left = fmt(self.entity1)
        if self.kind == "simple":
            return f"{left} = {fmt(self.targets[0])}"
        if self.kind == "union":
            return f"{left} = " + " OR ".join(fmt(t) for t in self.targets)
        if self.kind == "intersection":
            return f"{left} = " + " AND ".join(fmt(t) for t in self.targets)
        # restriction_*: base class AND (property condition)
        base = fmt(self.targets[0]) if self.targets else "?"
        r = self.restriction or {}
        prop = fmt(r.get("property", "")) if r.get("property") else "?"
        if self.kind == "restriction_type":
            cond = f"{prop} -> {fmt(r.get('value', ''))}"
        elif self.kind == "restriction_value":
            cond = f"{prop} {r.get('comparator', '=')} {r.get('value', '')}"
        else:  # restriction_occurrence
            cond = f"{prop} exists"
        return f"{left} = {base} AND ({cond})"

    def to_sparql_fragment(self, var: str = "?x") -> str:
        """Deterministically render the TARGET side as SPARQL triple patterns.

        This is the sound part of the rewriting: each EDOAL shape has exactly one
        correct translation, so we emit it here instead of letting the LLM guess
        (it previously turned an AttributeTypeCondition into `FILTER(?d = C)`,
        which is wrong — the condition constrains the value's TYPE).

        The entity1 side needs no template; it is simply `{var} a <entity1>`.
        """
        if not self.targets:
            return ""
        if self.kind == "simple":
            return f"{var} a <{self.targets[0]}> ."
        if self.kind == "union":
            return " UNION ".join(f"{{ {var} a <{t}> . }}" for t in self.targets)
        if self.kind == "intersection":
            return "\n".join(f"{var} a <{t}> ." for t in self.targets)

        base = self.targets[0]
        r = self.restriction or {}
        prop = r.get("property")
        lines = [f"{var} a <{base}> ."]
        if not prop:
            return "\n".join(lines)
        val_var = f"{var}_v"
        if self.kind == "restriction_type":
            # value must be an INSTANCE OF the given class
            lines.append(f"{var} <{prop}> {val_var} .")
            lines.append(f"{val_var} a <{r.get('value')}> .")
        elif self.kind == "restriction_value":
            # literal comparison; xsd:Equal (the common case) folds into the triple
            comparator = (r.get("comparator") or "").lower()
            value = r.get("value")
            if comparator.endswith("equal") or comparator in ("=", ""):
                lines.append(f'{var} <{prop}> "{value}" .')
            else:
                lines.append(f"{var} <{prop}> {val_var} .")
                lines.append(f'FILTER({val_var} {r.get("comparator")} "{value}")')
        else:  # restriction_occurrence -> the property just has to exist
            lines.append(f"{var} <{prop}> {val_var} .")
        return "\n".join(lines)


# --- parsing ----------------------------------------------------------------

def _parse_condition(and_node: ET.Element) -> Optional[dict]:
    """Extract a restriction dict from an <and> that holds an Attribute*Condition."""
    type_cond = _child(and_node, "AttributeTypeCondition")
    value_cond = _child(and_node, "AttributeValueCondition")
    occ_cond = _child(and_node, "AttributeOccurenceCondition")

    cond = type_cond if type_cond is not None else (
        value_cond if value_cond is not None else occ_cond)
    if cond is None:
        return None

    on_attr = _child(cond, "onAttribute")
    # property is expressed as <Relation> or <Property>
    prop_el = _child(on_attr, "Relation")
    if prop_el is None:
        prop_el = _child(on_attr, "Property")
    prop = _uri(prop_el) if prop_el is not None else None

    if type_cond is not None:
        value_el = _child(type_cond, "value")
        instance = _child(value_el, "Instance")
        return {"kind": "restriction_type", "property": prop,
                "value": _uri(instance) if instance is not None else None}
    if value_cond is not None:
        comparator = _child(value_cond, "comparator")
        value = _child(value_cond, "value")
        return {"kind": "restriction_value", "property": prop,
                "comparator": comparator.text.strip() if comparator is not None and comparator.text else "=",
                "value": value.text.strip() if value is not None and value.text else None}
    return {"kind": "restriction_occurrence", "property": prop, "value": None}


def _parse_entity2(entity2: ET.Element) -> Optional[Mapping]:
    """Turn the entity2 subtree into a (partial) Mapping without entity1 set yet."""
    # Simple: entity2 itself carries the class URI.
    direct = _uri(entity2)
    if direct:
        return Mapping(entity1="", kind="simple", targets=[direct])

    # Union directly under entity2: <or><class .../>...</or>
    or_node = _child(entity2, "or")
    if or_node is None:
        or_node = _child(_child(entity2, "class"), "or")
    if or_node is not None:
        targets = [u for c in _children(or_node, "class") if (u := _uri(c))]
        return Mapping(entity1="", kind="union", targets=targets)

    class_node = _child(entity2, "class")
    if class_node is None:
        return None

    and_nodes = _children(class_node, "and")
    base_classes: List[str] = []
    restriction: Optional[dict] = None
    for a in and_nodes:
        cls = _child(a, "class")
        if cls is not None and _uri(cls):
            base_classes.append(_uri(cls))
        cond = _parse_condition(a)
        if cond:
            restriction = cond

    if restriction:
        r = {k: v for k, v in restriction.items() if k != "kind"}
        return Mapping(entity1="", kind=restriction["kind"], targets=base_classes, restriction=r)
    if len(base_classes) > 1:
        return Mapping(entity1="", kind="intersection", targets=base_classes)
    if len(base_classes) == 1:
        return Mapping(entity1="", kind="simple", targets=base_classes)
    return None


class AlignmentStore:
    """Parsed alignment, indexed by class URI (both sides for cross-source lookup)."""

    def __init__(self):
        self.mappings: List[Mapping] = []
        self._by_class: Dict[str, List[Mapping]] = {}

    def load(self, path: Path = _DEFAULT_MAPPING_PATH) -> "AlignmentStore":
        root = ET.parse(str(path)).getroot()
        seen: Set[tuple] = set()   # dedup (entity1, rendered) — the file has duplicates

        for cell in root.iter():
            if _lname(cell) != "cell":
                continue
            e1 = _child(cell, "entity1")
            e2 = _child(cell, "entity2")
            rel = _child(cell, "relation")
            if e1 is None or e2 is None:
                continue
            e1_uri = _uri(e1)
            if not e1_uri:
                continue

            mapping = _parse_entity2(e2)
            if mapping is None or not mapping.targets:
                continue
            mapping.entity1 = e1_uri
            if rel is not None and rel.text:
                mapping.relation = rel.text.strip()

            dedup_key = (e1_uri, mapping.render())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            self.mappings.append(mapping)

        self._build_index()
        return self

    def _build_index(self):
        """Index each mapping under entity1 AND every target class, so a
        correspondence is retrievable whichever source ends up as the base."""
        self._by_class = {}
        for m in self.mappings:
            for uri in [m.entity1, *m.targets]:
                self._by_class.setdefault(uri, []).append(m)

    def get(self, class_uri: str) -> List[Mapping]:
        return self._by_class.get(class_uri, [])

    def counterparts(self, class_uri: str,
                     loaded_sources: Optional[Set[str]] = None) -> List[tuple]:
        """The classes in OTHER loaded ontologies that `class_uri` is aligned with.

        Returns (counterpart_uri, mapping) pairs, plain equivalences first. This
        is the lookup that lets the alignment drive retrieval: given a class we
        DID find, it names the classes we should have found in the other sources
        but could not, because they are lexicalised differently.

        ONLY A PLAIN `=` READS BOTH WAYS. Every other shape defines its left side
        as a NARROWER thing than its right side:

            ekaw:Accepted_Paper = cmt:Paper AND (cmt:hasDecision -> cmt:Acceptance)

        Read left to right that is sound. Read right to left it says every cmt
        paper is an accepted one, which is false — and it was arriving here as a
        plain candidate with the condition stripped off. One retrieved
        `cmt:Paper` pulled in Accepted_Paper, Rejected_Paper, Assigned_Paper and
        Presentation as if the question had named four different things, and
        Agent 3 then searched for relationships BETWEEN them. Measured on the
        saved runs, that was 7 of 15 injected classes on CJ-01 and 8 of 22 on
        CJ-03.

        Nothing is lost by refusing it: when a question really does ask for the
        narrower class, retrieval finds that class directly and this expansion is
        not involved. Plain `=` keeps reading both ways — the file states it in
        both directions 330 times out of 334, but 3 pairs are written once only
        (confof:Organization -> ekaw:Organisation among them), and those would
        disappear if direction were enforced here too.
        """
        loaded = {s.lower() for s in loaded_sources} if loaded_sources else None
        own = source_of(class_uri)
        out: List[tuple] = []
        seen: Set[str] = set()
        for m in self.get(class_uri):
            if m.kind != "simple" and m.entity1 != class_uri:
                continue                           # backwards read of a narrowing rule
            for uri in [m.entity1, *m.targets]:
                if not uri or uri == class_uri or uri in seen:
                    continue
                src = source_of(uri)
                if src == own:                     # same ontology: not a bridge
                    continue
                if loaded is not None and src not in loaded:
                    continue
                seen.add(uri)
                out.append((uri, m))
        out.sort(key=lambda pair: (_KIND_PRIORITY.get(pair[1].kind, 3), pair[0]))
        return out

    def find_for_classes(self, class_uris: Set[str],
                         loaded_sources: Optional[Set[str]] = None) -> List[Mapping]:
        """All correspondences touching any of `class_uris` whose OTHER side lives
        in a loaded source (so we never propose federation to an absent ontology).
        Deduplicated across the input classes."""
        loaded = {s.lower() for s in loaded_sources} if loaded_sources else None
        out: List[Mapping] = []
        seen: Set[int] = set()
        # The file states plain equivalences in BOTH directions ("A = B" and
        # "B = A"), which would feed the generator the same fact twice. Direction
        # is only meaningful for the conditional shapes, so we collapse the
        # symmetric ones onto an unordered pair.
        seen_simple: Set[frozenset] = set()
        for uri in class_uris:
            for m in self.get(uri):
                if id(m) in seen:
                    continue
                sources = {m.entity1_source, m.target_source}
                other = sources - {source_of(uri)}
                if loaded is not None and other and not (other & loaded):
                    continue
                if m.kind == "simple":
                    pair = frozenset((m.entity1, m.targets[0]))
                    if pair in seen_simple:
                        continue
                    seen_simple.add(pair)
                seen.add(id(m))
                out.append(m)
        return out

    def stats(self) -> Dict[str, int]:
        by_kind: Dict[str, int] = {}
        for m in self.mappings:
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        return {"total": len(self.mappings), "indexed_classes": len(self._by_class), **by_kind}


# module-level singleton, built once at bootstrap
_STORE: Optional[AlignmentStore] = None
_EMPTY_STORE = AlignmentStore()

def get_alignment_store() -> AlignmentStore:
    """The pipeline's only mapping access point, including implicit rendering uses.

    Disabled mode does not parse mapping.rdf and cannot expose a cached loaded
    store. All consumers receive the same empty lookup interface. This controls
    schema mappings only, not instance-level links stored in Fuseki.
    """
    global _STORE
    if not alignments_enabled():
        return _EMPTY_STORE
    if _STORE is None:
        _STORE = AlignmentStore().load()
    return _STORE


if __name__ == "__main__":
    store = get_alignment_store()
    print("stats:", store.stats())
    print("\n-- complex mappings (non-simple) --")
    for m in store.mappings:
        if m.kind != "simple":
            print(f"  [{m.kind}] {m.render()}")
    print("\n-- lookup ekaw:Rejected_Paper --")
    for m in store.get("http://ekaw#Rejected_Paper"):
        print(f"  {m.render()}")
