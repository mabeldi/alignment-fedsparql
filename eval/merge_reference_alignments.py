"""Fill the unfinished ontology pairs in app/resources/mapping.rdf from the
official OAEI conference reference alignment.

WHY
---
mapping.rdf was assembled by hand and three pairs were never completed:

    cmt <-> edas      0 correspondences (no section in the file at all)
    confof <-> edas   1 (`Country`, written in both directions)
    edas <-> ekaw     short

The `edas - confOf` banner at line ~1418 is followed by 2 real confOf-edas
cells and then 20 confOf-EKAW cells pasted under the wrong header, 19 of which
already exist in the proper `Ekaw - confOf` section. So the section was opened,
one correspondence added, another pair's block pasted below it, and nobody came
back. Adding `edas` to the pipeline is what exposed it.

WHAT THIS DOES
--------------
Adds ONLY the missing correspondences for the three broken pairs. It does not
touch the seven pairs that are already complete, and it never removes anything —
in particular the EDOAL entries (`ekaw:Accepted_Paper = cmt:Paper AND
(hasDecision -> Acceptance)`) are not in the official level-0 reference and are
the more interesting half of the alignment.

Existing correspondences are matched direction-insensitively, so nothing already
present is added again. New entries are written in both directions, matching how
the rest of the file stores them.

Run:  python eval/merge_reference_alignments.py            # report only
      python eval/merge_reference_alignments.py --write    # apply
"""

import io
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

RA_ZIP = "https://oaei.ontologymatching.org/2022/conference/data/reference-alignment.zip"
MAPPING = Path(__file__).resolve().parent.parent / "app" / "resources" / "mapping.rdf"

# Only the unfinished pairs. The other seven already match the reference, and
# re-merging them would just add duplicates for the parser to filter out.
TARGET_FILES = ["cmt-edas.rdf", "confOf-edas.rdf", "edas-ekaw.rdf"]

_ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"


def _lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _resource(cell, child_name: str):
    """The rdf:resource URI of <entity1>/<entity2>, or None if it is EDOAL.

    EDOAL entities nest a structured expression instead of carrying a plain
    resource. Those never appear in the official level-0 files, but skipping
    them keeps this safe against the local file, which does have them.
    """
    for child in cell:
        if _lname(child.tag) == child_name:
            for key, value in child.attrib.items():
                if key.endswith("resource"):
                    return value
    return None


def _cells(root):
    for node in root.iter():
        if _lname(node.tag) == "cell":
            yield node


def existing_pairs(path: Path):
    """Direction-insensitive set of the plain correspondences already in the file."""
    root = ET.parse(str(path)).getroot()
    pairs = set()
    for cell in _cells(root):
        a, b = _resource(cell, "entity1"), _resource(cell, "entity2")
        if a and b:
            pairs.add(frozenset((a, b)))
    return pairs


def official(zip_bytes: bytes):
    """{filename: [(entity1, entity2, relation), ...]} for the target pairs."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for want in TARGET_FILES:
            if want not in names:
                raise SystemExit(f"{want} not found in the reference archive")
            root = ET.fromstring(zf.read(names[want]))
            rows = []
            for cell in _cells(root):
                a, b = _resource(cell, "entity1"), _resource(cell, "entity2")
                if not (a and b):
                    continue
                rel = "="
                for child in cell:
                    if _lname(child.tag) == "relation" and child.text:
                        rel = child.text.strip()
                rows.append((a, b, rel))
            out[want] = rows
    return out


def render(a: str, b: str, rel: str) -> str:
    """One correspondence, both directions, in the file's existing style."""
    def cell(x, y):
        return ("<map>\n"
                "\t<Cell>\n"
                f'\t\t<entity1 rdf:resource="{x}"/>\n'
                f'\t\t<entity2 rdf:resource="{y}"/>\n'
                '\t\t<measure rdf:datatype="xsd:float">1.0</measure>\n'
                f"\t\t<relation>{rel}</relation>\n"
                "\t</Cell>\n"
                "</map>\n")
    return cell(a, b) + cell(b, a)


def banner(text: str) -> str:
    pad = "#" * max(4, 100 - len(text))
    return f"\n<!-- {'#' * 6} {text} {pad} -->\n\n"


def is_property(uri: str, graph_cache={}) -> bool:
    """Heuristic split for REPORTING only — nothing is filtered on it.

    Local names of properties in these ontologies start lowercase; classes do
    not. Good enough to tell you how much of the merge is class-level, which is
    what actually reaches Agent 7.
    """
    name = uri.split("#")[-1]
    return bool(name) and name[0].islower()


def main() -> int:
    write = "--write" in sys.argv

    print(f"reference : {RA_ZIP}")
    zip_bytes = requests.get(RA_ZIP, timeout=60).content
    ref = official(zip_bytes)

    have = existing_pairs(MAPPING)
    print(f"mapping   : {MAPPING}  ({len(have)} plain correspondences present)\n")

    additions = []
    for fname, rows in ref.items():
        missing = [(a, b, r) for a, b, r in rows if frozenset((a, b)) not in have]
        cls = sum(1 for a, b, _ in missing if not (is_property(a) or is_property(b)))
        print(f"{fname:18} official {len(rows):3}   already present {len(rows) - len(missing):3}   "
              f"to add {len(missing):3}  ({cls} class-level, {len(missing) - cls} property-level)")
        if missing:
            additions.append((fname, missing))

    total = sum(len(m) for _, m in additions)
    print(f"\ntotal new correspondences: {total}  "
          f"({total * 2} cells, each written in both directions)")

    if not total:
        print("nothing to do")
        return 0

    if not write:
        print("\nreport only — re-run with --write to apply")
        return 0

    text = MAPPING.read_text(encoding="utf-8")
    if "</Alignment>" not in text:
        raise SystemExit("no </Alignment> in mapping.rdf — refusing to guess where to insert")

    blocks = []
    for fname, missing in additions:
        pair = Path(fname).stem.replace("-", " - ")
        blocks.append(banner(f"{pair}  (merged from OAEI reference alignment)"))
        blocks.extend(render(a, b, r) for a, b, r in missing)

    backup = MAPPING.with_suffix(".rdf.bak")
    shutil.copy2(MAPPING, backup)
    print(f"\nbackup written: {backup}")

    # Insert before the closing tag so the additions sit inside <Alignment>.
    text = text.replace("</Alignment>", "".join(blocks) + "\n</Alignment>", 1)
    MAPPING.write_text(text, encoding="utf-8")
    print(f"updated: {MAPPING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
