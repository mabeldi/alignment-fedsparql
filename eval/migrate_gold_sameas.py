"""Bridge the join-shaped gold queries through owl:sameAs, in place.

    python eval/migrate_gold_sameas.py            # show what would change
    python eval/migrate_gold_sameas.py --apply    # write it

Why this exists as a script rather than a regeneration: gold_questions.json is
built by eval/build_gold_questions.py, but that needs eval/cqa_source/CQAs/,
which is not in this checkout — so the generator cannot be re-run to produce the
fix. The fix itself already lives there (bridge_block, called from
compile_question); this applies the same functions to the built file so the two
agree. Once the CQA files are restored, re-running the generator reproduces this
output and this script can go.

Only questions whose blocks are JOINED are touched. UNION questions ask each
source for its own instances and never correlate one individual across two, so
they need no link and get none.

Idempotent: a question that already has a link line is left alone.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gold_questions import (  # noqa: E402
    SAME_AS, analyse, bridge_block, build_query, vocabulary,
)

GOLD = Path(__file__).resolve().parent / "questions" / "gold_questions.json"
_SERVICE = re.compile(r"SERVICE\s+<([^>]+)>")


def block_order(gold_query: str) -> list:
    """Sources in the order the gold query names them. The first is the anchor.

    Read from the query rather than from gold_sources, which is sorted
    alphabetically and so does not say which block comes first.
    """
    return [url.rstrip("/").split("/")[-2] for url in _SERVICE.findall(gold_query)]


def migrate(entry: dict) -> bool:
    """Rewrite one question. True if it changed."""
    if entry.get("combination") == "union" or len(entry.get("sources", {})) < 2:
        return False
    join_var = entry.get("join_on")
    if not join_var or SAME_AS in entry["gold_query"]:
        return False

    order = block_order(entry["gold_query"])
    anchor, blocks = order[0], []
    for src in order:
        block = entry["sources"][src]
        if src != anchor:
            block["body"] = bridge_block(block["body"], join_var, src)
            patterns, ops = analyse(block["body"])
            classes, properties = vocabulary(patterns)
            block["triple_patterns"], block["operators"] = patterns, ops
            block["classes"], block["properties"] = classes, properties
        blocks.append((src, block["body"]))

    entry["gold_query"] = build_query(entry["select"], blocks, entry["combination"])
    entry["gold_classes"] = sorted({c for s in entry["sources"].values() for c in s["classes"]})
    entry["gold_properties"] = sorted({p for s in entry["sources"].values() for p in s["properties"]})
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the file")
    args = ap.parse_args()

    data = json.loads(GOLD.read_text(encoding="utf-8"))
    changed = [q["id"] for q in data["questions"] if migrate(q)]

    if not changed:
        print("nothing to do — every joined question already has a link line")
        return 0

    print(f"{len(changed)} questions bridged: {', '.join(changed)}\n")
    sample = next(q for q in data["questions"] if q["id"] == changed[0])
    print(f"{sample['id']} now reads:\n{sample['gold_query']}\n")

    if not args.apply:
        print("dry run — nothing written. Re-run with --apply.")
        return 0

    GOLD.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {GOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
