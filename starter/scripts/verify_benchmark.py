"""Verify gold evidence exists on gold pages before running retrieval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import _evidence_in_chunk
from src.ingestion import load_pdf, propagate_chapters, normalize_text

MANIFEST = REPO / "data" / "manifest.json"
BENCHMARK = REPO / "benchmark" / "questions.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    pdf = ROOT / manifest["source_path_or_url"]
    pages = {
        p.page: p
        for p in propagate_chapters(
            load_pdf(
                pdf,
                manifest["source_id"],
                manifest["source_path_or_url"],
                page_start=manifest["physical_page_range"]["start"],
                page_end=manifest["physical_page_range"]["end"],
            )
        )
    }
    ok = 0
    for q in bench["questions"]:
        page = pages.get(q["gold_page"])
        if not page:
            print("FAIL", q["id"], "missing page", q["gold_page"])
            continue
        if not _evidence_in_chunk(page.text, q["evidence_quote"]):
            # try normalized full page
            if q["evidence_quote"] not in normalize_text(page.text):
                print("FAIL", q["id"], "evidence not on page", q["gold_page"])
                continue
        ok += 1
    print(f"Verified {ok}/{len(bench['questions'])} questions")
    if ok != len(bench["questions"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
