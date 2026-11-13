"""Helper: print page snippets for writing benchmark gold evidence."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import load_pdf, propagate_chapters  # noqa: E402

PDF = ROOT / "data" / "college-success.pdf"


def main() -> None:
    pages = load_pdf(
        PDF,
        "college-success-2020",
        "data/college-success.pdf",
        page_start=25,
        page_end=100,
    )
    pages = propagate_chapters(pages)
    for p in pages:
        if p.metadata.get("chapter") in (1, 2, 3):
            lines = [ln.strip() for ln in p.text.split("\n") if len(ln.strip()) > 40]
            for ln in lines[:4]:
                print(f"p{p.page} ch{p.metadata.get('chapter')}: {ln[:140]}")


if __name__ == "__main__":
    main()
