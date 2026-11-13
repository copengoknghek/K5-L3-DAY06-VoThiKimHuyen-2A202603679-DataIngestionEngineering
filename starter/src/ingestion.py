"""Read and normalize source data."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

CONTENT_CHAR_THRESHOLD = 400


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_chapter(text: str) -> int | None:
    """Infer chapter number from page text (OpenStax headings)."""
    head = text[:2500]
    outline = re.search(r"(?i)chapter\s+outline\s+(\d+)\.\d+", head)
    if outline:
        return int(outline.group(1))

    patterns = [
        r"(?i)\bchapter\s+(\d+)\b",
        r"(?m)^\s*(\d+)\.\d{1,2}(?:\.\d+)?\s+",
        r"(?m)^\s*(\d+)\.\d{1,2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, head)
        if match:
            chapter = int(match.group(1))
            if 1 <= chapter <= 20:
                return chapter

    # Majority vote from section numbers in the first screenful of text
    sections = re.findall(r"\b(\d)\.\d{1,2}\b", head[:1200])
    if sections:
        from collections import Counter

        chapter, _ = Counter(sections).most_common(1)[0]
        return int(chapter)
    return None


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[DocumentPage]:
    """Extract each physical page; keep page number, source, and metadata."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    empty_pages: list[int] = []

    start_idx = (page_start - 1) if page_start else 0
    end_idx = page_end if page_end else len(reader.pages)
    end_idx = min(end_idx, len(reader.pages))

    for physical_page in range(start_idx + 1, end_idx + 1):
        page_index = physical_page - 1
        raw = reader.pages[page_index].extract_text() or ""
        text = normalize_text(raw)
        if len(text) < CONTENT_CHAR_THRESHOLD:
            empty_pages.append(physical_page)
            continue

        chapter = parse_chapter(text)
        metadata: dict = {}
        if chapter is not None:
            metadata["chapter"] = chapter
        metadata["char_count"] = len(text)
        metadata["physical_page"] = physical_page

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=physical_page,
                text=text,
                metadata=metadata,
            )
        )

    if empty_pages:
        print(
            f"Skipped {len(empty_pages)} low-content pages in range "
            f"{start_idx + 1}-{end_idx} (examples: {empty_pages[:5]})"
        )
    return pages


def select_content_page_range(path: Path, min_content_pages: int = 30) -> tuple[int, int, int]:
    """Scan PDF and return (start, end, content_page_count) for a contiguous range."""
    reader = PdfReader(str(path))
    content_indices: list[int] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = normalize_text(raw)
        if len(text) >= CONTENT_CHAR_THRESHOLD:
            content_indices.append(i + 1)

    if len(content_indices) < min_content_pages:
        raise ValueError(
            f"PDF has only {len(content_indices)} content pages; need {min_content_pages}"
        )

    # Prefer early chapters: find first run of enough consecutive content pages
    start = content_indices[0]
    best_start = start
    best_end = start
    count = 0
    for idx, pnum in enumerate(content_indices):
        if idx == 0:
            count = 1
            run_start = pnum
            continue
        prev = content_indices[idx - 1]
        if pnum - prev <= 3:
            count += 1
        else:
            if count >= min_content_pages:
                best_start, best_end = run_start, content_indices[idx - 1]
            run_start = pnum
            count = 1
        if count >= min_content_pages and pnum - run_start < 80:
            best_start, best_end = run_start, pnum

    # Extend to ~35-45 content pages if available
    target = min(42, len(content_indices))
    run: list[int] = []
    for pnum in content_indices:
        if pnum >= best_start:
            run.append(pnum)
        if len(run) >= target:
            break
    if len(run) >= min_content_pages:
        return best_start, run[-1], len(run)

    return best_start, best_end, min_content_pages


def propagate_chapters(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Forward-fill chapter metadata so filter works on continuation pages."""
    last: int | None = None
    updated: list[DocumentPage] = []
    for page in sorted(pages, key=lambda p: p.page):
        ch = page.metadata.get("chapter")
        if ch is not None:
            last = int(ch)
        elif last is not None:
            page.metadata["chapter"] = last
        updated.append(page)
    return updated
