"""Evaluate retrieved evidence and preserve per-query traces."""
from __future__ import annotations

import re
from typing import Any

from .ingestion import normalize_text


def _norm_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).lower()


def _evidence_in_chunk(chunk_text: str, evidence_quote: str) -> bool:
    return _norm_for_match(evidence_quote) in _norm_for_match(chunk_text)


def _hit_chunk(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    gold_source = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    alt_sources = question.get("alt_gold_source_ids") or []
    alt_pages = question.get("alt_gold_pages") or []

    doc_ok = hit.get("document_id") == gold_source or hit.get("document_id") in alt_sources
    page_ok = hit.get("page") == gold_page or hit.get("page") in alt_pages
    if not (doc_ok and page_ok):
        return False

    evidence = question.get("evidence_quote", "")
    alt_evidence = question.get("alt_evidence_quotes") or []
    if _evidence_in_chunk(hit.get("text", ""), evidence):
        return True
    return any(_evidence_in_chunk(hit.get("text", ""), alt) for alt in alt_evidence)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace."""
    hit_at_1 = bool(hits) and _hit_chunk(hits[0], question)
    hit_at_3 = any(_hit_chunk(h, question) for h in hits[:3])

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "question": question.get("question"),
        "filter": question.get("filter"),
        "hits": hits,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"n": 0, "hit_at_1": 0, "hit_at_3": 0, "hit_at_1_rate": 0.0, "hit_at_3_rate": 0.0}
        h1 = sum(1 for r in rows if r.get("hit_at_1"))
        h3 = sum(1 for r in rows if r.get("hit_at_3"))
        return {
            "n": n,
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_1_rate": h1 / n,
            "hit_at_3_rate": h3 / n,
        }

    by_split: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        split = trace.get("split", "unknown")
        by_split.setdefault(split, []).append(trace)

    return {
        "overall": stats(traces),
        "dev": stats(by_split.get("dev", [])),
        "test": stats(by_split.get("test", [])),
    }
