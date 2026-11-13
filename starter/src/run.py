"""Pipeline entry point: ingest, index three strategies, benchmark retrieval."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import GeminiEmbedder, LocalEmbedder, create_embedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf, propagate_chapters
from .models import Chunk, DocumentPage
from .store import PgVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_ROOT = Path(__file__).resolve().parents[1]
EMBED_CACHE_PATH = REPO_ROOT / "results" / "embed_cache.json"
EMBED_CACHE_META_PATH = REPO_ROOT / "results" / "embed_cache.meta.json"

STRATEGIES = {
    "A": {"chunking": "fixed_size", "chunk_size": 500, "overlap": 50, "metadata_filter": False},
    "B": {"chunking": "by_paragraph", "chunk_size": 500, "overlap": 0, "metadata_filter": False},
    "C": {"chunking": "by_paragraph", "chunk_size": 500, "overlap": 0, "metadata_filter": True},
}


def _repo_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    for base in (REPO_ROOT, STARTER_ROOT):
        candidate = base / rel
        if candidate.exists() or base == REPO_ROOT:
            return REPO_ROOT / rel if (REPO_ROOT / rel).parent.exists() or rel.startswith("data") else STARTER_ROOT / rel
    return REPO_ROOT / rel


def _resolve_data_path(manifest: dict[str, Any]) -> Path:
    rel = manifest["source_path_or_url"]
    for base in (STARTER_ROOT, REPO_ROOT):
        candidate = base / rel
        if candidate.exists():
            return candidate
    return STARTER_ROOT / rel


def load_embed_cache(dimensions: int, model_name: str) -> dict[str, list[float]]:
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EMBED_CACHE_META_PATH.exists():
        meta = json.loads(EMBED_CACHE_META_PATH.read_text(encoding="utf-8"))
        if meta.get("dimensions") != dimensions or meta.get("model_name") != model_name:
            print(
                f"Embed cache is for {meta.get('model_name')} "
                f"({meta.get('dimensions')}d); current run is {model_name} ({dimensions}d). "
                "Ignoring old cache (use a new model/dim)."
            )
            return {}
    if not EMBED_CACHE_PATH.exists():
        return {}
    data = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for key, vec in data.items():
        vector = list(vec)
        if len(vector) == dimensions:
            out[key] = vector
    dropped = len(data) - len(out)
    if dropped:
        print(f"Dropped {dropped} cached vectors with wrong dimension (expected {dimensions}).")
    return out


def save_embed_cache(
    cache: dict[str, list[float]], *, dimensions: int, model_name: str
) -> None:
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    EMBED_CACHE_META_PATH.write_text(
        json.dumps({"dimensions": dimensions, "model_name": model_name}, indent=2),
        encoding="utf-8",
    )


def build_chunks(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    cfg = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for page in pages:
        if cfg["chunking"] == "fixed_size":
            chunks.extend(fixed_size(page, cfg["chunk_size"], cfg["overlap"]))
        else:
            chunks.extend(by_paragraph(page, cfg["chunk_size"]))
    return chunks


def embed_chunks(
    embedder: GeminiEmbedder | LocalEmbedder,
    chunks: list[Chunk],
    *,
    global_cache: dict[str, list[float]] | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    cache = global_cache if global_cache is not None else {}
    texts: list[str] = []
    keys: list[str] = []
    for ch in chunks:
        key = hashlib.sha256(ch.text.encode("utf-8")).hexdigest()
        keys.append(key)
        texts.append(ch.text)

    unique_texts: list[str] = []
    unique_keys: list[str] = []
    for key, text in zip(keys, texts, strict=True):
        cached = cache.get(key)
        if cached is None or len(cached) != embedder.dimensions:
            if cached is not None and len(cached) != embedder.dimensions:
                del cache[key]
            unique_keys.append(key)
            unique_texts.append(text)

    started = time.perf_counter()
    if unique_texts:
        print(f"Embedding {len(unique_texts)} new texts ({len(cache)} already cached)...")
        step = 25
        for i in range(0, len(unique_texts), step):
            batch_keys = unique_keys[i : i + step]
            batch_texts = unique_texts[i : i + step]
            batch_vecs = embedder.embed_documents(batch_texts)
            for key, vec in zip(batch_keys, batch_vecs, strict=True):
                cache[key] = vec
            save_embed_cache(
                cache,
                dimensions=embedder.dimensions,
                model_name=embedder.model_name,
            )
            done = min(i + step, len(unique_texts))
            print(f"  embedded {done}/{len(unique_texts)} new texts")
        vectors = [cache[k] for k in unique_keys]
    elapsed = time.perf_counter() - started

    ordered = [cache[k] for k in keys]
    stats = {
        "unique_embeddings": len(unique_texts),
        "total_chunks": len(chunks),
        "embedding_seconds": round(elapsed, 3),
    }
    return ordered, stats


def run_retrieval(
    store: PgVectorStore,
    embedder: GeminiEmbedder | LocalEmbedder,
    strategy: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cfg = STRATEGIES[strategy]
    traces: list[dict[str, Any]] = []
    for q in questions:
        qvec = embedder.embed_query(q["question"])
        filters = q.get("filter") if cfg["metadata_filter"] else None
        hits = store.search(strategy, qvec, top_k=3, filters=filters)
        traces.append(evaluate_query(q, hits))
    return traces


def write_results(
    out_dir: Path,
    *,
    metrics: dict[str, Any],
    traces_by_strategy: dict[str, list[dict[str, Any]]],
    ingest_stats: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "ingest_stats.json").write_text(json.dumps(ingest_stats, indent=2), encoding="utf-8")
    for strategy, traces in traces_by_strategy.items():
        (out_dir / f"traces_{strategy}.json").write_text(
            json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def update_experiment_log(
    metrics: dict[str, Any], ingest_stats: dict[str, Any], embedder_model: str
) -> None:
    log_path = REPO_ROOT / "artifacts" / "experiment_log.csv"
    rows = []
    for strategy, label in [("A", "A"), ("B", "B"), ("C", "C")]:
        cfg = STRATEGIES[strategy]
        m = metrics[strategy]
        stats = ingest_stats[strategy]
        rows.append(
            {
                "strategy": label,
                "chunking": cfg["chunking"],
                "chunk_size": cfg["chunk_size"],
                "overlap": cfg["overlap"],
                "metadata": "source,page,chapter",
                "filter": "chapter when question.filter" if cfg["metadata_filter"] else "none",
                "vectorizer": embedder_model,
                "hypothesis": {
                    "A": "Fixed windows hurt coherence but may improve exact-span retrieval.",
                    "B": "Paragraph chunks improve coherence vs fixed windows.",
                    "C": "Chapter filter reduces cross-chapter noise when filter provided.",
                }[strategy],
                "dev_hit1": f"{m['dev']['hit_at_1']}/{m['dev']['n']}",
                "dev_hit3": f"{m['dev']['hit_at_3']}/{m['dev']['n']}",
                "test_hit1": f"{m['test']['hit_at_1']}/{m['test']['n']}",
                "test_hit3": f"{m['test']['hit_at_3']}/{m['test']['n']}",
                "notes": f"chunks={stats['chunk_count']}; embed_s={stats.get('embedding_seconds', '')}",
            }
        )

    fieldnames = list(rows[0].keys())
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    load_dotenv(STARTER_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL missing in starter/.env")

    manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    bench = json.loads((REPO_ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = bench["questions"]

    pdf_path = _resolve_data_path(manifest)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Corpus PDF not found: {pdf_path}")

    pr = manifest["physical_page_range"]
    pages = propagate_chapters(
        load_pdf(
            pdf_path,
            manifest["source_id"],
            manifest["source_path_or_url"],
            page_start=pr["start"],
            page_end=pr["end"],
        )
    )
    if len(pages) < 30:
        raise RuntimeError(f"Only {len(pages)} content pages; need >= 30")

    embedder = create_embedder()
    store = PgVectorStore(database_url, embedder.dimensions)
    store.ensure_schema()

    ingest_stats: dict[str, Any] = {}
    chunks_by_strategy: dict[str, list[Chunk]] = {}
    embed_cache = load_embed_cache(embedder.dimensions, embedder.model_name)

    for strategy in STRATEGIES:
        print(f"\n=== Strategy {strategy} ===")
        started = time.perf_counter()
        chunks = build_chunks(pages, strategy)
        chunks_by_strategy[strategy] = chunks
        vectors, emb_stats = embed_chunks(embedder, chunks, global_cache=embed_cache)
        store.upsert(strategy, chunks, vectors)
        count = store.count_strategy(strategy)
        ingest_stats[strategy] = {
            "chunk_count": len(chunks),
            "stored_count": count,
            "ingest_seconds": round(time.perf_counter() - started, 3),
            **emb_stats,
        }

    # Idempotent re-ingest check (strategy B)
    b_count_before = store.count_strategy("B")
    vectors_b, _ = embed_chunks(
        embedder, chunks_by_strategy["B"], global_cache=embed_cache
    )
    store.upsert("B", chunks_by_strategy["B"], vectors_b)
    b_count_after = store.count_strategy("B")
    ingest_stats["reingest_B"] = {"before": b_count_before, "after": b_count_after}

    metrics: dict[str, Any] = {}
    traces_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for strategy in STRATEGIES:
        traces = run_retrieval(store, embedder, strategy, questions)
        traces_by_strategy[strategy] = traces
        summary = summarize(traces)
        summary["strategy"] = strategy
        metrics[strategy] = summary

    results_dir = REPO_ROOT / "results"
    write_results(results_dir, metrics=metrics, traces_by_strategy=traces_by_strategy, ingest_stats=ingest_stats)
    ingest_stats["embedding_model"] = embedder.model_name
    ingest_stats["embedding_dimensions"] = embedder.dimensions
    update_experiment_log(metrics, ingest_stats, embedder.model_name)

    print(json.dumps({"metrics": metrics, "ingest": ingest_stats}, indent=2))


if __name__ == "__main__":
    main()
