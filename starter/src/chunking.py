"""Fixed-size and structure-aware chunking."""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk_id(document_id: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{page}:{index}:{text[:80]}".encode()).hexdigest()[:16]
    return f"{document_id}-p{page}-c{index}-{digest}"


def _base_metadata(page: DocumentPage) -> dict:
    meta = dict(page.metadata)
    meta["document_id"] = page.document_id
    meta["source"] = page.source
    meta["page"] = page.page
    return meta


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    text = page.text
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    index = 0
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(page.document_id, page.page, index, piece),
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=piece,
                    metadata=_base_metadata(page),
                )
            )
            index += 1
        if end >= len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata."""
    raw_parts = re.split(r"\n\n+", page.text)
    paragraphs = [p.strip() for p in raw_parts if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer = ""
    index = 0

    def flush(buf: str) -> None:
        nonlocal index
        if not buf.strip():
            return
        piece = buf.strip()
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(page.document_id, page.page, index, piece),
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=piece,
                metadata=_base_metadata(page),
            )
        )
        index += 1

    for para in paragraphs:
        if len(para) > max_size:
            if buffer:
                flush(buffer)
                buffer = ""
            start = 0
            while start < len(para):
                end = min(start + max_size, len(para))
                piece = para[start:end].strip()
                if piece:
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(page.document_id, page.page, index, piece),
                            document_id=page.document_id,
                            source=page.source,
                            page=page.page,
                            text=piece,
                            metadata=_base_metadata(page),
                        )
                    )
                    index += 1
                start = end
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= max_size:
            buffer = candidate
        else:
            flush(buffer)
            buffer = para

    if buffer:
        flush(buffer)

    return chunks
