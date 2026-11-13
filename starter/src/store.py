"""Store and search vectors using PostgreSQL + pgvector."""
from __future__ import annotations

import json
from typing import Any

import psycopg
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions
        self._table_ready: set[str] = set()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def ensure_schema(self) -> None:
        """Create or recreate `chunks` when stored vector dimension does not match."""
        dim = int(self.dimensions)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                      SELECT 1 FROM information_schema.tables
                      WHERE table_schema = 'public' AND table_name = 'chunks'
                    )
                    """
                )
                exists = bool(cur.fetchone()[0])
                if exists:
                    cur.execute(
                        """
                        SELECT atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = 'public' AND c.relname = 'chunks'
                          AND a.attname = 'embedding' AND NOT a.attisdropped
                        """
                    )
                    row = cur.fetchone()
                    if row and row[0] > 0:
                        existing_dim = (int(row[0]) - 4) // 2
                        if existing_dim != dim:
                            cur.execute("DROP TABLE IF EXISTS chunks CASCADE")
                            conn.commit()
                            exists = False
            conn.commit()
        self.create_schema()

    def create_schema(self) -> None:
        """Create table with vector column and source/page/filter metadata."""
        dim = int(self.dimensions)
        ddl = f"""
            CREATE TABLE IF NOT EXISTS chunks (
                strategy TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                chapter INTEGER,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dim}) NOT NULL,
                PRIMARY KEY (strategy, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks (strategy);
            CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks (strategy, chapter);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def clear_strategy(self, strategy: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions and write stable IDs without duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for vec in vectors:
            if len(vec) != self.dimensions:
                raise ValueError("vector dimension mismatch")

        if strategy not in self._table_ready:
            self.create_schema()
            self._table_ready.add(strategy)

        self.clear_strategy(strategy)

        insert = """
            INSERT INTO chunks (
                strategy, chunk_id, document_id, source, page, chapter, text, metadata, embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector
            )
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                chapter = EXCLUDED.chapter,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, vector in zip(chunks, vectors, strict=True):
                    chapter = chunk.metadata.get("chapter")
                    meta = dict(chunk.metadata)
                    vec_literal = "[" + ",".join(str(float(v)) for v in vector) + "]"
                    cur.execute(
                        insert,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chapter,
                            chunk.text,
                            json.dumps(meta),
                            vec_literal,
                        ),
                    )
            conn.commit()

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")

        where_parts = ["strategy = %s"]
        params: list[Any] = [strategy]

        if filters:
            for key, value in filters.items():
                if key == "chapter":
                    where_parts.append("chapter = %s")
                    params.append(int(value))
                elif key in {"document_id", "source", "page"}:
                    where_parts.append(f"{key} = %s")
                    params.append(value)
                else:
                    where_parts.append("metadata ->> %s = %s")
                    params.extend([key, str(value)])

        vec_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"
        query = f"""
            SELECT chunk_id, document_id, source, page, chapter, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE {" AND ".join(where_parts)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [vec_literal, *params, vec_literal, top_k]

        hits: list[dict[str, Any]] = []
        with self._connect() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(query, params)
                for row in cur.fetchall():
                    meta = row["metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    hits.append(
                        {
                            "chunk_id": row["chunk_id"],
                            "document_id": row["document_id"],
                            "source": row["source"],
                            "page": row["page"],
                            "chapter": row["chapter"],
                            "text": row["text"],
                            "metadata": meta,
                            "score": float(row["score"]),
                        }
                    )
        return hits

    def count_strategy(self, strategy: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE strategy = %s", (strategy,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
