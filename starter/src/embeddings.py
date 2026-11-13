"""Choose a real embedding provider (API or local) and implement this contract."""
from __future__ import annotations

import os
import re
import time
from typing import Protocol

from google import genai
from google.genai import types
from google.genai.errors import ClientError


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class GeminiEmbedder:
    """Gemini API embeddings — no silent fallback to mock vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        dimensions: int | None = None,
        batch_size: int = 16,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) missing; set it in starter/.env"
            )
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIM", "768"))
        self.batch_size = batch_size
        self.client = genai.Client(api_key=api_key)
        self._min_interval = float(os.getenv("GEMINI_EMBED_MIN_INTERVAL_SEC", "0.72"))
        self._last_call_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.perf_counter() - self._last_call_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _retry_delay(self, exc: Exception) -> float:
        message = str(exc)
        if "per day" in message.lower() or "perday" in message.lower():
            raise RuntimeError(
                "Gemini embedding quota theo NGÀY đã hết (free tier ~1000 request/ngày). "
                "Chờ quota reset (API thường ghi 'retry in …' vài giờ), hoặc chạy "
                "EMBEDDING_PROVIDER=local, hoặc dùng key/dự án khác. Chi tiết: results/gemini_quota_note.md"
            ) from exc
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0
        if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
            return 35.0
        return 3.0

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            last_err: Exception | None = None
            for attempt in range(20):
                try:
                    self._throttle()
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimensions,
                        ),
                    )
                    self._last_call_at = time.perf_counter()
                    emb = result.embeddings[0].values
                    if len(emb) != self.dimensions:
                        raise ValueError(
                            f"Expected dim {self.dimensions}, got {len(emb)}"
                        )
                    vectors.append(list(emb))
                    break
                except Exception as exc:  # noqa: BLE001 — surface API failures
                    last_err = exc
                    delay = self._retry_delay(exc)
                    time.sleep(delay)
            else:
                raise RuntimeError(f"Gemini embed failed after retries: {last_err}") from last_err
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._embed_batch([question], "RETRIEVAL_QUERY")[0]


class LocalEmbedder:
    """Local Sentence Transformers — used only when GEMINI_API_KEY is unset."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dimensions = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]


def create_embedder() -> GeminiEmbedder | LocalEmbedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "gemini":
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is empty")
        return GeminiEmbedder()
    if provider == "auto" and api_key:
        return GeminiEmbedder()
    if provider == "local" or provider == "auto":
        print(
            "WARNING: GEMINI_API_KEY missing — using local SentenceTransformer. "
            "Set GEMINI_API_KEY and EMBEDDING_PROVIDER=gemini for the lab target model."
        )
        return LocalEmbedder()
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER={provider}")


MyEmbedder = GeminiEmbedder
