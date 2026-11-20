"""Ortak test fikstürleri — API anahtarı gerektirmeyen sahte sağlayıcılar."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence

import pytest

from rag_assistant.config import Settings
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.vector_store import ChromaVectorStore

DIM = 64


class FakeEmbedder:
    """Kelime-torbası hashing embedder: aynı kelimeleri paylaşan metinler yakın düşer."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def stream(self, system_prompt, messages, temperature=None) -> Iterator[str]:
        self.calls.append((system_prompt, list(messages)))
        yield "Yıllık izin 14 gündür "
        yield "[1]."


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        CHROMA_DIR=tmp_path / "chroma",
        UPLOAD_DIR=tmp_path / "uploads",
        CHUNK_SIZE=300,
        CHUNK_OVERLAP=50,
        MIN_RELEVANCE=0.05,
        _env_file=None,
    )


@pytest.fixture
def store() -> ChromaVectorStore:
    import uuid

    return ChromaVectorStore(FakeEmbedder(), persist_dir=None, collection_name=f"t_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def pipeline(settings, store) -> RAGPipeline:
    settings.ensure_dirs()
    return RAGPipeline(settings, store.embedder, FakeChatModel(), store)
