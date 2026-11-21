"""Embedding sağlayıcıları.

`Embedder` protokolü sayesinde vektör deposu somut sağlayıcıya bağlı değildir;
testlerde sahte embedder enjekte edilir.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Modül düzeyinde import: fonksiyon içinden `from src…` yapmak, Streamlit sıcak yenilemesi `src.*`
# modüllerini sys.modules'tan çıkarırken aynı anda çalışırsa `KeyError: 'src'` üretir.
from src.llm import LLMError, call_with_retry
from src.text_utils import tokenize

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # Gemini embed_content tek istekte en fazla 100 girdi kabul eder


class EmbeddingError(Exception):
    """Embedding servisi hatası."""


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbedder:
    """Ağ gerektirmeyen, deterministik *geliştirme/çevrimdışı* embedder'ı.

    Kök (F5) token'ları ve karakter 3-gram'larını sabit boyutlu vektöre hash'ler.
    Anlamsal benzerliği yakalamaz; yalnızca API anahtarı olmadan hattı uçtan uca
    denemek ve testler içindir. Üretimde `GeminiEmbedder` kullanılır.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for tok in tokenize(text):
            features = [tok] + [tok[i : i + 3] for i in range(max(1, len(tok) - 2))]
            for j, feat in enumerate(features):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                v[h % self.dimension] += 1.0 if j == 0 else 0.5
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class GeminiEmbedder:
    """Google GenAI embedding (belge ve sorgu için ayrı `task_type`)."""

    def __init__(self, api_key: str | None, model: str = "gemini-embedding-001", dimension: int = 768):
        if not api_key:
            raise EmbeddingError("GOOGLE_API_KEY tanımlı değil.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def _call(self, batch: list[str], task_type: str, attempts: int = 3):
        from google.genai import types

        try:
            return call_with_retry(
                lambda: self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension),
                ),
                attempts=attempts, what="Embedding API",
            )
        except LLMError as exc:
            raise EmbeddingError(exc.user_message) from exc

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            response = self._call(list(texts[start : start + _BATCH_SIZE]), task_type)
            # 3072 altındaki boyutlar normalize gelmez → kosinüs benzerliği için normalize et
            vectors.extend(l2_normalize(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
