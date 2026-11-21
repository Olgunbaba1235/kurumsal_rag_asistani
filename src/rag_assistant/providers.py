"""Model sağlayıcıları (Google GenAI).

`Embedder` ve `ChatModel` protokolleri sayesinde pipeline somut sağlayıcıya
bağlı değildir; testlerde sahte (fake) uygulamalar enjekte edilir, ileride
farklı bir sağlayıcıya geçmek tek bir sınıf yazmak kadar kolaydır.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from rag_assistant.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Gemini embed_content tek istekte en fazla 100 girdi kabul eder.
_EMBED_BATCH_SIZE = 100


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class ChatModel(Protocol):
    def stream(
        self,
        system_prompt: str,
        messages: Sequence[tuple[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]: ...


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _with_retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.5) -> T:
    """Geçici API hatalarında (429/5xx) üstel geri çekilmeyle tekrar dener."""
    from google.genai import errors

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, "code", None) or 0
            transient = code == 429 or code >= 500
            if not transient or attempt == attempts:
                raise LLMError(f"Google GenAI hatası ({code}): {exc}") from exc
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Geçici API hatası (%s), %.1fs sonra tekrar deneniyor…", code, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _make_client(api_key: str | None):
    if not api_key:
        raise ConfigurationError(
            "GOOGLE_API_KEY tanımlı değil. `.env` dosyasına ekleyin veya arayüzden girin."
        )
    from google import genai

    return genai.Client(api_key=api_key)


class GeminiEmbedder:
    def __init__(self, api_key: str | None, model: str = "gemini-embedding-001", dimension: int = 768):
        self._client = _make_client(api_key)
        self.model = model
        self.dimension = dimension

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = list(texts[start : start + _EMBED_BATCH_SIZE])
            response = _with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )
            )
            # 3072 altındaki boyutlar normalize edilmiş gelmez → kosinüs için normalize et.
            vectors.extend(_l2_normalize(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChatModel:
    def __init__(self, api_key: str | None, model: str = "gemini-3.5-flash", temperature: float = 0.2):
        self._client = _make_client(api_key)
        self.model = model
        self.temperature = temperature

    def stream(
        self,
        system_prompt: str,
        messages: Sequence[tuple[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])
            for role, text in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature if temperature is None else temperature,
        )
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except errors.APIError as exc:
            raise LLMError(f"Yanıt üretilemedi ({getattr(exc, 'code', '?')}): {exc}") from exc
