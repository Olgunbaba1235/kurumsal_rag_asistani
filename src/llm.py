"""Dil modeli istemcisi (Google GenAI).

`LLM` protokolü sayesinde RAG motoru somut sağlayıcıya bağlı değildir;
testlerde sahte model enjekte edilir. İki model kullanılır:

* ``model``      — nihai yanıt üretimi (kalite öncelikli)
* ``fast_model`` — sorgu yeniden yazma ve reranking (gecikme öncelikli)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

Message = tuple[str, str]  # ("user" | "assistant", metin)


class LLMError(Exception):
    """Dil modeli çağrısı başarısız oldu."""


@runtime_checkable
class LLM(Protocol):
    def stream(
        self, messages: Sequence[Message], *, system: str | None = None, temperature: float | None = None
    ) -> Iterator[str]: ...

    def generate_json(
        self, prompt: str, schema: type[M], *, system: str | None = None, fast: bool = True
    ) -> M: ...


class GeminiLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-3.5-flash",
        fast_model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.1,
    ):
        if not api_key:
            raise LLMError("GOOGLE_API_KEY tanımlı değil.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.fast_model = fast_model
        self.temperature = temperature

    # ------------------------------------------------------------ yardımcılar
    @staticmethod
    def _retry(fn: Callable[[], T], attempts: int = 3) -> T:
        from google.genai import errors

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except errors.APIError as exc:
                code = getattr(exc, "code", 0) or 0
                if (code == 429 or code >= 500) and attempt < attempts:
                    delay = 1.5 * 2 ** (attempt - 1)
                    logger.warning("LLM geçici hata (%s), %.1fs sonra tekrar…", code, delay)
                    time.sleep(delay)
                    continue
                raise LLMError(f"Gemini hatası ({code}): {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _contents(messages: Sequence[Message]):
        from google.genai import types

        return [
            types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])
            for role, text in messages
        ]

    # ------------------------------------------------------------ API
    def stream(
        self, messages: Sequence[Message], *, system: str | None = None, temperature: float | None = None
    ) -> Iterator[str]:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
        )
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=self._contents(messages), config=config
            ):
                if chunk.text:
                    yield chunk.text
        except errors.APIError as exc:
            raise LLMError(f"Yanıt üretilemedi ({getattr(exc, 'code', '?')}): {exc}") from exc

    def generate_json(self, prompt: str, schema: type[M], *, system: str | None = None, fast: bool = True) -> M:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._retry(
            lambda: self._client.models.generate_content(
                model=self.fast_model if fast else self.model, contents=prompt, config=config
            )
        )
        try:
            return schema.model_validate_json(response.text or "")
        except ValueError as exc:
            raise LLMError(f"Model geçersiz JSON döndürdü: {exc}") from exc
