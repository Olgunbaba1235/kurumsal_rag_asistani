"""Dil modeli istemcisi (Google GenAI).

`LLM` protokolü sayesinde RAG motoru somut sağlayıcıya bağlı değildir;
testlerde sahte model enjekte edilir. İki model kullanılır:

* ``model``      — nihai yanıt üretimi (kalite öncelikli)
* ``fast_model`` — sorgu yeniden yazma ve reranking (gecikme öncelikli)
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

Message = tuple[str, str]  # ("user" | "assistant", metin)


class LLMError(Exception):
    """Dil modeli çağrısı başarısız oldu.

    ``user_message`` son kullanıcıya gösterilecek Türkçe açıklamadır (ham API hatası değil);
    ``code`` HTTP durum kodu (biliniyorsa), ``retryable`` geçici bir hata olup olmadığı.
    """

    def __init__(self, message: str, *, code: int | None = None, user_message: str | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = is_retryable(code)
        self.user_message = user_message or friendly_error(code)


RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable(code: int | None) -> bool:
    """429 (hız sınırı), 5xx (sunucu yoğunluğu) ve ağ hataları (code=None) geçicidir."""
    return code is None or code in RETRYABLE_CODES


def friendly_error(code: int | None) -> str:
    """HTTP koduna göre kullanıcıya gösterilecek Türkçe mesaj."""
    if code == 429:
        return ("Yapay zekâ servisinin istek sınırına ulaşıldı (429). Birkaç saniye bekleyip "
                "soruyu tekrar gönderin.")
    if code in (500, 502, 503, 504):
        return (f"Yapay zekâ servisi şu anda yoğun veya geçici olarak erişilemiyor ({code}). "
                "Birkaç saniye sonra tekrar deneyin.")
    if code in (401, 403):
        return "Gemini API anahtarı geçersiz veya yetkisiz. Yöneticinizden GOOGLE_API_KEY ayarını kontrol etmesini isteyin."
    if code == 404:
        return "Yapılandırılan Gemini modeli bulunamadı (404). LLM_MODEL / FAST_LLM_MODEL ayarlarını kontrol edin."
    if code == 400:
        return "Yapay zekâ servisi isteği reddetti (400). Soruyu kısaltıp tekrar deneyin."
    if code is None:
        return "Yapay zekâ servisine bağlanılamadı (ağ hatası). Bağlantınızı kontrol edip tekrar deneyin."
    return f"Yapay zekâ servisi beklenmeyen bir hata döndürdü ({code}). Lütfen tekrar deneyin."


def _transient_network_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


def call_with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0, what: str = "LLM") -> T:
    """Geçici hatalarda (429/5xx/ağ) üstel bekleme + rastgele sapmayla tekrar dener.

    Kalıcı hatalar (400/401/403/404) beklemeden `LLMError` olarak yükselir.
    """
    from google.genai import errors

    network = _transient_network_errors()
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, "code", None) or None
            failure = LLMError(f"Gemini hatası ({code}): {exc}", code=code)
        except network as exc:
            failure = LLMError(f"Gemini ağ hatası: {exc}", code=None)
        if not failure.retryable or attempt == attempts:
            raise failure
        delay = base_delay * 2 ** (attempt - 1) * (1 + random.random() * 0.25)
        logger.warning("%s geçici hata (%s), %.1fs sonra tekrar (%d/%d)…", what, failure.code, delay,
                       attempt, attempts - 1)
        time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


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
    retry_attempts = 3
    retry_base_delay = 1.0  # sn; testlerde 0'a çekilir

    def _retry(self, fn: Callable[[], T]) -> T:
        return call_with_retry(fn, attempts=self.retry_attempts, base_delay=self.retry_base_delay)

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
        """Akışlı yanıt. İlk parça gelmeden oluşan geçici hatalar (429/503…) tekrar denenir; akış
        başladıktan sonraki hata tekrar denenmez (kullanıcı yarım metni iki kez görmesin)."""
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
        )
        network = _transient_network_errors()
        for attempt in range(1, self.retry_attempts + 1):
            started = False
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=self.model, contents=self._contents(messages), config=config
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except errors.APIError as exc:
                code = getattr(exc, "code", None) or None
                failure = LLMError(f"Yanıt üretilemedi ({code}): {exc}", code=code)
            except network as exc:
                failure = LLMError(f"Yanıt üretilemedi (ağ): {exc}", code=None)
            if started or not failure.retryable or attempt == self.retry_attempts:
                raise failure
            delay = self.retry_base_delay * 2 ** (attempt - 1) * (1 + random.random() * 0.25)
            logger.warning("LLM akışı geçici hata (%s), %.1fs sonra tekrar…", failure.code, delay)
            time.sleep(delay)

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
