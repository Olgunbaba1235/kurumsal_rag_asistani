"""Dil modeli istemcisi (Google GenAI).

`LLM` protokolü sayesinde RAG motoru somut sağlayıcıya bağlı değildir;
testlerde sahte model enjekte edilir. İki model kullanılır:

* ``model``      — nihai yanıt üretimi (kalite öncelikli)
* ``fast_model`` — sorgu yeniden yazma ve reranking (gecikme öncelikli)

Dayanıklılık: her istek ``timeout`` saniyede (varsayılan 30) kesilir. Geçici hatalar (429/5xx/ağ) üstel
beklemeyle tekrar denenir; ana model buna rağmen 503 / "overloaded" / zaman aşımı verirse istek
``fast_model`` ile tekrarlanır (``fallback_to_fast``). Zaman aşımında aynı model tekrar beklenmez.
"""

from __future__ import annotations

import logging
import random
import re
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

    def __init__(self, message: str, *, code: int | None = None, user_message: str | None = None,
                 timed_out: bool = False):
        super().__init__(message)
        self.code = code
        self.timed_out = timed_out
        self.retryable = is_retryable(code)
        self.user_message = user_message or friendly_error(code, timed_out=timed_out)

    @property
    def overloaded(self) -> bool:
        """503 Service Unavailable veya mesajında "overloaded/unavailable" geçen sunucu yoğunluğu."""
        return self.code == 503 or bool(_OVERLOAD_RE.search(str(self)))

    @property
    def should_fallback(self) -> bool:
        """Ana model yerine fast_model denenmeli mi? (yoğunluk veya zaman aşımı; 429/4xx değil)"""
        return self.overloaded or self.timed_out


RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})
_OVERLOAD_RE = re.compile(r"overload|\bunavailable\b", re.IGNORECASE)
MAX_BACKOFF_SECONDS = 8.0


def is_retryable(code: int | None) -> bool:
    """429 (hız sınırı), 5xx (sunucu yoğunluğu) ve ağ hataları (code=None) geçicidir."""
    return code is None or code in RETRYABLE_CODES


def friendly_error(code: int | None, *, timed_out: bool = False) -> str:
    """HTTP koduna göre kullanıcıya gösterilecek Türkçe mesaj."""
    if timed_out:
        return ("Yapay zekâ servisi zamanında yanıt vermedi (zaman aşımı). Servis yoğun olabilir; "
                "birkaç saniye sonra tekrar deneyin.")
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


def _timeout_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [TimeoutError]
    try:
        import httpx

        errors.append(httpx.TimeoutException)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


def _classify(exc: BaseException, what: str) -> LLMError:
    """google-genai / ağ istisnasını `LLMError`'a çevirir (zaman aşımı ayrıca işaretlenir)."""
    if isinstance(exc, _timeout_errors()):
        return LLMError(f"{what}: zaman aşımı ({exc})", code=None, timed_out=True)
    code = getattr(exc, "code", None)
    code = code if isinstance(code, int) and code else None
    return LLMError(f"{what} ({code}): {exc}", code=code)


def backoff_delay(attempt: int, base_delay: float) -> float:
    """Üstel bekleme (1, 2, 4… × base) + %0–25 rastgele sapma, en çok MAX_BACKOFF_SECONDS."""
    return min(base_delay * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS) * (1 + random.random() * 0.25)


def _transient_network_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


def call_with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0, what: str = "LLM",
                    fail_fast_on_timeout: bool = False) -> T:
    """Geçici hatalarda (429/5xx/ağ) üstel bekleme + rastgele sapmayla tekrar dener.

    Kalıcı hatalar (400/401/403/404) beklemeden `LLMError` olarak yükselir. `fail_fast_on_timeout=True`
    ise zaman aşımı aynı modelde tekrar beklenmez (çağıran yedek modele geçer).
    """
    from google.genai import errors

    network = _transient_network_errors()
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (errors.APIError, *network) as exc:
            failure = _classify(exc, f"{what} hatası")
        if not failure.retryable or attempt == attempts or (failure.timed_out and fail_fast_on_timeout):
            raise failure
        delay = backoff_delay(attempt, base_delay)
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
        *,
        timeout: float = 30.0,
        fallback_to_fast: bool = True,
    ):
        if not api_key:
            raise LLMError("GOOGLE_API_KEY tanımlı değil.")
        from google import genai
        from google.genai import types

        # İstek başına üst sınır: google-genai zaman aşımını milisaniye olarak alır
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        self.model = model
        self.fast_model = fast_model
        self.temperature = temperature
        self.timeout = timeout
        self.fallback_to_fast = fallback_to_fast
        self.last_model: str | None = None  # son başarılı çağrıyı yanıtlayan model (gözlem/log için)

    # ------------------------------------------------------------ yardımcılar
    retry_attempts = 3
    retry_base_delay = 1.0  # sn; testlerde 0'a çekilir

    def _retry(self, fn: Callable[[], T], *, fail_fast_on_timeout: bool = False) -> T:
        return call_with_retry(fn, attempts=self.retry_attempts, base_delay=self.retry_base_delay,
                               fail_fast_on_timeout=fail_fast_on_timeout)

    def _fallback_for(self, model: str, failure: LLMError) -> str | None:
        """Ana model yoğun/zaman aşımındaysa fast_model'e geçilir (zaten fast_model ise geçiş yok)."""
        if self.fallback_to_fast and model != self.fast_model and failure.should_fallback:
            return self.fast_model
        return None

    def _call_with_fallback(self, op: Callable[[str], T], model: str) -> T:
        """`op(model)`'i tekrar denemelerle çağırır; başarısız olursa uygun durumda fast_model'e geçer."""
        can_fallback = self.fallback_to_fast and model != self.fast_model
        try:
            result = self._retry(lambda: op(model), fail_fast_on_timeout=can_fallback)
            self.last_model = model
            return result
        except LLMError as failure:
            fallback = self._fallback_for(model, failure)
            if fallback is None:
                raise
            logger.warning("%s kullanılamıyor (%s); yedek model %s ile deneniyor.", model,
                           "zaman aşımı" if failure.timed_out else failure.code, fallback)
        result = self._retry(lambda: op(fallback))
        self.last_model = fallback
        return result

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
        başladıktan sonraki hata tekrar denenmez (kullanıcı yarım metni iki kez görmesin).

        Ana model ilk parçadan önce 503/overloaded/zaman aşımı ile tükenirse yanıt fast_model'den akar.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
        )
        contents = self._contents(messages)
        state = {"started": False}
        try:
            yield from self._stream_model(self.model, contents, config, state)
            self.last_model = self.model
            return
        except LLMError as failure:
            fallback = None if state["started"] else self._fallback_for(self.model, failure)
            if fallback is None:
                raise
            logger.warning("%s akışı kullanılamıyor (%s); yedek model %s ile deneniyor.", self.model,
                           "zaman aşımı" if failure.timed_out else failure.code, fallback)
        yield from self._stream_model(fallback, contents, config, state)
        self.last_model = fallback

    def _stream_model(self, model: str, contents, config, state: dict) -> Iterator[str]:
        from google.genai import errors

        network = _transient_network_errors()
        fail_fast = self.fallback_to_fast and model != self.fast_model
        for attempt in range(1, self.retry_attempts + 1):
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        state["started"] = True
                        yield chunk.text
                return
            except (errors.APIError, *network) as exc:
                failure = _classify(exc, "Yanıt üretilemedi")
            if (state["started"] or not failure.retryable or attempt == self.retry_attempts
                    or (failure.timed_out and fail_fast)):
                raise failure
            delay = backoff_delay(attempt, self.retry_base_delay)
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
        response = self._call_with_fallback(
            lambda model: self._client.models.generate_content(model=model, contents=prompt, config=config),
            self.fast_model if fast else self.model,
        )
        try:
            return schema.model_validate_json(response.text or "")
        except ValueError as exc:
            raise LLMError(f"Model geçersiz JSON döndürdü: {exc}") from exc
