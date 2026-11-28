"""Gerçek `google-genai` SDK'sı ile sözleşme testleri (gerçek ağa çıkılmaz).

SDK, `HttpOptions.base_url` ile bu dosyanın başlattığı **yerel bir HTTP sunucusuna** yönlendirilir. Böylece
SDK'nın istek kurma (şema dönüştürme, yapılandırma, URL), taşıma katmanı (google-genai 3.x `httpx2`, eski
sürümler `httpx` — hangisi kuruluysa), yanıt çözme (JSON, SSE akışı) ve hata eşleme kodu GERÇEKTEN çalışır;
sahte `google.genai` modülüyle yazılan birim testlerinin (tests/test_robustness.py) varsayımları kurulu SDK
sürümüne karşı doğrulanır:

* Tüm yapısal çıktı şemaları SDK tarafından kabul edilir; yanıt şemasında ``default`` ilan edilmez.
* 503 / "overloaded" / zaman aşımı → fast_model'e geçiş; 429 tekrar denenir ama model değişmez; 400 hemen döner.
* Akış (SSE) parçaları sırayla gelir; ilk parçadan önce 503 → yanıt fast_model'den akar.
* İstek zaman aşımı (`LLM_TIMEOUT_SECONDS`) LLM ve embedding istemcisinde gerçekten uygulanır ve hangi HTTP
  kütüphanesinden gelirse gelsin "zaman aşımı" olarak sınıflanır.
* SDK istek kuramazsa (ValueError/TypeError) tekrar denenmeyen `LLMError` olur; öneri/yeniden yazma çökmez.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

genai = pytest.importorskip("google.genai", reason="google-genai kurulu değil (requirements.txt)")

import src.embeddings as embeddings_module
import src.llm as llm_module
from src.embeddings import EmbeddingError, GeminiEmbedder
from src.llm import GeminiLLM, LLMError
from src.rag_engine import rewrite_query
from src.search_types import SearchResult
from src.suggestions import QuerySuggester, build_corpus_hint

MAIN, FAST = "gemini-3.5-flash", "gemini-3.5-flash-lite"
SHORT_TIMEOUT = 0.5  # sn; "takılan" sunucu yanıtı bundan çok daha geç gelir
STALL = 3.0


# ====================================================================== yerel Gemini sunucusu
@dataclass
class Recorded:
    path: str
    query: dict
    headers: dict
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body or b"{}")


@dataclass
class Step:
    status: int
    payload: bytes
    content_type: str = "application/json"
    delay: float = 0.0


@dataclass
class Wire:
    """Sunucunun aldığı istekler + sıradaki yanıt betiği."""

    script: list = field(default_factory=list)
    requests: list[Recorded] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def queue(self, *steps) -> Wire:
        self.script.extend(steps)
        return self

    def next(self, req: Recorded) -> Step:
        with self.lock:
            self.requests.append(req)
            if not self.script:
                return Step(500, json.dumps({"error": {"code": 500, "message": f"beklenmeyen istek {req.path}",
                                                       "status": "INTERNAL"}}).encode())
            step = self.script.pop(0)
        return step(req) if callable(step) else step

    @staticmethod
    def target(req: Recorded) -> tuple[str, str]:
        """(model, yöntem): /v1beta/models/gemini-3.5-flash:generateContent → ("gemini-3.5-flash", "generateContent")"""
        last = req.path.rsplit("/", 1)[-1]
        model, _, method = last.partition(":")
        return model, method

    @property
    def models(self) -> list[str]:
        return [self.target(r)[0] for r in self.requests]

    def body(self, i: int = -1) -> dict:
        return self.requests[i].json()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # test çıktısını kirletmesin
        pass

    def _body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        req = Recorded(url.path, parse_qs(url.query), {k.lower(): v for k, v in self.headers.items()}, self._body())
        step = self.server.wire.next(req)
        if step.delay:
            time.sleep(step.delay)
        try:
            self.send_response(step.status)
            self.send_header("Content-Type", step.content_type)
            self.send_header("Content-Length", str(len(step.payload)))
            self.end_headers()
            self.wfile.write(step.payload)
        except OSError:  # istemci zaman aşımıyla bağlantıyı kapattı
            self.close_connection = True

    do_GET = do_POST


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    httpd.wire = Wire()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def wire(server, monkeypatch) -> Wire:
    """SDK'yı yerel sunucuya yönlendirir (zaman aşımı ve tekrar ayarları üretimdeki `http_options`'tan gelir)."""
    server.wire = w = Wire()
    real = llm_module.http_options

    def local(types, timeout):
        return real(types, timeout).model_copy(update={"base_url": server.url})

    monkeypatch.setattr(llm_module, "http_options", local)
    monkeypatch.setattr(embeddings_module, "http_options", local)
    return w


def _candidate(text: str) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }


def ok_text(text: str, delay: float = 0.0) -> Step:
    return Step(200, json.dumps(_candidate(text), ensure_ascii=False).encode(), delay=delay)


def ok_json(obj: dict) -> Step:
    return ok_text(json.dumps(obj, ensure_ascii=False))


def sse(*chunks: str) -> Step:
    body = "".join(f"data: {json.dumps(_candidate(c), ensure_ascii=False)}\r\n\r\n" for c in chunks).encode()
    return Step(200, body, "text/event-stream")


def error(code: int, status: str, message: str) -> Step:
    return Step(code, json.dumps({"error": {"code": code, "message": message, "status": status}}).encode())


def stall() -> Step:
    """İstemcinin zaman aşımından çok sonra gelen yanıt."""
    return ok_text("çok geç", delay=STALL)


OVERLOADED = error(503, "UNAVAILABLE", "The model is overloaded. Please try again later.")


def make_llm(**kwargs) -> GeminiLLM:
    llm = GeminiLLM("test-key", MAIN, FAST, **kwargs)
    llm.retry_base_delay = 0.0
    return llm


def _walk_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _walk_keys(v)
    return keys


# ====================================================================== yapısal çıktı şemaları
SCHEMAS = [
    ("src.rag_engine", "_QueryRewrite", {"standalone_query": "Yıllık izin kaç gün?", "search_variants": ["izin"]}),
    ("src.hybrid_retriever", "_RerankOutput", {"scores": [{"index": 1, "score": 9}, {"index": 2, "score": 0}]}),
    ("src.auditor", "_AuditVerdict", {"verdict": "revised", "corrected_answer": "Yanıt [a.md - Sayfa 1]",
                                      "issues": ["sayı yok"]}),
    ("src.suggestions", "_QuerySuggestionOut", {"has_suggestion": True, "suggested_query": "Yıllık izin kaç gün?",
                                                "alternatives": []}),
    ("app.services.analytics_service", "_RouteDecision", {"route": "documents", "reason": "Politika sorusu."}),
    ("app.services.orchestrator_service", "_IntentDecision", {"intent": "HYBRID", "reason": "İkisi de."}),
    ("app.services.orchestrator_service", "_ReportDraft", {"title": "Rapor", "executive_summary": "Özet",
                                                           "key_findings": ["a"], "recommendations": ["b"],
                                                           "caveats": []}),
    ("app.services.text_to_sql_service", "SQLGeneration", {"sql": "SELECT 1", "explanation": "Bir.",
                                                           "chart": {"type": "bar", "x": "a", "y": "b",
                                                                     "title": None}}),
    ("app.services.text_to_sql_service", "DataSummary", {"summary": "Özet", "highlights": ["x"]}),
]


@pytest.mark.parametrize(("module", "name", "payload"), SCHEMAS, ids=[n for _, n, _ in SCHEMAS])
def test_every_structured_output_schema_is_accepted_by_the_sdk(wire, module, name, payload):
    schema = getattr(importlib.import_module(module), name)
    llm = make_llm()
    wire.queue(ok_json(payload))
    out = llm.generate_json("Soru: …", schema, system="Talimat", fast=True)
    assert out == schema.model_validate(payload)

    model, method = wire.target(wire.requests[0])
    assert (model, method) == (FAST, "generateContent")
    assert wire.requests[0].headers.get("x-goog-api-key") == "test-key"
    body = wire.body()
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["temperature"] == 0.0
    wire_schema = cfg.get("responseSchema") or cfg.get("responseJsonSchema")
    assert wire_schema, f"SDK yanıt şemasını göndermedi: {sorted(cfg)}"
    assert "default" not in _walk_keys(wire_schema)  # Gemini API varsayılan değeri reddeder
    assert body["systemInstruction"]["parts"][0]["text"] == "Talimat"
    assert body["contents"][0]["parts"][0]["text"] == "Soru: …"


def test_no_structured_schema_advertises_defaults():
    """Şema SDK'ya hangi yoldan verilirse verilsin (pydantic → JSON şeması) `default` içermez."""
    for module, name, _ in SCHEMAS:
        schema = getattr(importlib.import_module(module), name)
        assert "default" not in _walk_keys(schema.model_json_schema()), name


def test_invalid_json_from_model_is_an_llm_error(wire):
    llm = make_llm()
    wire.queue(ok_text("bu JSON değil"))
    with pytest.raises(LLMError, match="geçersiz JSON"):
        llm.generate_json("x", importlib.import_module("src.suggestions")._QuerySuggestionOut)


def test_sdk_request_building_errors_become_non_retryable_llm_errors(wire, monkeypatch):
    llm = make_llm()

    def broken(**kwargs):
        raise ValueError("Default value is not supported in the response schema for the Gemini API.")

    monkeypatch.setattr(llm._client.models, "generate_content", broken)
    with pytest.raises(LLMError) as info:
        llm.generate_json("x", importlib.import_module("src.rag_engine")._QueryRewrite)
    assert info.value.retryable is False and not wire.requests
    assert "istek oluşturulamadı" in info.value.user_message


# ====================================================================== hata eşleme ve yedek model
def test_503_overloaded_retries_then_falls_back_to_fast_model(wire):
    llm = make_llm()
    wire.queue(OVERLOADED, OVERLOADED, OVERLOADED, ok_json({"standalone_query": "q", "search_variants": []}))
    out = llm.generate_json("x", importlib.import_module("src.rag_engine")._QueryRewrite, fast=False)
    assert out.standalone_query == "q"
    assert wire.models == [MAIN, MAIN, MAIN, FAST] and llm.last_model == FAST


def test_overloaded_message_without_503_also_falls_back(wire):
    llm = make_llm()
    busy = error(500, "INTERNAL", "Model is overloaded")
    wire.queue(busy, busy, busy, ok_text("yedekten"))
    assert "".join(llm.stream([("user", "Merhaba")])) == "yedekten"
    assert wire.models == [MAIN, MAIN, MAIN, FAST]


def test_timeout_switches_to_fast_model_without_waiting_again(wire):
    llm = make_llm(timeout=SHORT_TIMEOUT)
    wire.queue(stall(), ok_json({"standalone_query": "q", "search_variants": []}))
    t0 = time.perf_counter()
    out = llm.generate_json("x", importlib.import_module("src.rag_engine")._QueryRewrite, fast=False)
    assert out.standalone_query == "q" and wire.models == [MAIN, FAST]
    assert time.perf_counter() - t0 < STALL - 0.5  # istek zaman aşımı gerçekten uygulandı


def test_timeout_without_fallback_is_a_turkish_timeout_error(wire):
    llm = make_llm(timeout=SHORT_TIMEOUT, fallback_to_fast=False)
    llm.retry_attempts = 1
    wire.queue(stall())
    with pytest.raises(LLMError) as info:
        list(llm.stream([("user", "x")]))
    assert info.value.timed_out and "zaman aşımı" in info.value.user_message


def test_rate_limit_is_retried_but_never_switches_model(wire):
    llm = make_llm()
    limit = error(429, "RESOURCE_EXHAUSTED", "Quota exceeded")
    wire.queue(limit, limit, limit)
    with pytest.raises(LLMError) as info:
        llm.generate_json("x", importlib.import_module("src.rag_engine")._QueryRewrite, fast=False)
    assert info.value.code == 429 and "429" in info.value.user_message
    assert wire.models == [MAIN, MAIN, MAIN]


@pytest.mark.parametrize(("code", "status"), [(400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED"),
                                              (404, "NOT_FOUND")])
def test_permanent_errors_fail_fast_with_turkish_message(wire, code, status):
    llm = make_llm()
    wire.queue(error(code, status, "nope"))
    with pytest.raises(LLMError) as info:
        list(llm.stream([("user", "x")]))
    assert info.value.code == code and not info.value.retryable and len(wire.requests) == 1
    assert "nope" not in info.value.user_message  # ham API metni kullanıcıya gösterilmez


def test_fallback_can_be_disabled(wire):
    llm = make_llm(fallback_to_fast=False)
    wire.queue(OVERLOADED, OVERLOADED, OVERLOADED)
    with pytest.raises(LLMError) as info:
        list(llm.stream([("user", "x")]))
    assert info.value.overloaded and wire.models == [MAIN, MAIN, MAIN]


# ====================================================================== akış (SSE)
def test_streaming_parses_sse_chunks_and_sends_history_roles(wire):
    llm = make_llm()
    wire.queue(sse("Yıllık izin ", "14 iş ", "günüdür."))
    history = [("user", "İzin?"), ("assistant", "Hangi izin?"), ("user", "Yıllık izin kaç gün?")]
    parts = list(llm.stream(history, system="Yalnızca bağlam."))
    assert parts == ["Yıllık izin ", "14 iş ", "günüdür."]
    req = wire.requests[0]
    assert wire.target(req) == (MAIN, "streamGenerateContent") and req.query.get("alt") == ["sse"]
    body = wire.body()
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][2]["parts"][0]["text"] == "Yıllık izin kaç gün?"
    assert body["systemInstruction"]["parts"][0]["text"] == "Yalnızca bağlam."
    assert llm.last_model == MAIN


def test_stream_overloaded_before_first_token_continues_on_fast_model(wire):
    llm = make_llm()
    wire.queue(OVERLOADED, OVERLOADED, OVERLOADED, sse("Yedek ", "model"))
    assert "".join(llm.stream([("user", "x")])) == "Yedek model"
    assert wire.models == [MAIN, MAIN, MAIN, FAST] and llm.last_model == FAST


def test_stream_timeout_switches_model_once(wire):
    llm = make_llm(timeout=SHORT_TIMEOUT)
    wire.queue(stall(), sse("tamam"))
    assert "".join(llm.stream([("user", "x")])) == "tamam"
    assert wire.models == [MAIN, FAST]


def test_turkish_and_long_text_round_trip(wire):
    llm = make_llm()
    text = "Çalışanların ğ/ü/ş/ı/ö/ç içeren uzun yanıtı. " * 200
    wire.queue(sse(text[:4000], text[4000:]))
    assert "".join(llm.stream([("user", "İŞÇİ ÖĞÜN " * 300)])) == text
    assert "İŞÇİ ÖĞÜN" in wire.body()["contents"][0]["parts"][0]["text"]


# ====================================================================== embedding
def _embeddings(req: Recorded) -> Step:
    body = req.json()
    n = len(body.get("requests") or body.get("contents") or [body.get("content")])
    vectors = [{"values": [float(i + 1)] + [0.0] * 3} for i in range(n)]
    return Step(200, json.dumps({"embeddings": vectors, "embedding": vectors[0]}).encode())


def test_embedder_contract(wire):
    emb = GeminiEmbedder("test-key", "gemini-embedding-001", dimension=4)
    wire.queue(_embeddings)
    vecs = emb.embed_documents(["Yıllık izin", "Masraf beyanı"])
    assert vecs == [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]  # L2 normalize
    model, method = wire.target(wire.requests[0])
    assert model == "gemini-embedding-001" and "mbed" in method
    raw = json.dumps(wire.body(), ensure_ascii=False)
    assert "RETRIEVAL_DOCUMENT" in raw and "Masraf beyanı" in raw


def test_embedder_timeout_is_applied_and_retried(wire, monkeypatch):
    monkeypatch.setattr(llm_module, "backoff_delay", lambda attempt, base: 0.0)
    emb = GeminiEmbedder("test-key", "gemini-embedding-001", dimension=4, timeout=SHORT_TIMEOUT)
    wire.queue(stall(), _embeddings)
    t0 = time.perf_counter()
    assert emb.embed_query("izin") == [1.0, 0.0, 0.0, 0.0]
    assert len(wire.requests) == 2 and time.perf_counter() - t0 < STALL - 0.5


def test_embedder_errors_are_turkish_embedding_errors(wire):
    emb = GeminiEmbedder("test-key", "gemini-embedding-001", dimension=4)
    wire.queue(error(403, "PERMISSION_DENIED", "denied"))
    with pytest.raises(EmbeddingError) as info:
        emb.embed_query("izin")
    assert "denied" not in str(info.value) and len(wire.requests) == 1


# ====================================================================== çekirdek bileşenler gerçek SDK ile
def test_query_rewrite_through_real_sdk(wire):
    llm = make_llm()
    wire.queue(ok_json({"standalone_query": "Yıllık izin kaç iş günüdür?", "search_variants": ["yıllık izin süresi"]}))
    out = rewrite_query("peki kaç gün?", [("user", "Yıllık izin"), ("assistant", "Yıllık izin kurala bağlıdır.")], llm)
    assert out.standalone == "Yıllık izin kaç iş günüdür?" and "yıllık izin süresi" in out.variants


def _hint():
    chunks = [
        SearchResult(id="1", text="## Yıllık İzin\n\nYıllık izin 14 iş günüdür.",
                     metadata={"source": "IK.md", "department": "İK", "section": "Yıllık İzin"}, score=1.0),
        SearchResult(id="2", text="## Hastalık İzni\n\nRapor gerekir.",
                     metadata={"source": "IK.md", "department": "İK", "section": "Hastalık İzni"}, score=1.0),
    ]
    return build_corpus_hint(chunks)


def test_query_suggestion_through_real_sdk(wire):
    llm = make_llm()
    wire.queue(ok_json({"has_suggestion": True, "suggested_query": "Yıllık izin kaç gündür?", "alternatives": []}))
    out = QuerySuggester(llm).suggest("yıllık izn kaç gün", _hint(), scope=("t",))
    assert out.suggested_query == "Yıllık izin kaç gündür?" and out.method == "llm"
    assert wire.models == [FAST]
    prompt = wire.body()["contents"][0]["parts"][0]["text"]
    assert "IK.md" in prompt and "Yıllık İzin" in prompt


def test_query_suggestion_survives_overloaded_fast_model(wire):
    llm = make_llm()
    wire.queue(OVERLOADED, OVERLOADED, OVERLOADED)
    out = QuerySuggester(llm).suggest("yıllık izn kaç gün", _hint(), scope=("t",))
    assert out.method == "heuristic" and out.suggested_query == "yıllık izin kaç gün"
    assert wire.models == [FAST, FAST, FAST]  # zaten hızlı modelde: geçiş yok, sezgisel yedek
