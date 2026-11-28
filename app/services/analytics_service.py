"""Birleşik B2B analitik: doküman (RAG) + sayısal veri (Text-to-SQL).

1. **Yönlendirme** — soru doküman bilgisi mi, sayısal veri mi, ikisi birden mi
   gerektiriyor? LLM (yapısal çıktı) karar verir; LLM yoksa/başarısızsa anahtar
   kelime sezgiseli kullanılır. İstemci ``mode`` ile kararı zorlayabilir.
2. **Paralel çalıştırma** — seçilen kollar ``asyncio.gather`` ile eşzamanlı çalışır;
   bir kolun hatası diğerini düşürmez (uyarı olarak raporlanır).
3. **Birleştirme** — yanıt metni ayrıca LLM'e *yeniden yazdırılmaz*: her bölüm kendi
   doğrulanmış kaynağından gelir (RAG: atıflı yanıt, SQL: sorgu sonucuna dayalı özet).
   Böylece birleştirme adımı halüsinasyon riski eklemez.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from app.services.rag_service import RAGService
from app.services.tenant_service import TenantContext
from app.services.text_to_sql_service import SCHEMA_DESCRIPTION, SQLAnswer, TextToSQLService
from src.llm import LLM, LLMError, Message, StructuredOutput
from src.rag_engine import NO_INFO_ANSWER, RAGResponse
from src.text_utils import fold

logger = logging.getLogger(__name__)

Route = Literal["documents", "data", "both"]

_DATA_HINTS = re.compile(
    r"\b(kac|toplam|ciro|satis|satilan|siparis|musteri|urun|stok|ortalama|en cok|en az|en yuksek|en dusuk|"
    # Not: "yıllık" bilinçli olarak yok — İK'daki "yıllık izin" sorularını veri koluna yönlendirirdi
    r"trend|aylik|ceyrek|donem|kar|marj|tutar|adet|miktar|segment|bolge|kanal|sku|fatura|"
    r"listele|sirala|karsilastir|grafik|20\d\d)",
)
_DOC_HINTS = re.compile(
    r"\b(politika|prosedur|yonetmelik|kural|izin|onay|yetki|surec|nasil|kim onaylar|gerekir|zorunlu|"
    r"belge|dokuman|madde|limit(i|ine|ler)? nedir|harcirah|avans|masraf beyan|uzaktan calisma)",
)


class _RouteDecision(StructuredOutput):
    route: Route
    reason: str = Field(description="Kısa Türkçe gerekçe")


_ROUTE_SYSTEM = f"""Kurumsal bir asistanın yönlendiricisisin. Soruyu yanıtlamak için hangi kaynağın gerektiğine karar ver:
- "documents": şirket politikaları, prosedürler, yönetmelikler, kurallar (metin belgeleri)
- "data": satış veritabanındaki sayısal/listesel bilgiler (müşteri, ürün, sipariş, ciro, stok)
- "both": soru hem bir kural/limit hem de veriye uygulanmasını içeriyorsa
  (ör. "prosedürdeki onay limitini aşan siparişler hangileri?")

Veritabanı şeması:
{SCHEMA_DESCRIPTION}"""


def heuristic_route(question: str) -> tuple[Route, str]:
    q = fold(question)
    data, docs = bool(_DATA_HINTS.search(q)), bool(_DOC_HINTS.search(q))
    if data and docs:
        return "both", "Soru hem kural/prosedür hem sayısal veri ifadeleri içeriyor."
    if data:
        return "data", "Soru sayısal/listesel veri ifadeleri içeriyor."
    return "documents", "Soru doküman bilgisi gerektiriyor (varsayılan)."


@dataclass(slots=True)
class SourceRef:
    type: Literal["document", "database"]
    label: str
    detail: dict = field(default_factory=dict)


@dataclass(slots=True)
class AnalyticsResult:
    question: str
    route: Route
    route_reason: str
    route_method: str
    answer: str
    is_answered: bool
    confidence: int
    sources: list[SourceRef]
    documents: RAGResponse | None
    data: SQLAnswer | None
    departments: list[str]
    timings_ms: dict[str, float]
    warnings: list[str]


def data_confidence(answer: SQLAnswer) -> int:
    """Veri yanıtı güveni: sonuç doğrudan veritabanından gelir; düzeltme denemeleri ve
    boş/kısaltılmış sonuçlar güveni düşürür."""
    if not answer.success:
        return 0
    score = 95 - 10 * (len(answer.attempts) - 1)
    if answer.row_count == 0:
        score = min(score, 50)
    if answer.truncated:
        score -= 10
    return max(0, min(100, score))


class AnalyticsService:
    def __init__(self, rag: RAGService, sql: TextToSQLService, llm: LLM | None):
        self.rag = rag
        self.sql = sql
        self.llm = llm

    async def route(self, question: str, history: Sequence[Message]) -> tuple[Route, str, str]:
        if self.llm is not None:
            ctx = "\n".join(f"{r}: {t[:200]}" for r, t in list(history)[-4:])
            prompt = f"GEÇMİŞ:\n{ctx}\n\nSORU: {question}" if ctx else f"SORU: {question}"
            try:
                d = await asyncio.to_thread(self.llm.generate_json, prompt, _RouteDecision, system=_ROUTE_SYSTEM)
                return d.route, d.reason, "llm"
            except LLMError as exc:
                logger.warning("Yönlendirme LLM'i başarısız (%s); sezgisel yönteme düşülüyor.", exc)
        route, reason = heuristic_route(question)
        return route, reason, "heuristic"

    async def query(
        self,
        tenant: TenantContext,
        question: str,
        departments: Sequence[str],
        history: Sequence[Message] = (),
        *,
        mode: Literal["auto", "documents", "data", "both"] = "auto",
        top_k: int | None = None,
    ) -> AnalyticsResult:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        t0 = time.perf_counter()
        if mode == "auto":
            route, reason, method = await self.route(question, history)
        else:
            route, reason, method = mode, "İstemci tarafından seçildi.", "client"
        timings["route_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        async def run_docs() -> tuple[RAGResponse, float]:
            t = time.perf_counter()
            r = await self.rag.ask(tenant, question, departments, history, top_k=top_k)
            return r, (time.perf_counter() - t) * 1000

        async def run_data() -> tuple[SQLAnswer, float]:
            t = time.perf_counter()
            r = await self.sql.run(tenant, question, history)
            return r, (time.perf_counter() - t) * 1000

        tasks = {}
        if route in ("documents", "both"):
            tasks["documents"] = run_docs()
        if route in ("data", "both"):
            tasks["data"] = run_data()
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)

        docs: RAGResponse | None = None
        data: SQLAnswer | None = None
        for name, outcome in zip(tasks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.exception("[%s] %s kolu başarısız", tenant.slug, name, exc_info=outcome)
                warnings.append(f"{'Doküman' if name == 'documents' else 'Veri'} analizi başarısız: {type(outcome).__name__}")
                continue
            result, ms = outcome
            timings[f"{name}_ms"] = round(ms, 1)
            if name == "documents":
                docs = result
                warnings.extend(f"[doküman] {w}" for w in result.warnings)
            else:
                data = result
                warnings.extend(f"[veri] {w}" for w in result.warnings)

        sections: list[str] = []
        confidences: list[int] = []
        sources: list[SourceRef] = []
        if docs is not None and docs.is_answered:
            sections.append(f"📄 **Doküman bulguları**\n{docs.answer}")
            confidences.append(docs.confidence)
            sources += [
                SourceRef(
                    "document",
                    c.label,
                    {"source": c.source, "page": c.page, "department": c.department, "relevance": c.relevance},
                )
                for c in docs.citations
            ]
        if data is not None and data.success:
            sections.append(f"📊 **Veri analizi**\n{data.summary}")
            confidences.append(data_confidence(data))
            sources.append(
                SourceRef(
                    "database",
                    f"Satış veritabanı: {', '.join(data.tables)}",
                    {"tables": data.tables, "row_count": data.row_count, "sql": data.sql},
                )
            )

        is_answered = bool(sections)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return AnalyticsResult(
            question=question.strip(),
            route=route,
            route_reason=reason,
            route_method=method,
            answer="\n\n".join(sections) if sections else NO_INFO_ANSWER,
            is_answered=is_answered,
            confidence=round(sum(confidences) / len(confidences)) if confidences else 0,
            sources=sources,
            documents=docs,
            data=data,
            departments=list(departments),
            timings_ms=timings,
            warnings=warnings,
        )
