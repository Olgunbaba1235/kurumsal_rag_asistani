"""Multi-Agent orkestratör: niyet tespiti → paralel uzman ajanlar → kurumsal sentez.

1. **Niyet:** DOCUMENT_ONLY / SQL_ONLY / HYBRID. LLM (yapısal çıktı) karar verir; LLM yoksa ya da
   başarısızsa anahtar kelime sezgiseline düşülür. İstemci niyeti zorlayabilir.
2. **Paralel çalıştırma:** Hybrid RAG ve Text-to-SQL ``asyncio.gather`` ile eşzamanlı çalışır;
   bir kolun hatası diğerini düşürmez (rapora "caveat" olarak yansır).
3. **Sentez:** LLM yalnızca iki ajanın *doğrulanmış çıktısından* rapor yazar; yeni rakam/olgu
   eklememesi istenir. LLM yok/başarısızsa deterministik şablon raporu üretilir.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from app.schemas.agent import Intent
from app.services.analytics_service import data_confidence, heuristic_route
from app.services.rag_service import RAGService
from app.services.tenant_service import TenantContext
from app.services.text_to_sql_service import SCHEMA_DESCRIPTION, SQLAnswer, TextToSQLService
from src.llm import LLM, LLMError, Message, StructuredOutput
from src.rag_engine import RAGResponse

logger = logging.getLogger(__name__)

_ROUTE_TO_INTENT = {"documents": Intent.DOCUMENT_ONLY, "data": Intent.SQL_ONLY, "both": Intent.HYBRID}
_MAX_ROWS_FOR_SYNTHESIS = 20


class _IntentDecision(StructuredOutput):
    intent: Intent
    reason: str = Field(description="Kısa Türkçe gerekçe")


class _ReportDraft(StructuredOutput):
    title: str
    executive_summary: str
    key_findings: list[str]
    recommendations: list[str]
    caveats: list[str] = Field(default_factory=list)


_INTENT_SYSTEM = f"""Kurumsal bir asistanın niyet sınıflandırıcısısın. Soruyu şunlardan birine ata:
- DOCUMENT_ONLY: yalnızca şirket politikası/prosedür/yönetmelik belgelerinden yanıtlanır
- SQL_ONLY: yalnızca satış veritabanındaki sayısal/listesel veriden yanıtlanır
- HYBRID: hem belgelerdeki kural/limit hem de veri gerekir (ör. "onay limitini aşan siparişler")

Veritabanı şeması:
{SCHEMA_DESCRIPTION}"""

_SYNTH_SYSTEM = """Kıdemli bir B2B analistisin. Sana iki uzman ajanın çıktısı verilir: doküman (RAG) bulguları ve
SQL veri analizi. Bunlardan tutarlı bir Türkçe kurumsal rapor yaz.
KURALLAR: Yalnızca verilen çıktılardaki bilgi ve rakamları kullan; yeni rakam, kural veya olgu UYDURMA.
Doküman atıflarını ([Dosya - Sayfa X]) koru. Bir kaynak yoksa/başarısızsa bunu caveats içinde belirt.
Öneriler yalnızca bulgulardan türesin. Doküman kuralları ile veri çelişirse açıkça belirt."""


@dataclass(slots=True)
class OrchestratorResult:
    question: str
    intent: Intent
    intent_reason: str
    intent_method: str
    is_answered: bool
    confidence: int
    report: dict[str, Any]
    documents: RAGResponse | None
    data: SQLAnswer | None
    departments: list[str]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _template_report(
    question: str, docs: RAGResponse | None, data: SQLAnswer | None, caveats: list[str]
) -> dict[str, Any]:
    findings: list[str] = []
    recs: list[str] = []
    if docs is not None and docs.is_answered:
        findings.append(f"Doküman: {docs.answer}")
    if data is not None and data.success:
        findings.append(f"Veri: {data.summary}")
        findings.extend(f"İçgörü: {i}" for i in data.insights)
    if not findings:
        summary = "Mevcut doküman ve veri kaynaklarında bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
    else:
        summary = " ".join(findings[:2])
        recs.append("Bulguları ilgili kaynak belge ve sorgu sonuçlarıyla doğrulayarak karara dönüştürün.")
    return {
        "title": f"Kurumsal Analiz: {question.strip()[:120]}",
        "executive_summary": summary,
        "key_findings": findings,
        "recommendations": recs,
        "caveats": caveats,
        "synthesis_method": "template",
    }


class MultiAgentOrchestrator:
    def __init__(self, rag: RAGService, sql: TextToSQLService, llm: LLM | None):
        self.rag = rag
        self.sql = sql
        self.llm = llm

    async def classify(self, question: str, history: Sequence[Message]) -> tuple[Intent, str, str]:
        if self.llm is not None:
            ctx = "\n".join(f"{r}: {t[:200]}" for r, t in list(history)[-4:])
            prompt = f"GEÇMİŞ:\n{ctx}\n\nSORU: {question}" if ctx else f"SORU: {question}"
            try:
                d = await asyncio.to_thread(self.llm.generate_json, prompt, _IntentDecision, system=_INTENT_SYSTEM)
                return d.intent, d.reason, "llm"
            except LLMError as exc:
                logger.warning("Niyet LLM'i başarısız (%s); sezgisel yönteme düşülüyor.", exc)
        route, reason = heuristic_route(question)
        return _ROUTE_TO_INTENT[route], reason, "heuristic"

    async def _synthesize(
        self, question: str, docs: RAGResponse | None, data: SQLAnswer | None, caveats: list[str]
    ) -> dict[str, Any]:
        fallback = _template_report(question, docs, data, caveats)
        if self.llm is None or not fallback["key_findings"]:
            return fallback
        parts = [f"SORU: {question}"]
        if docs is not None and docs.is_answered:
            parts.append(f"DOKÜMAN AJANI ÇIKTISI:\n{docs.answer}")
        if data is not None and data.success:
            sample = {
                "sql_ozeti": data.summary,
                "sutunlar": data.columns,
                "satirlar": data.rows[:_MAX_ROWS_FOR_SYNTHESIS],
                "toplam_satir": data.row_count,
                "icgoruler": data.insights,
            }
            parts.append("SQL AJANI ÇIKTISI:\n" + json.dumps(sample, ensure_ascii=False, default=str))
        if caveats:
            parts.append("BİLİNEN SORUNLAR:\n" + "\n".join(caveats))
        try:
            d = await asyncio.to_thread(
                self.llm.generate_json, "\n\n".join(parts), _ReportDraft, system=_SYNTH_SYSTEM, fast=False
            )
        except Exception as exc:  # noqa: BLE001 — sentez başarısızlığı yanıtı düşürmemeli
            logger.warning("Sentez LLM'i başarısız (%s); şablon rapor kullanılıyor.", exc)
            return fallback
        report = d.model_dump()
        report["caveats"] = list(dict.fromkeys([*caveats, *d.caveats]))
        report["synthesis_method"] = "llm"
        return report

    async def run(
        self,
        tenant: TenantContext,
        question: str,
        departments: Sequence[str],
        history: Sequence[Message] = (),
        *,
        intent: Intent | None = None,
        top_k: int | None = None,
    ) -> OrchestratorResult:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        t0 = time.perf_counter()
        if intent is None:
            intent, reason, method = await self.classify(question, history)
        else:
            reason, method = "İstemci tarafından seçildi.", "client"
        timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        async def run_docs() -> tuple[RAGResponse, float]:
            t = time.perf_counter()
            r = await self.rag.ask(tenant, question, departments, history, top_k=top_k)
            return r, (time.perf_counter() - t) * 1000

        async def run_sql() -> tuple[SQLAnswer, float]:
            t = time.perf_counter()
            r = await self.sql.run(tenant, question, history)
            return r, (time.perf_counter() - t) * 1000

        tasks: dict[str, Any] = {}
        if intent in (Intent.DOCUMENT_ONLY, Intent.HYBRID):
            tasks["documents"] = run_docs()
        if intent in (Intent.SQL_ONLY, Intent.HYBRID):
            tasks["data"] = run_sql()
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)

        docs: RAGResponse | None = None
        data: SQLAnswer | None = None
        caveats: list[str] = []
        for name, outcome in zip(tasks, outcomes, strict=True):
            label = "Doküman" if name == "documents" else "Veri"
            if isinstance(outcome, BaseException):
                logger.exception("[%s] %s ajanı başarısız", tenant.slug, name, exc_info=outcome)
                msg = f"{label} ajanı başarısız oldu ({type(outcome).__name__})."
                warnings.append(msg)
                caveats.append(msg)
                continue
            result, ms = outcome
            timings[f"{name}_ms"] = round(ms, 1)
            warnings.extend(f"[{label.lower()}] {w}" for w in result.warnings)
            if name == "documents":
                docs = result
                if not result.is_answered:
                    caveats.append("Dokümanlarda soruyla ilgili yeterli bilgi bulunamadı.")
            else:
                data = result
                if not result.success:
                    caveats.append("Veri ajanı geçerli bir sorgu sonucu üretemedi.")

        t1 = time.perf_counter()
        report = await self._synthesize(question, docs, data, caveats)
        timings["synthesis_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        confidences: list[int] = []
        if docs is not None and docs.is_answered:
            confidences.append(docs.confidence)
        if data is not None and data.success:
            confidences.append(data_confidence(data))
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return OrchestratorResult(
            question=question.strip(),
            intent=intent,
            intent_reason=reason,
            intent_method=method,
            is_answered=bool(confidences),
            confidence=round(sum(confidences) / len(confidences)) if confidences else 0,
            report=report,
            documents=docs,
            data=data,
            departments=list(departments),
            timings_ms=timings,
            warnings=warnings,
        )
