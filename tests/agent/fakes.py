"""Multi-Agent orkestrasyon testleri için izole sahte servisler (DB, LLM, ağ yok)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.schemas.agent import Intent
from app.services.orchestrator_service import OrchestratorResult, _template_report
from app.services.tenant_service import TenantContext
from app.services.text_to_sql_service import Attempt, SQLAnswer
from src.llm import LLMError
from src.rag_engine import Citation, Evidence, RAGResponse

TENANT_A = TenantContext(
    id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
    slug="ege-vadi",
    name="Ege Vadi Gıda A.Ş.",
    departments=("Genel", "Finans", "Satış"),
)
TENANT_B = TenantContext(
    id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    slug="kuzey-ege",
    name="Kuzey Ege Yağ Ltd.",
    departments=("Genel", "Üretim"),
)
KEY_A = "rag_test_key_tenant_a"
KEY_B = "rag_test_key_tenant_b"
LABEL = "Iskonto_Politikasi.pdf - Sayfa 2"


# ------------------------------------------------------------------ örnek ajan çıktıları
def make_rag_response(
    question: str = "soru",
    *,
    answered: bool = True,
    confidence: int = 82,
    warnings: Sequence[str] = (),
) -> RAGResponse:
    text = "Bir kalemdeki toplam efektif iskonto liste fiyatının %12'sini aşamaz."
    return RAGResponse(
        question=question,
        answer=f"İskonto tavanı %12'dir [{LABEL}]." if answered else "Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır.",
        is_answered=answered,
        confidence=confidence if answered else 0,
        citations=[Citation(LABEL, "Iskonto_Politikasi.pdf", 2, "Finans", 91.0, text)] if answered else [],
        evidence=[Evidence(LABEL, "Iskonto_Politikasi.pdf", 2, "Finans", text, 91.0, 0.83, 7.2, ["vector", "bm25"], answered)],
        rewritten_query=question,
        search_queries=[question],
        departments=["Finans"],
        timings_ms={"total_ms": 40.0},
        warnings=list(warnings),
        reranker="lexical",
    )


def make_sql_answer(
    question: str = "soru",
    *,
    success: bool = True,
    attempts: int = 1,
    rows: list[list[Any]] | None = None,
    truncated: bool = False,
    warnings: Sequence[str] = (),
) -> SQLAnswer:
    sql = "SELECT order_no, discount_pct FROM orders WHERE discount_pct > 12 LIMIT 500"
    rows = [["SO-2026-0018", 14.5]] if rows is None else rows
    tried = [Attempt(sql=sql, error="column x does not exist") for _ in range(attempts - 1)]
    tried.append(Attempt(sql=sql, error=None if success else "Yalnızca SELECT sorgularına izin verilir."))
    return SQLAnswer(
        question=question,
        success=success,
        sql=sql if success else "",
        explanation="Tavanı aşan satırlar" if success else "",
        columns=["order_no", "discount_pct"] if success else [],
        rows=rows if success else [],
        row_count=len(rows) if success else 0,
        truncated=truncated,
        tables=["orders"] if success else [],
        numeric_stats={"discount_pct": {"min": 14.5, "max": 14.5}} if success else {},
        insights=["1 sipariş tavanı aşıyor."] if success else [],
        chart={"mark": "bar", "data": {"values": [{"order_no": "SO-2026-0018", "discount_pct": 14.5}]}}
        if success else None,
        summary="1 sipariş %12 tavanını aşıyor (SO-2026-0018, %14,5)." if success else "",
        attempts=tried,
        timings_ms={"execute_ms": 3.0},
        warnings=list(warnings),
    )


def make_result(
    intent: Intent = Intent.HYBRID,
    *,
    tenant: TenantContext = TENANT_A,
    documents: RAGResponse | None = None,
    data: SQLAnswer | None = None,
    warnings: Sequence[str] = (),
) -> OrchestratorResult:
    """`MultiAgentOrchestrator.run` çıktısı taklidi (uç nokta eşleme testleri için)."""
    docs = documents if documents is not None else (
        make_rag_response() if intent in (Intent.DOCUMENT_ONLY, Intent.HYBRID) else None
    )
    sql = data if data is not None else (make_sql_answer() if intent in (Intent.SQL_ONLY, Intent.HYBRID) else None)
    report = _template_report(f"{tenant.slug} sorusu", docs, sql, [])
    answered = bool(report["key_findings"])
    return OrchestratorResult(
        question="soru",
        intent=intent,
        intent_reason="Test niyeti",
        intent_method="llm",
        is_answered=answered,
        confidence=88 if answered else 0,
        report=report,
        documents=docs,
        data=sql,
        departments=list(tenant.departments),
        timings_ms={"intent_ms": 1.5, "synthesis_ms": 2.0, "total_ms": 50.0},
        warnings=list(warnings),
    )


# ------------------------------------------------------------------ sahte uzman ajanlar
@dataclass
class FakeRAG:
    """Doküman ajanı (`RAGService.ask`) taklidi."""

    response: RAGResponse | None = None
    error: BaseException | None = None
    before: Callable[[], Any] | None = None  # async kanca (eşzamanlılık testleri)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def ask(self, tenant, question, departments, history=(), *, top_k=None):  # noqa: ANN001, ANN201
        self.calls.append({"tenant": tenant, "question": question, "departments": list(departments),
                           "history": list(history), "top_k": top_k})
        if self.before is not None:
            await self.before()
        if self.error is not None:
            raise self.error
        return self.response or make_rag_response(question)


@dataclass
class FakeSQL:
    """SQL ajanı (`TextToSQLService.run`) taklidi."""

    answer: SQLAnswer | None = None
    error: BaseException | None = None
    before: Callable[[], Any] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def run(self, tenant, question, history=()):  # noqa: ANN001, ANN201
        self.calls.append({"tenant": tenant, "question": question, "history": list(history)})
        if self.before is not None:
            await self.before()
        if self.error is not None:
            raise self.error
        return self.answer or make_sql_answer(question)


class FakeLLM:
    """Niyet (`_IntentDecision`) ve sentez (`_ReportDraft`) için yapısal çıktı üreten LLM taklidi.

    `handlers[şema_adı]` bir dict, çağrılabilir (prompt → dict) ya da istisna olabilir;
    tanımlı olmayan şema için `LLMError` fırlatılır.
    """

    def __init__(self, **handlers: Any):
        self.handlers = handlers
        self.calls: list[tuple[str, str, str | None]] = []

    def generate_json(self, prompt, schema, *, system=None, fast=True):  # noqa: ANN001, ANN201
        self.calls.append((schema.__name__, prompt, system))
        h = self.handlers.get(schema.__name__)
        if h is None:
            raise LLMError("testte yapısal çıktı yok")
        if isinstance(h, BaseException):
            raise h
        return schema.model_validate(h(prompt) if callable(h) else h)

    def prompts(self, schema_name: str) -> list[str]:
        return [p for n, p, _ in self.calls if n == schema_name]

    def stream(self, messages, *, system=None, temperature=None):  # noqa: ANN001, ANN201
        yield ""


class MockOrchestrator:
    """`container.orchestrator` taklidi (`/agent/query` uç nokta testleri için)."""

    def __init__(self, result_fn: Callable[[TenantContext], OrchestratorResult] | None = None,
                 error: BaseException | None = None):
        self.result_fn = result_fn or (lambda tenant: make_result(tenant=tenant))
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def run(self, tenant, question, departments, history=(), *, intent=None, top_k=None):  # noqa: ANN001, ANN201
        self.calls.append({"tenant": tenant, "question": question, "departments": list(departments),
                           "history": list(history), "intent": intent, "top_k": top_k})
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return self.result_fn(tenant)
