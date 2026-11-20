"""Multi-Agent orkestratör (`POST /api/v1/agent/query`) istek/yanıt şemaları."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatTurn


class Intent(StrEnum):
    DOCUMENT_ONLY = "DOCUMENT_ONLY"
    SQL_ONLY = "SQL_ONLY"
    HYBRID = "HYBRID"


class AgentQuery(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=2000,
        examples=["En çok ciro yapan 5 müşteri hangileri ve harcama onay prosedürüne göre kim onaylamalı?"],
    )
    intent: Intent | None = Field(default=None, description="Verilirse otomatik niyet tespiti atlanır")
    departments: list[str] | None = Field(default=None, description="Doküman araması için departmanlar")
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagSourceOut(BaseModel):
    label: str = Field(description="Örn. 'Politika.pdf - Sayfa 3'")
    source: str
    page: int | None = None
    department: str | None = None
    relevance: float | None = None


class SqlSummaryOut(BaseModel):
    success: bool
    sql: str
    tables: list[str]
    columns: list[str]
    row_count: int
    truncated: bool
    attempts: int = Field(description="SQL üretim/düzeltme deneme sayısı")
    summary: str
    insights: list[str]
    error: str | None = None


class B2BReport(BaseModel):
    """Sentezlenmiş kurumsal analiz raporu."""

    title: str
    executive_summary: str
    key_findings: list[str]
    recommendations: list[str]
    caveats: list[str] = Field(default_factory=list, description="Eksik/başarısız kol ve belirsizlikler")
    synthesis_method: Literal["llm", "template"]


class AgentResponse(BaseModel):
    question: str
    intent: Intent
    intent_reason: str
    intent_method: Literal["llm", "heuristic", "client"]
    is_answered: bool
    confidence: int = Field(ge=0, le=100)
    report: B2BReport
    rag_sources: list[RagSourceOut]
    sql: SqlSummaryOut | None
    data: dict[str, Any] | None = Field(default=None, description="Sonuç satırları, istatistikler ve Vega-Lite grafiği")
    timings_ms: dict[str, float]
    warnings: list[str]
