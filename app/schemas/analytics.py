from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatResponse, ChatTurn


class AnalyticsQuery(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=2000,
        examples=["2026'da en çok ciro yapan 5 müşteri kim ve harcama onay prosedürüne göre kim onaylamalı?"],
    )
    mode: Literal["auto", "documents", "data", "both"] = Field(
        default="auto", description="auto: soruya göre otomatik yönlendirme"
    )
    departments: list[str] | None = Field(default=None, description="Doküman araması için departmanlar")
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    type: Literal["document", "database"]
    label: str
    detail: dict[str, Any]


class AttemptOut(BaseModel):
    sql: str
    error: str | None


class DataResultOut(BaseModel):
    success: bool
    sql: str
    explanation: str
    summary: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    tables: list[str]
    numeric_stats: dict[str, dict[str, float]]
    insights: list[str]
    chart: dict[str, Any] | None = Field(description="Vega-Lite v5 grafik spesifikasyonu (vega-embed ile çizilir)")
    attempts: list[AttemptOut]
    timings_ms: dict[str, float]
    warnings: list[str]


class AnalyticsResponse(BaseModel):
    question: str
    route: Literal["documents", "data", "both"]
    route_reason: str
    route_method: Literal["llm", "heuristic", "client"]
    answer: str
    is_answered: bool
    confidence: int = Field(ge=0, le=100)
    sources: list[SourceOut]
    documents: ChatResponse | None
    data: DataResultOut | None
    timings_ms: dict[str, float]
    warnings: list[str]


class DemoDataResult(BaseModel):
    created: bool
    customers: int
    products: int
    orders: int
    order_items: int


class SchemaInfo(BaseModel):
    tables: list[str]
    description: str
