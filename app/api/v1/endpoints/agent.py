from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import Container, CurrentTenant
from app.schemas.agent import AgentQuery, AgentResponse, B2BReport, RagSourceOut, SqlSummaryOut
from app.schemas.common import ErrorResponse
from app.services.tenant_service import DepartmentNotAllowedError

router = APIRouter()


@router.post(
    "/query",
    response_model=AgentResponse,
    summary="Multi-Agent orkestratör: niyet → paralel RAG + SQL → B2B raporu",
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
async def agent_query(body: AgentQuery, tenant: CurrentTenant, container: Container) -> AgentResponse:
    """Sorunun niyetini (`DOCUMENT_ONLY`, `SQL_ONLY`, `HYBRID`) belirler, uzman ajanları paralel çalıştırır
    ve sonuçları tek bir kurumsal analiz raporunda sentezler. Tüm veri erişimi tenant ile sınırlıdır."""
    try:
        departments = tenant.resolve_departments(body.departments)
    except DepartmentNotAllowedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    r = await container.orchestrator.run(
        tenant,
        body.question,
        departments,
        [(t.role, t.content) for t in body.history],
        intent=body.intent,
        top_k=body.top_k,
    )
    sources = (
        [
            RagSourceOut(
                label=c.label, source=c.source, page=c.page, department=c.department, relevance=c.relevance
            )
            for c in r.documents.citations
        ]
        if r.documents
        else []
    )
    sql = None
    data = None
    if r.data is not None:
        d = r.data
        sql = SqlSummaryOut(
            success=d.success,
            sql=d.sql,
            tables=d.tables,
            columns=d.columns,
            row_count=d.row_count,
            truncated=d.truncated,
            attempts=len(d.attempts),
            summary=d.summary,
            insights=d.insights,
            error=None if d.success else next((a.error for a in reversed(d.attempts) if a.error), None),
        )
        if d.success:
            data = {"rows": d.rows, "numeric_stats": d.numeric_stats, "chart": d.chart}
    return AgentResponse(
        question=r.question,
        intent=r.intent,
        intent_reason=r.intent_reason,
        intent_method=r.intent_method,
        is_answered=r.is_answered,
        confidence=r.confidence,
        report=B2BReport.model_validate(r.report),
        rag_sources=sources,
        sql=sql,
        data=data,
        timings_ms=r.timings_ms,
        warnings=r.warnings,
    )
