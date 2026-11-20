from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from app.api.deps import Container, CurrentTenant
from app.schemas.analytics import (
    AnalyticsQuery,
    AnalyticsResponse,
    DataResultOut,
    DemoDataResult,
    SchemaInfo,
    SourceOut,
)
from app.schemas.chat import ChatResponse
from app.schemas.common import ErrorResponse
from app.services.demo_data import seed_demo_sales
from app.services.tenant_service import DepartmentNotAllowedError
from app.services.text_to_sql_service import SCHEMA_DESCRIPTION
from app.models.sql_models import ANALYTICS_TABLES

router = APIRouter()


@router.post(
    "/query",
    response_model=AnalyticsResponse,
    summary="Birleşik analitik: dokümanlar (RAG) + satış verisi (Text-to-SQL)",
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
async def analytics_query(body: AnalyticsQuery, tenant: CurrentTenant, container: Container) -> AnalyticsResponse:
    """Soruyu otomatik olarak **doküman araması**, **satış veritabanı** ya da **ikisine birden** yönlendirir
    ve sonuçları tek bir B2B yanıtında birleştirir.

    * **Doküman bölümü:** `/chat/query` ile aynı katı bağlamlı, `[Dosya - Sayfa X]` atıflı yanıt.
    * **Veri bölümü:** üretilen ve güvenlik doğrulamasından geçen SQL, sonuç satırları, istatistikler,
      içgörüler ve Vega-Lite grafik spesifikasyonu.
    * Tüm sorgular yalnızca kendi şirketinizin verisinde çalışır (RLS + salt-okunur rol).
    """
    try:
        departments = tenant.resolve_departments(body.departments)
    except DepartmentNotAllowedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    r = await container.analytics.query(
        tenant,
        body.question,
        departments,
        [(t.role, t.content) for t in body.history],
        mode=body.mode,
        top_k=body.top_k,
    )
    return AnalyticsResponse(
        question=r.question,
        route=r.route,
        route_reason=r.route_reason,
        route_method=r.route_method,
        answer=r.answer,
        is_answered=r.is_answered,
        confidence=r.confidence,
        sources=[SourceOut(type=s.type, label=s.label, detail=s.detail) for s in r.sources],
        documents=ChatResponse.from_result(r.documents, departments) if r.documents else None,
        data=DataResultOut.model_validate(asdict(r.data)) if r.data else None,
        timings_ms=r.timings_ms,
        warnings=r.warnings,
    )


@router.get("/schema", response_model=SchemaInfo, summary="Sorgulanabilir veri şeması")
async def analytics_schema(tenant: CurrentTenant) -> SchemaInfo:
    """Text-to-SQL ajanının erişebildiği tablolar ve sütunlar."""
    return SchemaInfo(tables=list(ANALYTICS_TABLES), description=SCHEMA_DESCRIPTION)


@router.post(
    "/demo-data",
    response_model=DemoDataResult,
    summary="Demo satış verisi oluştur (yalnızca kendi şirketiniz için)",
    responses={403: {"model": ErrorResponse}},
)
async def create_demo_data(tenant: CurrentTenant, container: Container) -> DemoDataResult:
    """Şirketiniz için gerçekçi B2B satış verisi (müşteri, ürün, sipariş, kalem) üretir.
    Veri zaten varsa dokunmaz. `ENABLE_DEMO_DATA=false` ile kapatılır (üretimde önerilir)."""
    if not container.settings.enable_demo_data:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Demo veri üretimi bu ortamda kapalı.")
    return DemoDataResult.model_validate(asdict(await seed_demo_sales(container.db, tenant)))
