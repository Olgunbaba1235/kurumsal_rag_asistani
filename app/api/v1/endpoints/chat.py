from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import Container, CurrentTenant
from app.schemas.chat import ChatQuery, ChatResponse
from app.schemas.common import ErrorResponse
from app.services.tenant_service import DepartmentNotAllowedError

router = APIRouter()


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="Belgelere soru sor (hibrit arama + kaynak referanslı yanıt)",
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def query(body: ChatQuery, tenant: CurrentTenant, container: Container) -> ChatResponse:
    """Soruyu yalnızca **kendi şirketinizin** ve seçilen departmanların belgelerinde arar.

    * Sohbet geçmişi verilirse takip soruları bağımsız sorguya çevrilir.
    * Yanıt yalnızca bulunan bağlama dayanır; bilgi yoksa
      `"Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır."` döner.
    * Her iddia `[Dosya Adı - Sayfa X]` ile atıflanır; kanıt metinleri, güven skoru (%)
      ve adım adım süreler (ms) yanıtla birlikte döner.
    """
    if len(body.question) > container.settings.max_question_chars:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Soru çok uzun.")
    try:
        departments = tenant.resolve_departments(body.departments)
    except DepartmentNotAllowedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    result = await container.rag.ask(
        tenant,
        body.question,
        departments,
        [(t.role, t.content) for t in body.history],
        top_k=body.top_k,
    )
    return ChatResponse.from_result(result, departments)
