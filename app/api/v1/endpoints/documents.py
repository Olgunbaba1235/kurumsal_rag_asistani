from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Response, UploadFile, status

from app.api.deps import Container, CurrentTenant
from app.schemas.common import ErrorResponse
from app.schemas.document import DocumentList, DocumentOut, UploadResponse
from app.services.document_service import DocumentNotFoundError
from app.services.tenant_service import DepartmentNotAllowedError
from src.document_loader import DocumentProcessingError, UnsupportedFileTypeError

router = APIRouter()

_ERRORS = {
    401: {"model": ErrorResponse, "description": "API anahtarı eksik/geçersiz"},
    429: {"model": ErrorResponse, "description": "İstek sınırı aşıldı"},
}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Belge yükle (PDF, DOCX, XLSX, TXT, MD)",
    responses={
        200: {"model": UploadResponse, "description": "Aynı içerik zaten indeksli (duplicate)"},
        201: {"model": UploadResponse, "description": "wait=true: belge işlendi"},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        **_ERRORS,
    },
)
async def upload_document(
    tenant: CurrentTenant,
    container: Container,
    background: BackgroundTasks,
    response: Response,
    file: Annotated[UploadFile, File(description="Yüklenecek belge")],
    department: Annotated[str, Form(description="Belgenin ait olduğu departman", examples=["İnsan Kaynakları"])],
    wait: Annotated[bool, Query(description="true: işleme bitene kadar bekle (küçük dosyalar/test için)")] = False,
) -> UploadResponse:
    """Belgeyi tenant'ın seçilen departmanına yükler.

    Varsayılan olarak işleme (ayrıştırma, chunking, embedding) arka planda yapılır ve
    **202** döner; durum `GET /documents/{id}` ile izlenir. Aynı ad ve içerik zaten
    indeksliyse **200 duplicate**; aynı ad farklı içerikle yüklenirse işleme bitince
    eski sürümün yerini atomik olarak alır.
    """
    limit = container.settings.max_file_size_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Dosya {container.settings.max_file_size_mb} MB sınırını aşıyor.")

    try:
        doc, duplicate = await container.documents.register_upload(tenant, department, file.filename or "", data)
    except DepartmentNotAllowedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    if duplicate:
        response.status_code = status.HTTP_200_OK
        return UploadResponse(
            document=DocumentOut.model_validate(doc),
            outcome="duplicate",
            message="Bu içerik bu departmanda zaten indekslenmiş.",
        )

    if wait:
        doc = await container.documents.process(tenant, doc.id, data)
        ok = doc.status == "ready"
        response.status_code = status.HTTP_201_CREATED if ok else status.HTTP_422_UNPROCESSABLE_CONTENT
        return UploadResponse(
            document=DocumentOut.model_validate(doc),
            outcome="indexed" if ok else "failed",
            message=f"{doc.chunk_count} parça indekslendi." if ok else (doc.error or "İşleme başarısız."),
        )

    background.add_task(container.documents.process, tenant, doc.id, data)
    return UploadResponse(
        document=DocumentOut.model_validate(doc),
        outcome="accepted",
        message="Belge kuyruğa alındı; durumunu GET /api/v1/documents/{id} ile izleyebilirsiniz.",
    )


@router.get("", response_model=DocumentList, summary="Belgeleri listele", responses=_ERRORS)
async def list_documents(
    tenant: CurrentTenant,
    container: Container,
    department: Annotated[str | None, Query()] = None,
    status_filter: Annotated[Literal["processing", "ready", "failed"] | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    try:
        docs, total = await container.documents.list(
            tenant, department=department, status=status_filter, limit=limit, offset=offset
        )
    except DepartmentNotAllowedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DocumentList(items=[DocumentOut.model_validate(d) for d in docs], total=total, limit=limit, offset=offset)


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Belge durumu",
    responses={404: {"model": ErrorResponse}, **_ERRORS},
)
async def get_document(document_id: uuid.UUID, tenant: CurrentTenant, container: Container) -> DocumentOut:
    try:
        return DocumentOut.model_validate(await container.documents.get(tenant, document_id))
    except DocumentNotFoundError as exc:
        # Başka tenant'ın belgesi de 404 döner: varlığı bile sızdırılmaz
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Belge bulunamadı.") from exc


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Belgeyi sil",
    responses={404: {"model": ErrorResponse}, **_ERRORS},
)
async def delete_document(document_id: uuid.UUID, tenant: CurrentTenant, container: Container) -> Response:
    try:
        await container.documents.delete(tenant, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Belge bulunamadı.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
