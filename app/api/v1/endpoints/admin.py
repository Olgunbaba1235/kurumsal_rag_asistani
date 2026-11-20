from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import Container, require_admin
from app.schemas.common import ErrorResponse
from app.schemas.tenant import TenantCreate, TenantCreated, TenantOut
from app.services.tenant_service import TenantConflictError

router = APIRouter(dependencies=[Depends(require_admin)])


@router.post(
    "/tenants",
    response_model=TenantCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni tenant (şirket) oluştur",
    responses={401: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def create_tenant(body: TenantCreate, container: Container) -> TenantCreated:
    """Yeni bir şirket hesabı oluşturur ve **yalnızca bir kez** gösterilen API anahtarını döndürür."""
    try:
        tenant, api_key = await container.tenants.create(body)
    except TenantConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    public = TenantOut.model_validate(tenant).model_dump()
    return TenantCreated(**public, api_key=api_key)
