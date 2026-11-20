"""Ortak FastAPI bağımlılıkları: servis kabı, tenant kimlik doğrulama, rate limit, admin."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader

from app.core.security import constant_time_equals
from app.services.container import ServiceContainer
from app.services.tenant_service import TenantContext

tenant_api_key = APIKeyHeader(
    name="X-API-Key",
    scheme_name="TenantApiKey",
    description="Şirketinize (tenant) özel API anahtarı. Tüm veri erişimi bu anahtarın tenant'ı ile sınırlıdır.",
    auto_error=False,
)
admin_api_key = APIKeyHeader(
    name="X-Admin-Key",
    scheme_name="AdminApiKey",
    description="Platform yöneticisi anahtarı (ADMIN_API_KEY). Yalnızca tenant yönetimi içindir.",
    auto_error=False,
)


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


Container = Annotated[ServiceContainer, Depends(get_container)]


async def get_tenant(
    container: Container,
    api_key: Annotated[str | None, Depends(tenant_api_key)],
) -> TenantContext:
    if not api_key:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "X-API-Key başlığı gerekli.", headers={"WWW-Authenticate": "ApiKey"}
        )
    tenant = await container.tenants.authenticate(api_key)
    if tenant is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Geçersiz veya pasif API anahtarı.", headers={"WWW-Authenticate": "ApiKey"}
        )
    return tenant


async def rate_limited_tenant(
    container: Container,
    response: Response,
    tenant: Annotated[TenantContext, Depends(get_tenant)],
) -> TenantContext:
    result = await container.rate_limiter.hit(str(tenant.id))
    if container.rate_limiter.enabled:
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Dakikalık istek sınırı aşıldı.",
            headers={"Retry-After": str(result.retry_after)},
        )
    return tenant


CurrentTenant = Annotated[TenantContext, Depends(rate_limited_tenant)]


async def require_admin(
    container: Container,
    key: Annotated[str | None, Depends(admin_api_key)],
) -> None:
    configured = container.settings.admin_api_key
    if configured is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ADMIN_API_KEY yapılandırılmamış.")
    if not constant_time_equals(key, configured.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Geçersiz yönetici anahtarı.")
