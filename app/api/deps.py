"""Ortak FastAPI bağımlılıkları: servis kabı, kimlik (oturum veya API anahtarı), rate limit, yetki.

İki kimlik yolu birlikte çalışır:

* ``Authorization: Bearer rs_…`` → kullanıcı oturumu; rol ve departmanlar üyelikten gelir.
* ``X-API-Key`` → şirket API anahtarı; admin yetkisi, şirketin tüm departmanları (eski davranış).
* İkisi birden → 400; hiçbiri → 401.

Uç noktalar ``CurrentTenant`` ile izinli departmanlara daraltılmış ``TenantContext`` alır;
servisler veritabanı oturumuna da bu departmanları verir (departman RLS'i).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.permissions import Action, Actor, can
from app.core.security import constant_time_equals
from app.services.auth_service import Principal
from app.services.container import ServiceContainer
from app.services.tenant_service import TenantContext

tenant_api_key = APIKeyHeader(
    name="X-API-Key",
    scheme_name="TenantApiKey",
    description="Şirketinize (tenant) özel API anahtarı. Tüm veri erişimi bu anahtarın tenant'ı ile sınırlıdır.",
    auto_error=False,
)
user_bearer = HTTPBearer(
    scheme_name="UserSession",
    description="Kullanıcı oturum belirteci (`/api/v1/auth/login` yanıtındaki `token`).",
    auto_error=False,
)
admin_api_key = APIKeyHeader(
    name="X-Admin-Key",
    scheme_name="AdminApiKey",
    description="Platform yöneticisi anahtarı (ADMIN_API_KEY). Yalnızca tenant yönetimi içindir.",
    auto_error=False,
)

_AUTH_REQUIRED = "Kimlik doğrulaması gerekli: X-API-Key veya Authorization: Bearer <oturum belirteci>."


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


Container = Annotated[ServiceContainer, Depends(get_container)]


async def get_principal(
    request: Request,
    container: Container,
    api_key: Annotated[str | None, Depends(tenant_api_key)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(user_bearer)],
) -> Principal:
    has_bearer = bearer is not None or bool(request.headers.get("authorization"))
    if api_key and has_bearer:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Aynı istekte hem X-API-Key hem Authorization kullanılamaz."
        )
    if has_bearer:
        token = bearer.credentials if bearer is not None else None
        principal = await container.auth.authenticate_session(token)
        if principal is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Oturum geçersiz veya süresi dolmuş.", headers={"WWW-Authenticate": "Bearer"}
            )
        return principal
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _AUTH_REQUIRED, headers={"WWW-Authenticate": "ApiKey"})
    tenant = await container.tenants.authenticate(api_key)
    if tenant is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Geçersiz veya pasif API anahtarı.", headers={"WWW-Authenticate": "ApiKey"}
        )
    return Principal(tenant=tenant, actor=Actor.api_key())


async def rate_limited_principal(
    container: Container,
    response: Response,
    principal: Annotated[Principal, Depends(get_principal)],
) -> Principal:
    keys = [str(principal.tenant.id)]
    if principal.user_id is not None:  # bir çalışan tüm şirketin kotasını tüketemesin
        keys.append(f"user:{principal.user_id}")
    for key in keys:
        result = await container.rate_limiter.hit(key)
        if container.rate_limiter.enabled and key == keys[0]:
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        if not result.allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Dakikalık istek sınırı aşıldı.",
                headers={"Retry-After": str(result.retry_after)},
            )
    return principal


CurrentPrincipal = Annotated[Principal, Depends(rate_limited_principal)]


async def current_tenant(principal: CurrentPrincipal) -> TenantContext:
    return principal.tenant


CurrentTenant = Annotated[TenantContext, Depends(current_tenant)]


def require(action: Action):  # noqa: ANN201
    """Rol kapısı: ``dependencies=[Depends(require(Action.DOCS_UPLOAD))]``."""

    async def _check(principal: CurrentPrincipal) -> Principal:
        if not can(principal.actor, action):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Bu işlem için yetkiniz yok.")
        return principal

    return _check


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def auth_rate_limit(request: Request, container: Container) -> None:
    """Kayıt / giriş / davet kabulü: IP başına dakikalık sınır (kaba kuvvet koruması)."""
    result = await container.auth_rate_limiter.hit(f"auth:{client_ip(request)}")
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Çok fazla deneme. Lütfen biraz sonra tekrar deneyin.",
            headers={"Retry-After": str(result.retry_after)},
        )


async def require_admin(
    container: Container,
    key: Annotated[str | None, Depends(admin_api_key)],
) -> None:
    configured = container.settings.admin_api_key
    if configured is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ADMIN_API_KEY yapılandırılmamış.")
    if not constant_time_equals(key, configured.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Geçersiz yönetici anahtarı.")
