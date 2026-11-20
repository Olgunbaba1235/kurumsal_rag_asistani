from __future__ import annotations

import time

from fastapi import APIRouter, Response, status

from app import __version__
from app.api.deps import Container
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter()


async def _probe(fn) -> ComponentHealth:  # noqa: ANN001
    t0 = time.perf_counter()
    try:
        await fn()
        return ComponentHealth(status="ok", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status="error", detail=type(exc).__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Servis sağlık durumu",
    responses={503: {"model": HealthResponse, "description": "Bir veya daha fazla bileşen erişilemez"}},
)
async def health(container: Container, response: Response) -> HealthResponse:
    """Veritabanı, Redis ve model sağlayıcılarının durumunu döndürür. Kimlik doğrulama gerektirmez."""
    s = container.settings
    components = {"database": await _probe(container.db.ping)}
    if container.redis is None:
        components["redis"] = ComponentHealth(status="disabled", detail="REDIS_URL tanımlı değil")
    else:
        components["redis"] = await _probe(container.redis.ping)
    components["embedding"] = ComponentHealth(status="ok", detail=s.embedding_provider)
    components["llm"] = ComponentHealth(
        status="ok" if s.llm_provider != "none" else "disabled", detail=s.llm_provider
    )

    # Veritabanı zorunlu; Redis hatası servisi "degraded" yapar ama durdurmaz
    db_ok = components["database"].status == "ok"
    degraded = not db_ok or components["redis"].status == "error"
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        environment=s.environment,
        components=components,
    )
