"""FastAPI uygulama fabrikası.

Çalıştırma:  uvicorn app.main:create_app --factory --reload
Swagger UI:  http://localhost:8000/docs

Fabrika modu, ayarların import anında değil uygulama başlarken okunmasını sağlar
(testler farklı ayarlarla `create_app(settings)` çağırabilir).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.v1.router import api_router
from app.core.config import ApiSettings, get_api_settings
from app.services.container import ServiceContainer
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)

ContainerFactory = Callable[[ApiSettings], ServiceContainer]

DESCRIPTION = """
Çok kiracılı (multi-tenant) kurumsal RAG API'si.

**Kimlik doğrulama:** Sağ üstteki **Authorize** düğmesine şirketinizin `X-API-Key` değerini girin.
Tüm belge ve arama işlemleri yalnızca bu anahtarın ait olduğu şirketin verisiyle sınırlıdır
(uygulama filtresi + PostgreSQL Row-Level Security).

**Akış:** `POST /documents/upload` → `GET /documents/{id}` (durum `ready`) → `POST /chat/query`
"""


async def _verify_embedding_dimension(container: ServiceContainer) -> None:
    """Şemadaki vektör boyutu ile EMBEDDING_DIMENSION uyuşmazsa açılışta hızlı başarısız ol."""
    column = await container.db.embedding_column_type()
    expected = f"vector({container.settings.embedding_dimension})"
    if column is None:
        logger.warning("document_chunks tablosu yok; `alembic upgrade head` çalıştırıldı mı?")
    elif column != expected:
        raise RuntimeError(
            f"Embedding boyutu uyuşmazlığı: veritabanı {column}, yapılandırma {expected}. "
            "EMBEDDING_DIMENSION değerini düzeltin veya yeni bir migration oluşturun."
        )


def create_app(settings: ApiSettings | None = None, container_factory: ContainerFactory | None = None) -> FastAPI:
    settings = settings or get_api_settings()
    setup_logging(settings.log_level)
    factory = container_factory or ServiceContainer.build

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = factory(settings)
        await _verify_embedding_dimension(container)
        app.state.container = container
        logger.info(
            "%s v%s başladı (env=%s, embedding=%s, llm=%s)",
            settings.app_name, __version__, settings.environment, settings.embedding_provider, settings.llm_provider,
        )
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # Üretimde etkileşimli dokümantasyon ve şema herkese açık sunulmaz
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
        openapi_url=None if settings.environment == "production" else "/openapi.json",
        openapi_tags=[
            {"name": "Sağlık", "description": "Servis ve bağımlılık durumu"},
            {"name": "Belgeler", "description": "Belge yükleme, listeleme ve silme (tenant + departman kapsamlı)"},
            {"name": "Sohbet", "description": "Hibrit arama ve kaynak referanslı yanıt üretimi"},
            {"name": "Analitik", "description": "Text-to-SQL veri analizi ve doküman + veri birleşik yanıtlar"},
            {"name": "Yönetim", "description": "Platform yöneticisi: tenant oluşturma"},
        ],
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("İşlenmeyen hata [request_id=%s]", request_id)
            response = JSONResponse(
                status_code=500, content={"detail": "Beklenmeyen bir hata oluştu.", "request_id": request_id}
            )
        elapsed = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        logger.info("%s %s → %s (%.0f ms) [%s]", request.method, request.url.path, response.status_code, elapsed, request_id)
        return response

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        info = {"service": settings.app_name, "version": __version__}
        if settings.environment != "production":
            info["docs"] = "/docs"
        return info

    return app
