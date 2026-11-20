"""Uygulama genelindeki servislerin tek noktadan kurulumu (bağımlılık kabı)."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ApiSettings
from app.core.db import Database
from app.core.redis import RateLimiter, RedisLike, create_redis
from app.services.analytics_service import AnalyticsService
from app.services.document_service import DocumentService
from app.services.providers import Providers, build_providers
from app.services.rag_service import RAGService
from app.services.retrieval_service import PgHybridRetriever
from app.services.tenant_service import TenantService
from app.services.text_to_sql_service import TextToSQLService


@dataclass(slots=True)
class ServiceContainer:
    settings: ApiSettings
    db: Database
    redis: RedisLike | None
    rate_limiter: RateLimiter
    providers: Providers
    tenants: TenantService
    documents: DocumentService
    retriever: PgHybridRetriever
    rag: RAGService
    text_to_sql: TextToSQLService
    analytics: AnalyticsService

    @classmethod
    def build(
        cls,
        settings: ApiSettings,
        *,
        db: Database | None = None,
        providers: Providers | None = None,
        redis: RedisLike | None = None,
        use_redis: bool = True,
    ) -> ServiceContainer:
        db = db or Database(settings)
        providers = providers or build_providers(settings)
        if redis is None and use_redis:
            redis = create_redis(settings.redis_url)
        retriever = PgHybridRetriever(settings, providers.embedder, providers.reranker)
        rag = RAGService(db, settings, retriever, providers.llm)
        text_to_sql = TextToSQLService(db, settings, providers.llm)
        return cls(
            settings=settings,
            db=db,
            redis=redis,
            rate_limiter=RateLimiter(redis, settings.rate_limit_per_minute),
            providers=providers,
            tenants=TenantService(db),
            documents=DocumentService(db, settings, providers.embedder),
            retriever=retriever,
            rag=rag,
            text_to_sql=text_to_sql,
            analytics=AnalyticsService(rag, text_to_sql, providers.llm),
        )

    async def close(self) -> None:
        await self.db.dispose()
        if self.redis is not None and hasattr(self.redis, "aclose"):
            try:
                await self.redis.aclose()
            except Exception:  # noqa: BLE001
                pass
