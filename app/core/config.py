"""API yapılandırması.

`src.config.Settings` (chunking, retrieval, model ayarları) genişletilir; böylece
çekirdek RAG fonksiyonları API ayarlarını doğrudan kabul eder.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from src.config import Settings

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class ApiSettings(Settings):
    # ------------------------------------------------------------ Uygulama
    app_name: str = Field(default="Kurumsal RAG API", alias="APP_NAME")
    environment: Literal["development", "test", "production"] = Field(default="development", alias="ENVIRONMENT")
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=list, alias="CORS_ORIGINS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ------------------------------------------------------------ Veritabanı
    # Uygulama, RLS'e tabi olan ve superuser OLMAYAN rolle bağlanır (ör. rag_app).
    database_url: str = Field(
        default="postgresql+asyncpg://rag_app:rag_app_password@localhost:5432/rag", alias="DATABASE_URL"
    )
    # Migration'lar tablo sahibi rolle çalışır; boşsa DATABASE_URL kullanılır.
    database_migration_url: str | None = Field(default=None, alias="DATABASE_MIGRATION_URL")
    db_pool_size: int = Field(default=10, ge=1, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, ge=0, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: float = Field(default=30.0, gt=0, alias="DB_POOL_TIMEOUT")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    # Bağlantı ayrıcalıklı bir rolle açılıyorsa (ör. PgBouncer arkasında ortak kullanıcı),
    # her transaction'da `SET LOCAL ROLE` ile RLS'e tabi role geçilir.
    db_session_role: str | None = Field(default=None, alias="DB_SESSION_ROLE")
    db_app_role: str = Field(default="rag_app", alias="DB_APP_ROLE")
    db_dispose_timeout: float = Field(default=5.0, gt=0, alias="DB_DISPOSE_TIMEOUT")
    # pgvector >= 0.8: filtreli HNSW sorgularında eksik sonuç dönmesini önler
    pgvector_iterative_scan: bool = Field(default=True, alias="PGVECTOR_ITERATIVE_SCAN")

    # ------------------------------------------------------------ Redis / rate limit
    redis_url: str | None = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    rate_limit_per_minute: int = Field(default=60, ge=0, alias="RATE_LIMIT_PER_MINUTE")  # 0 = kapalı

    # ------------------------------------------------------------ Güvenlik
    admin_api_key: SecretStr | None = Field(default=None, alias="ADMIN_API_KEY")

    # ------------------------------------------------------------ Sağlayıcılar
    embedding_provider: Literal["gemini", "hashing"] = Field(default="gemini", alias="EMBEDDING_PROVIDER")
    llm_provider: Literal["gemini", "none"] = Field(default="gemini", alias="LLM_PROVIDER")

    # ------------------------------------------------------------ Arama
    bm25_cache_size: int = Field(default=32, ge=1, alias="BM25_CACHE_SIZE")
    bm25_max_chunks: int = Field(default=50_000, ge=100, alias="BM25_MAX_CHUNKS")
    max_question_chars: int = Field(default=2000, ge=10, alias="MAX_QUESTION_CHARS")

    # ------------------------------------------------------------ Analitik (Text-to-SQL)
    # Üretilen SQL'ler yalnızca SELECT yetkili bu rolle, READ ONLY transaction'da çalışır
    analytics_readonly_role: str = Field(default="rag_readonly", alias="ANALYTICS_READONLY_ROLE")
    analytics_max_rows: int = Field(default=500, ge=1, le=10_000, alias="ANALYTICS_MAX_ROWS")
    analytics_statement_timeout_ms: int = Field(default=5000, ge=100, alias="ANALYTICS_STATEMENT_TIMEOUT_MS")
    analytics_max_attempts: int = Field(default=3, ge=1, le=5, alias="ANALYTICS_MAX_ATTEMPTS")
    # Demo satış verisi üretme uç noktası (üretimde kapalı tutun)
    enable_demo_data: bool = Field(default=True, alias="ENABLE_DEMO_DATA")

    @field_validator("db_session_role", "db_app_role", "analytics_readonly_role", mode="after")
    @classmethod
    def _valid_identifier(cls, v: str | None) -> str | None:
        # Rol adı SQL'e tanımlayıcı olarak gömülür → sıkı doğrulama
        if v is not None and not _IDENTIFIER_RE.match(v):
            raise ValueError(f"Geçersiz PostgreSQL rol adı: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_providers(self) -> ApiSettings:
        needs_key = self.embedding_provider == "gemini" or self.llm_provider == "gemini"
        if needs_key and not self.api_key and self.environment != "test":
            raise ValueError(
                "GOOGLE_API_KEY tanımlı değil. Gemini kullanmak için anahtarı ekleyin ya da geliştirme için "
                "EMBEDDING_PROVIDER=hashing ve LLM_PROVIDER=none ayarlayın."
            )
        return self

    @property
    def migration_url(self) -> str:
        return self.database_migration_url or self.database_url


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()
