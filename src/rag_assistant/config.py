"""Uygulama yapılandırması.

Tüm ayarlar ortam değişkenlerinden / `.env` dosyasından okunur ve tip
güvenliği için Pydantic ile doğrulanır.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Google GenAI ---
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, ge=128, le=3072, alias="EMBEDDING_DIMENSION")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")

    # --- Depolama ---
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_DIR")
    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads", alias="UPLOAD_DIR")
    collection_name: str = Field(default="kurumsal_dokumanlar", alias="COLLECTION_NAME")

    # --- Chunking ---
    chunk_size: int = Field(default=1000, ge=100, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, ge=0, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    top_k: int = Field(default=5, ge=1, le=20, alias="TOP_K")
    min_relevance: float = Field(default=0.35, ge=0.0, le=1.0, alias="MIN_RELEVANCE")

    # --- Limitler ---
    max_file_size_mb: int = Field(default=25, ge=1, alias="MAX_FILE_SIZE_MB")

    @field_validator("chroma_dir", "upload_dir", mode="after")
    @classmethod
    def _resolve_relative(cls, v: Path) -> Path:
        # Göreli yollar çalışma dizinine değil proje köküne göre çözülür.
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük olmalıdır.")
        return self

    @property
    def api_key(self) -> str | None:
        return self.google_api_key.get_secret_value() if self.google_api_key else None

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
