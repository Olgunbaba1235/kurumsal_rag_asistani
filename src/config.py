"""Merkezi uygulama yapılandırması.

Tüm ayarlar ortam değişkenlerinden / `.env` dosyasından okunur ve Pydantic
ile doğrulanır. Kod içinde `from src.config import get_settings` ile erişilir.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DEPARTMENTS: tuple[str, ...] = ("Genel", "İnsan Kaynakları", "Finans", "Teknik")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------ Google GenAI
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-3.5-flash", alias="LLM_MODEL")
    # Sorgu yeniden yazma / reranking gibi hızlı yardımcı çağrılar için
    fast_llm_model: str = Field(default="gemini-3.5-flash-lite", alias="FAST_LLM_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, ge=128, le=3072, alias="EMBEDDING_DIMENSION")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    # Tek bir Gemini isteğinin en uzun bekleme süresi (sn). Aşılırsa istek kesilir; ana model zaman aşımına
    # uğrarsa aynı modelde beklemeden fast_llm_model'e geçilir.
    llm_timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0, alias="LLM_TIMEOUT_SECONDS")
    # Ana model 503 / "overloaded" / zaman aşımı verirse (tekrar denemelerden sonra) fast_llm_model ile dene
    llm_fallback_to_fast: bool = Field(default=True, alias="LLM_FALLBACK_TO_FAST")

    # ------------------------------------------------------------ ChromaDB
    chroma_persist_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="kurumsal_belgeler", alias="COLLECTION_NAME")

    # ------------------------------------------------------------ Dosyalar
    upload_dir: Path = Field(default=Path("data/uploads"), alias="UPLOAD_DIR")
    sample_docs_dir: Path = Field(default=Path("data/sample_docs"), alias="SAMPLE_DOCS_DIR")
    max_file_size_mb: int = Field(default=25, ge=1, alias="MAX_FILE_SIZE_MB")

    # ------------------------------------------------------------ Chunking
    chunk_size: int = Field(default=1000, ge=200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, ge=0, alias="CHUNK_OVERLAP")

    # ------------------------------------------------------------ Departmanlar
    departments: list[str] = Field(default_factory=lambda: list(DEFAULT_DEPARTMENTS), alias="DEPARTMENTS")
    default_department: str = Field(default="Genel", alias="DEFAULT_DEPARTMENT")

    # ------------------------------------------------------------ Retrieval
    top_k: int = Field(default=5, ge=1, le=20, alias="TOP_K")
    # Her arama kolundan (vektör / BM25) füzyona alınacak aday sayısı
    candidate_k: int = Field(default=20, ge=5, le=100, alias="CANDIDATE_K")
    rrf_k: int = Field(default=60, ge=1, alias="RRF_K")
    vector_weight: float = Field(default=1.0, ge=0.0, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=1.0, ge=0.0, alias="BM25_WEIGHT")
    use_llm_reranker: bool = Field(default=True, alias="USE_LLM_RERANKER")
    # Rerank sonrası en iyi parçanın alaka skoru bunun altındaysa LLM çağrılmaz (0..1)
    min_relevance: float = Field(default=0.30, ge=0.0, le=1.0, alias="MIN_RELEVANCE")

    # ------------------------------------------------------------ Üretim
    query_rewriting: bool = Field(default=True, alias="QUERY_REWRITING")
    # "Bunu mu sormak istediniz?": aramadan önce fast_llm_model soruyu yazım/belirsizlik için denetler
    query_suggestions: bool = Field(default=True, alias="QUERY_SUGGESTIONS")
    history_turns: int = Field(default=4, ge=0, le=20, alias="HISTORY_TURNS")

    # ------------------------------------------------------------ Doğrulama
    @field_validator("chroma_persist_dir", "upload_dir", "sample_docs_dir", mode="after")
    @classmethod
    def _resolve_relative(cls, v: Path) -> Path:
        """Göreli yollar çalışma dizinine değil proje köküne göre çözülür."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @field_validator("departments", mode="after")
    @classmethod
    def _clean_departments(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(d.strip() for d in v if d and d.strip()))
        if not cleaned:
            raise ValueError("En az bir departman tanımlanmalıdır.")
        return cleaned

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük olmalıdır.")
        if self.default_department not in self.departments:
            raise ValueError(
                f"DEFAULT_DEPARTMENT '{self.default_department}' departman listesinde yok: {self.departments}"
            )
        return self

    # ------------------------------------------------------------ Yardımcılar
    @property
    def api_key(self) -> str | None:
        return self.google_api_key.get_secret_value() if self.google_api_key else None

    def validate_department(self, department: str) -> str:
        if department not in self.departments:
            raise ValueError(f"Geçersiz departman: '{department}'. Geçerli: {', '.join(self.departments)}")
        return department

    def ensure_dirs(self) -> None:
        for d in (self.chroma_persist_dir, self.upload_dir, self.sample_docs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
