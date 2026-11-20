from __future__ import annotations

import enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin
from src.config import get_settings

# Vektör boyutu şemaya sabitlenir; değiştirmek yeni bir migration gerektirir.
# Uygulama açılışta DB sütun tipiyle karşılaştırır (bkz. app.main).
EMBEDDING_DIM = get_settings().embedding_dimension


class DocumentStatus(enum.StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Bileşik FK hedefi: parçanın tenant'ı belgeninkiyle aynı olmak zorunda
        UniqueConstraint("id", "tenant_id", name="uq_documents_id_tenant"),
        # Aynı tenant + departmanda aynı adla yalnızca bir HAZIR sürüm olabilir
        Index(
            "uq_documents_ready_filename",
            "tenant_id",
            "department",
            "filename",
            unique=True,
            postgresql_where=text("status = 'ready'"),
        ),
        Index("ix_documents_tenant_department", "tenant_id", "department"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    department: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=DocumentStatus.PROCESSING)
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class DocumentChunk(Base):
    """Silme, veritabanındaki ON DELETE CASCADE ile yapılır (ORM ilişkisi gerekmez)."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "tenant_id"],
            ["documents.id", "documents.tenant_id"],
            ondelete="CASCADE",
            name="fk_document_chunks_document_tenant",
        ),
        Index("ix_document_chunks_tenant_department", "tenant_id", "department"),
        Index("ix_document_chunks_document", "document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    department: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)  # dosya adı (atıf için)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(200))
    sheet_name: Mapped[str | None] = mapped_column(String(100))
    content_type: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

