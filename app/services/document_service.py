"""Belge yükleme, işleme ve yönetimi (tenant kapsamlı).

Yükleme akışı
-------------
1. `register_upload` — uzantı/boyut/departman doğrulanır, içerik hash'i hesaplanır.
   * Aynı tenant + departman + dosya adı + içerik zaten HAZIR ise → ``duplicate``.
   * Değilse ``processing`` durumunda bir belge kaydı oluşturulur.
2. `process` — (arka planda ya da `wait=true` ile senkron) mevcut akıllı belge
   işleyici ile ayrıştırma + tablo korumalı chunking, embedding, ardından **tek
   bir transaction** içinde:
   * aynı adlı eski HAZIR sürüm silinir (parçaları CASCADE ile gider),
   * yeni parçalar eklenir, belge ``ready`` yapılır.
   Böylece arama hiçbir anda yarım indekslenmiş belge görmez.
   Hata olursa belge ``failed`` olarak işaretlenir ve hata mesajı saklanır.

Her sorgu hem açık ``tenant_id`` filtresi (uygulama katmanı) hem de PostgreSQL
RLS (veritabanı katmanı) ile kapsamlanır.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select, update

from app.core.config import ApiSettings
from app.core.db import Database
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.tenant_service import TenantContext
from src.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentProcessingError,
    UnsupportedFileTypeError,
    compute_doc_id,
    process_document,
)
from src.embeddings import Embedder

logger = logging.getLogger(__name__)

_INSERT_BATCH = 200


class DocumentNotFoundError(Exception):
    pass


class DocumentService:
    def __init__(self, db: Database, settings: ApiSettings, embedder: Embedder):
        self.db = db
        self.settings = settings
        self.embedder = embedder

    # ------------------------------------------------------------ doğrulama
    def validate_upload(self, filename: str, size: int) -> str:
        name = Path(filename or "").name.strip()
        if not name:
            raise DocumentProcessingError("Dosya adı boş olamaz.")
        if len(name) > 255:
            raise DocumentProcessingError("Dosya adı en fazla 255 karakter olabilir.")
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"'{ext or name}' desteklenmiyor. Desteklenen türler: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        if size == 0:
            raise DocumentProcessingError(f"'{name}' boş bir dosya.")
        if size > self.settings.max_file_size_mb * 1024 * 1024:
            raise DocumentProcessingError(f"'{name}' {self.settings.max_file_size_mb} MB sınırını aşıyor.")
        return name

    # ------------------------------------------------------------ yükleme
    async def register_upload(
        self, tenant: TenantContext, department: str, filename: str, data: bytes
    ) -> tuple[Document, bool]:
        """Belge kaydını oluşturur. Dönen bool: içerik zaten indeksli (duplicate) mi?"""
        [department] = tenant.resolve_departments([department])
        name = self.validate_upload(filename, len(data))
        content_hash = compute_doc_id(data)

        async with self.db.tenant_session(tenant.id) as session:
            existing = await session.scalar(
                select(Document).where(
                    Document.tenant_id == tenant.id,
                    Document.department == department,
                    Document.filename == name,
                    Document.status == DocumentStatus.READY,
                )
            )
            if existing is not None and existing.content_hash == content_hash:
                return existing, True

            doc = Document(
                tenant_id=tenant.id,
                department=department,
                filename=name,
                file_type=Path(name).suffix.lower().lstrip("."),
                size_bytes=len(data),
                content_hash=content_hash,
                status=DocumentStatus.PROCESSING,
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            return doc, False

    async def process(self, tenant: TenantContext, document_id: uuid.UUID, data: bytes) -> Document:
        """Ayrıştır → chunk → embed → atomik olarak yayınla. Hata durumunda `failed`."""
        doc = await self.get(tenant, document_id)
        try:
            # Departman doğrulaması tenant'ın kendi departman listesine göre yapılır
            scoped_settings = self.settings.model_copy(
                update={"departments": list(tenant.departments), "default_department": tenant.departments[0]}
            )
            processed = await asyncio.to_thread(
                process_document, doc.filename, data, doc.department, scoped_settings
            )
            texts = [c.text for c in processed.chunks]
            vectors = await asyncio.to_thread(self.embedder.embed_documents, texts)
            if len(vectors) != len(texts):
                raise DocumentProcessingError("Embedding sayısı parça sayısıyla eşleşmiyor.")

            async with self.db.tenant_session(tenant.id) as session, session.begin():
                # 1) Aynı adlı eski hazır sürüm(ler)i kaldır
                await session.execute(
                    delete(Document).where(
                        Document.tenant_id == tenant.id,
                        Document.department == doc.department,
                        Document.filename == doc.filename,
                        Document.status == DocumentStatus.READY,
                        Document.id != doc.id,
                    )
                )
                # 2) Yeni parçaları ekle
                rows = [
                    {
                        "id": uuid.uuid4(),
                        "tenant_id": tenant.id,
                        "document_id": doc.id,
                        "department": doc.department,
                        "source": doc.filename,
                        "chunk_index": int(c.metadata["chunk_index"]),
                        "page": c.metadata.get("page"),
                        "section": c.metadata.get("section"),
                        "sheet_name": c.metadata.get("sheet_name"),
                        "content_type": str(c.metadata.get("content_type", "text")),
                        "content": c.text,
                        "embedding": vec,
                    }
                    for c, vec in zip(processed.chunks, vectors, strict=True)
                ]
                for start in range(0, len(rows), _INSERT_BATCH):
                    await session.execute(DocumentChunk.__table__.insert(), rows[start : start + _INSERT_BATCH])
                # 3) Belgeyi yayınla
                await session.execute(
                    update(Document)
                    .where(Document.id == doc.id, Document.tenant_id == tenant.id)
                    .values(
                        status=DocumentStatus.READY,
                        error=None,
                        page_count=processed.page_count,
                        chunk_count=len(rows),
                        updated_at=func.now(),
                    )
                )
            logger.info("[%s] %s/%s → %d parça indekslendi", tenant.slug, doc.department, doc.filename, len(rows))
        except Exception as exc:  # noqa: BLE001 — her hata belgeye yazılmalı
            message = str(exc) if isinstance(exc, DocumentProcessingError) else f"İşleme hatası: {exc}"
            logger.exception("[%s] %s işlenemedi", tenant.slug, doc.filename)
            async with self.db.tenant_session(tenant.id) as session, session.begin():
                await session.execute(
                    update(Document)
                    .where(Document.id == doc.id, Document.tenant_id == tenant.id)
                    .values(status=DocumentStatus.FAILED, error=message[:2000], updated_at=func.now())
                )
        return await self.get(tenant, document_id)

    # ------------------------------------------------------------ okuma / silme
    async def get(self, tenant: TenantContext, document_id: uuid.UUID) -> Document:
        async with self.db.tenant_session(tenant.id) as session:
            doc = await session.scalar(
                select(Document).where(Document.id == document_id, Document.tenant_id == tenant.id)
            )
        if doc is None:
            raise DocumentNotFoundError(str(document_id))
        return doc

    async def list(
        self,
        tenant: TenantContext,
        *,
        department: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        conditions = [Document.tenant_id == tenant.id]
        if department:
            conditions.append(Document.department == tenant.resolve_departments([department])[0])
        if status:
            conditions.append(Document.status == status)
        async with self.db.tenant_session(tenant.id) as session:
            total = await session.scalar(select(func.count()).select_from(Document).where(*conditions)) or 0
            docs = (
                await session.scalars(
                    select(Document)
                    .where(*conditions)
                    .order_by(Document.department, Document.filename, Document.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return list(docs), int(total)

    async def delete(self, tenant: TenantContext, document_id: uuid.UUID) -> None:
        async with self.db.tenant_session(tenant.id) as session, session.begin():
            result = await session.execute(
                delete(Document).where(Document.id == document_id, Document.tenant_id == tenant.id)
            )
        if result.rowcount == 0:
            raise DocumentNotFoundError(str(document_id))
