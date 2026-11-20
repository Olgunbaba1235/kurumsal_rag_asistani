"""ChromaDB kalıcı vektör deposu.

* Tek kalıcı koleksiyon; departman izolasyonu ``department`` metadata
  filtresiyle sağlanır (her sorgu yalnızca seçilen departman(lar)da arar).
* Kosinüs mesafesi kullanılır; skorlar ``1 - distance`` ile 0..1 benzerliğe çevrilir.
* Embedding'ler `Embedder` ile dışarıda üretilir → Chroma'nın varsayılan ONNX
  modeline bağımlılık yok, sağlayıcı değiştirmek depoyu etkilemez.
* Aynı içerik aynı departmana tekrar yüklenirse atlanır; aynı **dosya adı**
  farklı içerikle yüklenirse eski sürüm silinip yenisi yazılır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import Settings, get_settings
from src.document_loader import DocumentChunk, ProcessedDocument
from src.embeddings import Embedder
from src.search_types import SearchResult

__all__ = ["IN_MEMORY", "AddResult", "DocumentInfo", "SearchResult", "VectorStore", "build_where"]

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 256
IN_MEMORY = ":memory:"


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    source: str
    department: str
    file_type: str
    doc_id: str
    chunk_count: int
    page_count: int


@dataclass(frozen=True, slots=True)
class AddResult:
    source: str
    department: str
    chunks_added: int
    skipped: bool = False
    replaced: bool = False

    @property
    def message(self) -> str:
        if self.skipped:
            return "Bu belge bu departmanda zaten kayıtlı; atlandı."
        verb = "güncellendi" if self.replaced else "eklendi"
        return f"{self.chunks_added} parça {verb}."


def build_where(
    departments: list[str] | None = None, source: str | None = None, doc_id: str | None = None
) -> dict[str, Any] | None:
    """Chroma `where` filtresi oluşturur (birden çok koşul `$and` ile birleşir)."""
    clauses: list[dict[str, Any]] = []
    if departments:
        clauses.append(
            {"department": departments[0]} if len(departments) == 1 else {"department": {"$in": departments}}
        )
    if source:
        clauses.append({"source": source})
    if doc_id:
        clauses.append({"doc_id": doc_id})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        settings: Settings | None = None,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder
        # Her yazma/silme işleminde artar; BM25 gibi türetilmiş indeksler önbelleği bununla geçersiz kılar.
        self.revision = 0
        chroma_settings = ChromaSettings(anonymized_telemetry=False)

        target = persist_dir if persist_dir is not None else self.settings.chroma_persist_dir
        if str(target) == IN_MEMORY:
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            Path(target).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(target), settings=chroma_settings)

        # Embedding boyutu koleksiyon adına eklenir: boyut değişirse eski vektörlerle karışmaz.
        name = collection_name or f"{self.settings.collection_name}_{self.settings.embedding_dimension}"
        self._collection = self._client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    # ================================================================ yazma
    def add_document(self, document: ProcessedDocument) -> AddResult:
        dept = self.settings.validate_department(document.department)

        existing = self._collection.get(where=build_where([dept], source=document.filename), include=["metadatas"])
        replaced = False
        if existing["ids"]:
            old_doc_ids = {m.get("doc_id") for m in existing["metadatas"] or []}
            if old_doc_ids == {document.doc_id}:
                return AddResult(document.filename, dept, 0, skipped=True)
            self._collection.delete(ids=existing["ids"])
            replaced = True

        self._upsert(document.chunks)
        self.revision += 1
        logger.info("%s [%s]: %d parça yazıldı.", document.filename, dept, len(document.chunks))
        return AddResult(document.filename, dept, len(document.chunks), replaced=replaced)

    def _upsert(self, chunks: list[DocumentChunk]) -> None:
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
            )

    def delete_document(self, source: str, department: str | None = None) -> int:
        """Dosya adına göre (isteğe bağlı departmanla sınırlı) siler; silinen parça sayısını döndürür."""
        where = build_where([department] if department else None, source=source)
        ids = self._collection.get(where=where, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self.revision += 1
        logger.info("%s silindi (%d parça).", source, len(ids))
        return len(ids)

    def reset(self, department: str | None = None) -> int:
        where = build_where([department] if department else None)
        ids = self._collection.get(where=where, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self.revision += 1
        return len(ids)

    # ================================================================ okuma
    def count(self, departments: list[str] | None = None) -> int:
        if not departments:
            return self._collection.count()
        return len(self._collection.get(where=build_where(departments), include=[])["ids"])

    def list_documents(self, department: str | None = None) -> list[DocumentInfo]:
        where = build_where([department] if department else None)
        metadatas = self._collection.get(where=where, include=["metadatas"])["metadatas"] or []

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for md in metadatas:
            key = (str(md["source"]), str(md["department"]))
            g = groups.setdefault(
                key, {"file_type": md.get("file_type", ""), "doc_id": md.get("doc_id", ""), "n": 0, "pages": set()}
            )
            g["n"] += 1
            g["pages"].add(md.get("page"))

        docs = [
            DocumentInfo(
                source=src,
                department=dept,
                file_type=str(g["file_type"]),
                doc_id=str(g["doc_id"]),
                chunk_count=g["n"],
                page_count=len(g["pages"]),
            )
            for (src, dept), g in groups.items()
        ]
        return sorted(docs, key=lambda d: (d.department, d.source.lower()))

    def department_stats(self) -> dict[str, int]:
        """Departman başına belge sayısı (tanımlı tüm departmanlar dahil)."""
        stats = dict.fromkeys(self.settings.departments, 0)
        for doc in self.list_documents():
            stats[doc.department] = stats.get(doc.department, 0) + 1
        return stats

    def get_chunks(self, departments: list[str] | None = None) -> list[SearchResult]:
        """Filtreye uyan tüm parçalar (ileride BM25 indeksi için kullanılır)."""
        res = self._collection.get(where=build_where(departments), include=["documents", "metadatas"])
        return [
            SearchResult(id=i, text=t, metadata=dict(m), score=0.0)
            for i, t, m in zip(res["ids"], res["documents"] or [], res["metadatas"] or [], strict=True)
        ]

    def search(
        self,
        query: str,
        departments: list[str] | None = None,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Anlamsal arama; `departments` verilirse yalnızca o departmanlarda arar."""
        top_k = top_k or self.settings.top_k
        available = self.count(departments)
        if available == 0 or not query.strip():
            return []

        res = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, available),
            where=build_where(departments),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for cid, text, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            score = max(0.0, 1.0 - float(dist))
            if score >= min_score:
                results.append(SearchResult(id=cid, text=text, metadata=dict(md), score=round(score, 4)))
        return results
