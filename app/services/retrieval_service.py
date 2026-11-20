"""PostgreSQL/pgvector tabanlı hibrit arama (vektör + BM25 → RRF → reranking).

Çekirdek mantık (`fuse_rankings`, `BM25Index`, reranker'lar) `src.hybrid_retriever`
ile ortaktır; burada yalnızca veri erişimi PostgreSQL'e uyarlanır:

* **Vektör kolu**: pgvector ``<=>`` (kosinüs mesafesi) + HNSW indeksi. Sorgu
  her zaman ``tenant_id`` VE ``department IN (...)`` ile filtrelenir; RLS ayrıca
  veritabanı düzeyinde aynı kısıtı uygular. Filtreli HNSW'de eksik sonuç
  dönmemesi için transaction'da ``hnsw.iterative_scan = relaxed_order`` açıktır.
* **BM25 kolu**: tenant + departman kümesinin parçaları için BM25Plus indeksi
  süreç içinde LRU önbellekte tutulur. Önbellek anahtarı belge sayısı ve en son
  güncelleme zamanını içerir → belge eklenince/silinince otomatik yenilenir.
  Önbellek anahtarında tenant_id bulunduğundan indeksler tenant'lar arasında
  asla paylaşılmaz.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import ApiSettings
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.tenant_service import TenantContext
from src.embeddings import Embedder
from src.hybrid_retriever import BM25Index, Reranker, RetrievedChunk, fuse_rankings, unique_queries
from src.search_types import SearchResult

logger = logging.getLogger(__name__)

_CHUNK_COLUMNS = (
    DocumentChunk.id,
    DocumentChunk.content,
    DocumentChunk.source,
    DocumentChunk.page,
    DocumentChunk.department,
    DocumentChunk.section,
    DocumentChunk.sheet_name,
    DocumentChunk.content_type,
    DocumentChunk.chunk_index,
    DocumentChunk.document_id,
)


def _to_result(row, score: float = 0.0) -> SearchResult:  # noqa: ANN001
    metadata = {
        "source": row.source,
        "page": row.page,
        "department": row.department,
        "content_type": row.content_type,
        "chunk_index": row.chunk_index,
        "document_id": str(row.document_id),
    }
    if row.section:
        metadata["section"] = row.section
    if row.sheet_name:
        metadata["sheet_name"] = row.sheet_name
    return SearchResult(id=str(row.id), text=row.content, metadata=metadata, score=score)


class PgHybridRetriever:
    def __init__(self, settings: ApiSettings, embedder: Embedder, reranker: Reranker, *, variant_weight: float = 0.7):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.variant_weight = variant_weight
        self._cache: OrderedDict[tuple, BM25Index] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    # ------------------------------------------------------------ vektör kolu
    async def vector_search(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        query_vector: Sequence[float],
        departments: Sequence[str],
        limit: int,
    ) -> list[SearchResult]:
        distance = DocumentChunk.embedding.cosine_distance(list(query_vector)).label("distance")
        stmt = (
            select(*_CHUNK_COLUMNS, distance)
            .where(DocumentChunk.tenant_id == tenant.id, DocumentChunk.department.in_(list(departments)))
            .order_by(distance)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        # relaxed_order iterative scan sonuçları hafifçe sırasız döndürebilir → yeniden sırala
        rows.sort(key=lambda r: r.distance)
        return [_to_result(r, max(0.0, 1.0 - float(r.distance))) for r in rows]

    # ------------------------------------------------------------ BM25 kolu
    async def bm25_index(self, session: AsyncSession, tenant: TenantContext, departments: Sequence[str]) -> BM25Index:
        depts = tuple(sorted(departments))
        version = (
            await session.execute(
                select(func.count(Document.id), func.max(Document.updated_at)).where(
                    Document.tenant_id == tenant.id,
                    Document.department.in_(depts),
                    Document.status == DocumentStatus.READY,
                )
            )
        ).one()
        key = (tenant.id, depts, int(version[0]), version[1])

        async with self._cache_lock:
            index = self._cache.get(key)
            if index is not None:
                self._cache.move_to_end(key)
                return index

        cap = self.settings.bm25_max_chunks
        rows = (
            await session.execute(
                select(*_CHUNK_COLUMNS)
                .where(DocumentChunk.tenant_id == tenant.id, DocumentChunk.department.in_(depts))
                .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                .limit(cap + 1)
            )
        ).all()
        if len(rows) > cap:
            logger.warning("[%s] BM25 indeksi %d parça ile sınırlandı (BM25_MAX_CHUNKS).", tenant.slug, cap)
            rows = rows[:cap]
        index = await asyncio.to_thread(BM25Index, [_to_result(r) for r in rows])

        async with self._cache_lock:
            # Aynı tenant/departman kümesinin eski sürümlerini at
            for stale in [k for k in self._cache if k[:2] == key[:2] and k != key]:
                del self._cache[stale]
            self._cache[key] = index
            while len(self._cache) > self.settings.bm25_cache_size:
                self._cache.popitem(last=False)
        return index

    # ------------------------------------------------------------ ana giriş
    async def retrieve(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        query: str,
        departments: Sequence[str],
        *,
        top_k: int,
        extra_queries: Sequence[str] = (),
    ) -> tuple[list[RetrievedChunk], dict[str, float], str]:
        """Dönen değer: (alaka sırasıyla parçalar, süreler, kullanılan reranker)."""
        if not departments:
            raise ValueError("En az bir departman gereklidir.")
        s = self.settings
        t0 = time.perf_counter()
        queries = unique_queries(query, extra_queries)
        q_vectors = await asyncio.to_thread(lambda: [self.embedder.embed_query(q) for q in queries])
        t_embed = time.perf_counter()

        vector_runs = [await self.vector_search(session, tenant, v, departments, s.candidate_k) for v in q_vectors]
        index = await self.bm25_index(session, tenant, departments)
        bm25_runs = [index.search(q, s.candidate_k) for q in queries]
        candidates = fuse_rankings(
            vector_runs,
            bm25_runs,
            rrf_k=s.rrf_k,
            vector_weight=s.vector_weight,
            bm25_weight=s.bm25_weight,
            variant_weight=self.variant_weight,
            limit=s.candidate_k,
        )
        t_search = time.perf_counter()

        def _rerank() -> tuple[list[RetrievedChunk], str]:
            ranked = self.reranker.rerank(query, candidates, index.idf) if candidates else []
            return ranked, getattr(self.reranker, "last_used", None) or self.reranker.name

        ranked, reranker_used = await asyncio.to_thread(_rerank)
        t_rerank = time.perf_counter()
        timings = {
            "embed_ms": round((t_embed - t0) * 1000, 1),
            "search_ms": round((t_search - t_embed) * 1000, 1),
            "rerank_ms": round((t_rerank - t_search) * 1000, 1),
        }
        return ranked[:top_k], timings, reranker_used
