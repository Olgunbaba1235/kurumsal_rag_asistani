"""ChromaDB kalıcı vektör deposu.

Embedding'ler `Embedder` ile dışarıda üretilip Chroma'ya hazır vektör olarak
verilir; bu sayede Chroma'nın varsayılan (ONNX) embedding fonksiyonuna
bağımlılık olmaz ve sağlayıcı değiştirmek depoyu etkilemez.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_assistant.models import Chunk, IndexedDocument, RetrievedChunk
from rag_assistant.providers import Embedder

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 500


class ChromaVectorStore:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path | str | None,
        collection_name: str = "kurumsal_dokumanlar",
    ):
        self.embedder = embedder
        settings = ChromaSettings(anonymized_telemetry=False)
        if persist_dir is None:  # testler için bellek içi istemci
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    # ---------------------------------------------------------------- yazma
    def has_document(self, file_hash: str) -> bool:
        return bool(self._collection.get(where={"file_hash": file_hash}, limit=1, include=[])["ids"])

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
            )
        logger.info("%d parça vektör deposuna yazıldı.", len(chunks))
        return len(chunks)

    def delete_document(self, file_hash: str) -> None:
        self._collection.delete(where={"file_hash": file_hash})

    def reset(self) -> None:
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)

    # ---------------------------------------------------------------- okuma
    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[IndexedDocument]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        counts: Counter[str] = Counter()
        info: dict[str, tuple[str, str]] = {}
        for md in metadatas:
            h = str(md["file_hash"])
            counts[h] += 1
            info.setdefault(h, (str(md["source"]), str(md.get("file_type", ""))))
        docs = [
            IndexedDocument(filename=info[h][0], file_hash=h, file_type=info[h][1], chunk_count=n)
            for h, n in counts.items()
        ]
        return sorted(docs, key=lambda d: d.filename.lower())

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        file_hashes: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        where = None
        if file_hashes:
            where = (
                {"file_hash": file_hashes[0]}
                if len(file_hashes) == 1
                else {"file_hash": {"$in": file_hashes}}
            )
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        for cid, text, md, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            score = max(0.0, 1.0 - float(dist))  # kosinüs mesafesi → benzerlik
            if score < min_score:
                continue
            page = md.get("page")
            hits.append(
                RetrievedChunk(
                    id=cid,
                    text=text,
                    source=str(md["source"]),
                    page=int(page) if page is not None else None,
                    chunk_index=int(md.get("chunk_index", 0)),
                    score=round(score, 4),
                )
            )
        return hits
