"""RAG orkestrasyonu: yükle → parçala → indeksle → getir → üret."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.exceptions import DocumentLoadError
from rag_assistant.ingestion import SmartChunker, load_document
from rag_assistant.models import IngestionResult, RAGAnswer, RetrievedChunk
from rag_assistant.prompts import SYSTEM_PROMPT, build_user_message
from rag_assistant.providers import ChatModel, Embedder, GeminiChatModel, GeminiEmbedder
from rag_assistant.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "Yüklenen dokümanlarda bu soruyla yeterince ilgili bir bölüm bulamadım. "
    "Soruyu farklı ifade etmeyi deneyin veya ilgili dokümanı yükleyin."
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_HISTORY_TURNS = 3  # bağlama eklenecek önceki soru-cevap çifti sayısı


def cited_indices(answer: str, n_sources: int) -> list[int]:
    """Yanıtta geçen geçerli [n] atıflarını (1 tabanlı, sıralı, tekil) döndürür."""
    found = {int(m) for m in _CITATION_RE.findall(answer)}
    return sorted(i for i in found if 1 <= i <= n_sources)


@dataclass
class StreamingAnswer:
    """UI'nin akış halinde gösterebileceği yanıt + kaynaklar."""

    question: str
    sources: list[RetrievedChunk]
    stream: Iterator[str]


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, llm: ChatModel, store: ChromaVectorStore):
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = store
        self.chunker = SmartChunker(settings.chunk_size, settings.chunk_overlap)

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        settings.ensure_dirs()
        embedder = GeminiEmbedder(settings.api_key, settings.embedding_model, settings.embedding_dimension)
        llm = GeminiChatModel(settings.api_key, settings.llm_model, settings.llm_temperature)
        # Embedding modeli/boyutu değişirse eski vektörlerle karışmasın diye koleksiyon adına ekle.
        collection = f"{settings.collection_name}_{settings.embedding_dimension}"
        store = ChromaVectorStore(embedder, settings.chroma_dir, collection)
        return cls(settings, embedder, llm, store)

    # ------------------------------------------------------------ ingestion
    def ingest(self, filename: str, data: bytes, *, save_copy: bool = True) -> IngestionResult:
        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise DocumentLoadError(
                f"'{filename}' {self.settings.max_file_size_mb} MB sınırını aşıyor."
            )

        document = load_document(filename, data)
        if self.store.has_document(document.file_hash):
            return IngestionResult(document.filename, 0, skipped=True, message="Zaten indekslenmiş.")

        chunks = self.chunker.split(document)
        added = self.store.add_chunks(chunks)
        if save_copy:
            target = self.settings.upload_dir / f"{document.file_hash[:12]}_{document.filename}"
            if not target.exists():
                target.write_bytes(data)
        return IngestionResult(document.filename, added, message=f"{added} parça indekslendi.")

    def ingest_path(self, path: Path | str) -> IngestionResult:
        p = Path(path)
        return self.ingest(p.name, p.read_bytes(), save_copy=False)

    def delete_document(self, file_hash: str) -> None:
        self.store.delete_document(file_hash)
        for f in self.settings.upload_dir.glob(f"{file_hash[:12]}_*"):
            f.unlink(missing_ok=True)

    # ------------------------------------------------------------ retrieval
    def retrieve(
        self, question: str, top_k: int | None = None, file_hashes: list[str] | None = None
    ) -> list[RetrievedChunk]:
        return self.store.search(
            question,
            top_k=top_k or self.settings.top_k,
            min_score=self.settings.min_relevance,
            file_hashes=file_hashes,
        )

    def _build_messages(
        self, question: str, sources: Sequence[RetrievedChunk], history: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        recent = list(history)[-_HISTORY_TURNS * 2 :]
        return [*recent, ("user", build_user_message(question, sources))]

    def ask_stream(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        *,
        top_k: int | None = None,
        temperature: float | None = None,
        file_hashes: list[str] | None = None,
    ) -> StreamingAnswer:
        question = question.strip()
        sources = self.retrieve(question, top_k, file_hashes)
        if not sources:
            return StreamingAnswer(question, [], iter([NO_CONTEXT_ANSWER]))
        stream = self.llm.stream(SYSTEM_PROMPT, self._build_messages(question, sources, history), temperature)
        return StreamingAnswer(question, sources, stream)

    def ask(self, question: str, history: Sequence[tuple[str, str]] = (), **kwargs) -> RAGAnswer:
        result = self.ask_stream(question, history, **kwargs)
        return RAGAnswer(result.question, "".join(result.stream), result.sources)
