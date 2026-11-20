"""Katmanlar arasında taşınan veri modelleri."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PageContent:
    """Yükleyiciden çıkan ham metin parçası (PDF için sayfa, diğerleri için bölüm)."""

    text: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Okunmuş doküman: içerik + kimlik bilgisi."""

    filename: str
    file_hash: str
    file_type: str
    pages: list[PageContent]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass(frozen=True, slots=True)
class Chunk:
    """Vektör veritabanına yazılacak metin parçası."""

    id: str
    text: str
    metadata: dict[str, str | int | float]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Sorgu sonucunda dönen, skorlu parça."""

    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int
    score: float  # 0..1 arası kosinüs benzerliği

    @property
    def citation_label(self) -> str:
        return f"{self.source}, s. {self.page}" if self.page else self.source


@dataclass(slots=True)
class IngestionResult:
    filename: str
    chunks_added: int
    skipped: bool = False
    message: str = ""


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.sources)


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Veritabanında kayıtlı doküman özeti (UI listesi için)."""

    filename: str
    file_hash: str
    file_type: str
    chunk_count: int
