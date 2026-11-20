"""Depolama arka ucundan bağımsız arama tipleri.

ChromaDB (`src.vector_store`) ve PostgreSQL/pgvector (`app.services`) aynı
`SearchResult` tipini üretir; böylece hibrit arama ve reranking kodu her iki
arka uçta da aynen kullanılır ve API katmanı ChromaDB'ye bağımlı olmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # 0..1 kosinüs benzerliği

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page is not None else None

    @property
    def department(self) -> str:
        return str(self.metadata.get("department", ""))

    @property
    def citation(self) -> str:
        return f"[{self.source} - Sayfa {self.page}]" if self.page else f"[{self.source}]"
