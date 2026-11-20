"""Akıllı chunking.

Strateji:
* `RecursiveCharacterTextSplitter` önce başlık / paragraf, sonra cümle,
  en son kelime sınırından böler — anlamsal bütünlük korunur.
* Parçalar arasında örtüşme (overlap) bağlam kaybını azaltır.
* Her parça sayfa numarası ve kaynak dosya bilgisini taşır (atıf için).
* Çok kısa, anlamsız parçalar (ör. yalnızca sayfa numarası) elenir.
* Parça kimlikleri dosya içeriğinin hash'inden türetilir → aynı dosya
  tekrar yüklendiğinde kopya oluşmaz (idempotent upsert).
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.models import Chunk, LoadedDocument

_SEPARATORS = [
    "\n## ",  # DOCX başlıkları
    "\n\n",  # paragraf
    "\n",  # satır
    ". ",  # cümle
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]


class SmartChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, min_chunk_chars: int = 40):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap, chunk_size değerinden küçük olmalı")
        self.min_chunk_chars = min_chunk_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_SEPARATORS,
            keep_separator="end",
            strip_whitespace=True,
        )

    def split(self, document: LoadedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page in document.pages:
            for piece in self._splitter.split_text(page.text):
                piece = piece.removeprefix("## ").strip()
                if len(piece) < self.min_chunk_chars:
                    continue
                metadata: dict[str, str | int | float] = {
                    "source": document.filename,
                    "file_hash": document.file_hash,
                    "file_type": document.file_type,
                    "chunk_index": index,
                    "char_count": len(piece),
                }
                # ChromaDB metadata değerleri None olamaz → yalnızca varsa ekle.
                if page.page is not None:
                    metadata["page"] = page.page
                chunks.append(
                    Chunk(id=f"{document.file_hash[:16]}-{index:05d}", text=piece, metadata=metadata)
                )
                index += 1
        return chunks
