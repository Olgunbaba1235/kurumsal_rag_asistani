"""PDF / DOCX / TXT yükleyicileri.

Her yükleyici ham byte alır ve `PageContent` listesi döndürür. Böylece
dosya sistemi ya da Streamlit `UploadedFile` farkı yükleyiciyi etkilemez.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from collections.abc import Callable
from pathlib import Path

from rag_assistant.exceptions import DocumentLoadError, UnsupportedFileTypeError
from rag_assistant.models import LoadedDocument, PageContent

logger = logging.getLogger(__name__)

_TEXT_ENCODINGS = ("utf-8-sig", "cp1254", "latin-1")  # cp1254: Türkçe Windows


def _normalize(text: str) -> str:
    """Fazla boşlukları temizler, paragraf sınırlarını korur."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # PDF'lerde satır sonunda tireyle bölünmüş kelimeleri birleştir: "kurum-\nsal" -> "kurumsal"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _load_pdf(data: bytes) -> list[PageContent]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise DocumentLoadError("PDF şifreli; okunamıyor.") from exc
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = _normalize(page.extract_text() or "")
            if text:
                pages.append(PageContent(text=text, page=i))
        return pages
    except PdfReadError as exc:
        raise DocumentLoadError(f"PDF okunamadı: {exc}") from exc


def _load_docx(data: bytes) -> list[PageContent]:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, KeyError, ValueError) as exc:
        raise DocumentLoadError(f"DOCX okunamadı: {exc}") from exc

    blocks: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        # Başlıkları işaretle: chunker başlık sınırlarından bölmeyi tercih eder.
        if style.startswith("heading") or style.startswith("başlık") or style == "title":
            blocks.append(f"\n## {text}")
        else:
            blocks.append(text)

    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append("\n".join(rows))

    text = _normalize("\n\n".join(blocks))
    return [PageContent(text=text)] if text else []


def _load_txt(data: bytes) -> list[PageContent]:
    for enc in _TEXT_ENCODINGS:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover — latin-1 her zaman çözer
        raise DocumentLoadError("Metin dosyasının kodlaması çözülemedi.")
    text = _normalize(text)
    return [PageContent(text=text)] if text else []


_LOADERS: dict[str, Callable[[bytes], list[PageContent]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
    ".md": _load_txt,
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(_LOADERS)


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_document(filename: str, data: bytes) -> LoadedDocument:
    """Dosya adını ve içeriğini alıp okunmuş doküman döndürür."""
    ext = Path(filename).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"'{ext or filename}' desteklenmiyor. Desteklenenler: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not data:
        raise DocumentLoadError(f"'{filename}' boş bir dosya.")

    pages = loader(data)
    if not pages:
        raise DocumentLoadError(
            f"'{filename}' içinden metin çıkarılamadı (taranmış/görsel PDF olabilir)."
        )

    doc = LoadedDocument(
        filename=Path(filename).name,
        file_hash=compute_hash(data),
        file_type=ext.lstrip("."),
        pages=pages,
    )
    logger.info("Yüklendi: %s (%d bölüm, %d karakter)", doc.filename, len(pages), doc.char_count)
    return doc
