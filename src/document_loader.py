"""Belge yükleme ve tablo yapısını koruyan akıllı chunking.

Akış
----
1. **Yükleme** — her format sayfa sayfa *Markdown metne* dönüştürülür:
   * PDF  : pdfplumber ile metin + tablolar (tablolar Markdown tabloya çevrilir,
            sayfadaki konumlarına göre sıralanır). Hata halinde pypdf'e düşülür.
   * DOCX : gövde sırayla dolaşılır (paragraf ve tablolar iç içe), başlıklar
            `#` ile işaretlenir, sayfa sonları sayılarak sayfa numarası tahmin edilir.
   * XLSX : her çalışma sayfası (sekme) bir "sayfa"dır; boş satırlarla ayrılmış
            her bölge ayrı bir Markdown tablosuna çevrilir.
   * TXT/MD: `\\f` ya da `<!-- pagebreak -->` sayfa ayırıcı olarak kabul edilir.
2. **Bloklama** — Markdown metin, *metin* ve *tablo* bloklarına ayrılır; her blok
   ait olduğu bölüm başlığını (section) taşır.
3. **Chunking** — metin blokları paragraf → satır → cümle → kelime sırasıyla
   örtüşmeli bölünür; tablolar asla satır ortasından bölünmez, büyük tablolar
   satır gruplarına ayrılır ve **her parçada başlık satırı tekrarlanır**.
   Aynı sayfadaki küçük ardışık parçalar `chunk_size`'a kadar birleştirilir.

Her chunk şu metadata'yı taşır: ``source``, ``page``, ``department``,
``file_type``, ``doc_id``, ``chunk_index``, ``content_type`` ve varsa
``section`` / ``sheet_name``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# =============================================================== Hatalar
class DocumentProcessingError(Exception):
    """Belge okunamadı / işlenemedi."""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Desteklenmeyen dosya uzantısı."""


# =============================================================== Veri modelleri
@dataclass(slots=True)
class PageContent:
    """Bir sayfanın (veya Excel sekmesinin) Markdown metni."""

    page: int
    text: str
    sheet_name: str | None = None


@dataclass(slots=True)
class Block:
    kind: str  # "text" | "table"
    text: str
    section: str | None = None


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(slots=True)
class ProcessedDocument:
    filename: str
    doc_id: str
    file_type: str
    department: str
    page_count: int
    chunks: list[DocumentChunk] = field(default_factory=list)


# =============================================================== Yardımcılar
_PAGEBREAK_RE = re.compile(r"\f|<!--\s*pagebreak\s*-->", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_TEXT_ENCODINGS = ("utf-8-sig", "cp1254", "latin-1")  # cp1254: Türkçe Windows
_TR_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.translate(_TR_ASCII)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "x"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "kurum-\nsal" -> "kurumsal"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 4)
    if isinstance(value, dt.datetime):
        value = value.date().isoformat() if value.time() == dt.time() else value.isoformat(sep=" ")
    elif isinstance(value, dt.date):
        value = value.isoformat()
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return re.sub(r"\s+", " ", text)


def rows_to_markdown(rows: list[list[object]]) -> str:
    """İlk satırı başlık kabul ederek Markdown tablo üretir. Boş sütunları atar."""
    cells = [[_format_cell(c) for c in row] for row in rows]
    cells = [r for r in cells if any(r)]
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]
    keep = [i for i in range(width) if any(r[i] for r in cells)]
    cells = [[r[i] for i in keep] for r in cells]
    header = [h or f"Sütun {i + 1}" for i, h in enumerate(cells[0])]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


# =============================================================== Yükleyiciler
def _load_pdf(data: bytes) -> list[PageContent]:
    try:
        return _load_pdf_plumber(data)
    except Exception as exc:  # noqa: BLE001 — her türlü ayrıştırma hatasında pypdf'e düş
        logger.warning("pdfplumber başarısız (%s); pypdf ile deneniyor.", exc)
        return _load_pdf_pypdf(data)


_SOFT_WRAP_NEXT_RE = re.compile(r"^[a-zçğıöşü0-9%(\"“'‘]")


def _unwrap_lines(text: str) -> str:
    """PDF'in görsel satır kaydırmalarını birleştirir: cümle ortasında bölünmüş satırlar tek satır olur.

    Önceki satır cümle sonu işaretiyle (. : ; ! ?) bitmiyorsa ve sonraki satır küçük harf/rakamla başlıyorsa
    satırlar birleştirilir. Başlıklar ("Madde 3 – …") ve büyük harfle başlayan yeni cümleler korunur.
    """
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (out and out[-1] and stripped and not out[-1].rstrip().endswith((".", ":", ";", "!", "?"))
                and _SOFT_WRAP_NEXT_RE.match(stripped)):
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        else:
            out.append(line)
    return "\n".join(out)


def _load_pdf_plumber(data: bytes) -> list[PageContent]:
    import pdfplumber

    pages: list[PageContent] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            x0, top, x1, bottom = page.bbox
            tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
            parts: list[str] = []
            cursor = top
            for table in tables:
                t_top, t_bottom = table.bbox[1], table.bbox[3]
                if t_top > cursor:  # tablodan önceki metin
                    parts.append(_unwrap_lines(page.crop((x0, cursor, x1, t_top)).extract_text() or ""))
                parts.append(rows_to_markdown(table.extract()))
                cursor = max(cursor, t_bottom)
            if cursor < bottom:
                parts.append(_unwrap_lines(page.crop((x0, cursor, x1, bottom)).extract_text() or ""))
            text = "\n\n".join(p for p in (s.strip() for s in parts) if p)
            pages.append(PageContent(page=number, text=_normalize_text(text)))
    return pages


def _load_pdf_pypdf(data: bytes) -> list[PageContent]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        return [
            PageContent(page=i, text=_normalize_text(_unwrap_lines(p.extract_text() or "")))
            for i, p in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"PDF okunamadı: {exc}") from exc


def _load_docx(data: bytes) -> list[PageContent]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"DOCX okunamadı: {exc}") from exc

    pages: dict[int, list[str]] = {1: []}
    page = 1

    def new_page() -> None:
        nonlocal page
        page += 1
        pages.setdefault(page, [])

    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "tbl":
            table = Table(element, document)
            md = rows_to_markdown([[c.text for c in row.cells] for row in table.rows])
            if md:
                pages[page].append(md)
            continue
        if tag != "p":
            continue

        para = Paragraph(element, document)
        # Word'ün kaydettiği "işlenmiş" sayfa sonu ve "önce sayfa sonu" özelliği → yeni sayfa önce başlar
        if element.xpath("./w:pPr/w:pageBreakBefore") or (
            para.text.strip() and element.xpath("./w:r[1]/w:lastRenderedPageBreak")
        ):
            if pages[page]:
                new_page()

        text = para.text.strip()
        if text:
            style = (para.style.name or "").lower() if para.style is not None else ""
            level = None
            if style == "title":
                level = 1
            elif m := re.match(r"(?:heading|başlık)\s*(\d)", style):
                level = min(int(m.group(1)) + 1, 6)
            elif style.startswith("list"):
                text = f"- {text}"
            pages[page].append(f"{'#' * level} {text}" if level else text)

        # Elle eklenmiş sayfa sonları (Ctrl+Enter) → sonraki içerik yeni sayfada
        for _ in element.xpath('.//w:br[@w:type="page"]'):
            new_page()

    return [
        PageContent(page=n, text=_normalize_text("\n\n".join(parts)))
        for n, parts in sorted(pages.items())
    ]


def _load_xlsx(data: bytes) -> list[PageContent]:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"XLSX okunamadı: {exc}") from exc

    pages: list[PageContent] = []
    try:
        for index, ws in enumerate(wb.worksheets, start=1):
            regions: list[list[list[object]]] = [[]]
            for row in ws.iter_rows(values_only=True):
                if any(c is not None and str(c).strip() for c in row):
                    regions[-1].append(list(row))
                elif regions[-1]:
                    regions.append([])  # boş satır → yeni tablo bölgesi
            tables = [md for region in regions if region and (md := rows_to_markdown(region))]
            if tables:
                body = "\n\n".join(tables)
                pages.append(PageContent(page=index, text=f"## {ws.title}\n\n{body}", sheet_name=ws.title))
    finally:
        wb.close()
    return pages


def _load_text(data: bytes) -> list[PageContent]:
    # Windows Not Defteri "Unicode" (UTF-16) kaydı: BOM varsa önce ona göre çöz
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return _split_text_pages(data.decode("utf-16"))
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(f"UTF-16 metin çözülemedi: {exc}") from exc
    if b"\x00" in data[:4096]:
        raise DocumentProcessingError("Metin dosyası ikili (binary) içerik barındırıyor; UTF-8 olarak kaydedip tekrar deneyin.")
    text = data.decode("latin-1")
    for enc in _TEXT_ENCODINGS:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return _split_text_pages(text)


def _split_text_pages(text: str) -> list[PageContent]:
    parts = _PAGEBREAK_RE.split(text.replace("\x00", ""))
    return [PageContent(page=i, text=_normalize_text(p)) for i, p in enumerate(parts, start=1)]


LOADERS: dict[str, Callable[[bytes], list[PageContent]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".xlsx": _load_xlsx,
    ".txt": _load_text,
    ".md": _load_text,
}
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(LOADERS)


def load_pages(filename: str, data: bytes) -> list[PageContent]:
    """Dosyayı okuyup içi dolu sayfaları döndürür."""
    ext = Path(filename).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"'{ext or filename}' desteklenmiyor. Desteklenen türler: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not data:
        raise DocumentProcessingError(f"'{filename}' boş bir dosya.")
    pages = [p for p in loader(data) if p.text]
    if not pages:
        hint = " (taranmış/görsel bir PDF olabilir)" if ext == ".pdf" else " (dosya yalnızca boşluk içeriyor)"
        raise DocumentProcessingError(f"'{filename}' içinden metin çıkarılamadı{hint}.")
    return pages


# =============================================================== Bloklama
def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def split_blocks(text: str, section: str | None = None) -> tuple[list[Block], str | None]:
    """Markdown metni metin/tablo bloklarına ayırır; güncel bölüm başlığını izler."""
    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_kind = "text"
    buffer_section = section

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            blocks.append(Block(buffer_kind, content, buffer_section))
        buffer = []

    for line in text.split("\n"):
        kind = "table" if _is_table_line(line) else "text"
        heading = _HEADING_RE.match(line.strip()) if kind == "text" else None
        if heading:
            # Her başlık yeni bir blok başlatır ve sonraki içeriğin bölümünü belirler.
            flush()
            section = heading.group(2).strip()
            buffer_kind, buffer_section = "text", section
        elif kind != buffer_kind:
            flush()
            buffer_kind, buffer_section = kind, section
        buffer.append(line)
    flush()
    return blocks, section


# =============================================================== Chunking
class SmartChunker:
    _SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_chars: int = 30):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap, chunk_size değerinden küçük olmalı")
        self.chunk_size = chunk_size
        self.min_chunk_chars = min_chunk_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._SEPARATORS,
            keep_separator="start",
            strip_whitespace=True,
        )

    def _split_table(self, table: str) -> list[str]:
        """Tabloyu satır sınırlarından böler; her parçada başlık + ayraç satırı tekrarlanır."""
        if len(table) <= self.chunk_size:
            return [table]
        lines = table.split("\n")
        header, rows = lines[:2], lines[2:]
        header_text = "\n".join(header)
        parts, current = [], []
        size = len(header_text)
        for row in rows:
            if current and size + len(row) + 1 > self.chunk_size:
                parts.append("\n".join([header_text, *current]))
                current, size = [], len(header_text)
            current.append(row)
            size += len(row) + 1
        if current:
            parts.append("\n".join([header_text, *current]))
        return parts

    def _pieces(self, block: Block) -> list[tuple[str, str]]:
        if block.kind == "table":
            return [("table", p) for p in self._split_table(block.text)]
        return [("text", p) for p in self._splitter.split_text(block.text)]

    def chunk(
        self,
        pages: list[PageContent],
        *,
        source: str,
        department: str,
        doc_id: str,
        file_type: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        section: str | None = None
        dept_slug = slugify(department)

        for page in pages:
            blocks, section = split_blocks(page.text, section)
            # (metin, içerik türleri, bölüm) — aynı sayfadaki küçük parçaları birleştir
            merged: list[tuple[str, set[str], str | None]] = []
            for block in blocks:
                for kind, piece in self._pieces(block):
                    piece = piece.strip()
                    if merged and len(merged[-1][0]) + len(piece) + 2 <= self.chunk_size:
                        text, kinds, sec = merged[-1]
                        merged[-1] = (f"{text}\n\n{piece}", kinds | {kind}, sec)
                    else:
                        merged.append((piece, {kind}, block.section))

            for text, kinds, sec in merged:
                if len(text) < self.min_chunk_chars:
                    continue
                index = len(chunks)
                metadata: dict[str, str | int] = {
                    "source": source,
                    "page": page.page,
                    "department": department,
                    "file_type": file_type,
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "content_type": "mixed" if len(kinds) > 1 else next(iter(kinds)),
                    "char_count": len(text),
                }
                # ChromaDB metadata'sı None kabul etmez → yalnızca varsa ekle
                if sec:
                    metadata["section"] = sec[:200]
                if page.sheet_name:
                    metadata["sheet_name"] = page.sheet_name
                chunks.append(
                    DocumentChunk(id=f"{doc_id[:12]}-{dept_slug}-{index:05d}", text=text, metadata=metadata)
                )
        return chunks


# =============================================================== Ana giriş noktası
def process_document(
    filename: str,
    data: bytes,
    department: str,
    settings: Settings | None = None,
) -> ProcessedDocument:
    """Dosyayı okur, parçalar ve metadata'lı chunk'larını döndürür."""
    settings = settings or get_settings()
    settings.validate_department(department)

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise DocumentProcessingError(f"'{filename}' {settings.max_file_size_mb} MB sınırını aşıyor.")

    name = Path(filename).name
    file_type = Path(name).suffix.lower().lstrip(".")
    pages = load_pages(name, data)
    doc_id = compute_doc_id(data)
    chunker = SmartChunker(settings.chunk_size, settings.chunk_overlap)
    chunks = chunker.chunk(pages, source=name, department=department, doc_id=doc_id, file_type=file_type)
    if not chunks:
        raise DocumentProcessingError(f"'{name}' içinden anlamlı metin parçası çıkarılamadı.")

    logger.info("%s → %d sayfa, %d parça (%s)", name, len(pages), len(chunks), department)
    return ProcessedDocument(name, doc_id, file_type, department, len(pages), chunks)


def process_file(path: str | Path, department: str, settings: Settings | None = None) -> ProcessedDocument:
    p = Path(path)
    return process_document(p.name, p.read_bytes(), department, settings)
