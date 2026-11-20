import datetime as dt
import io

import pytest

from src.config import Settings
from src.document_loader import (
    DocumentProcessingError,
    SmartChunker,
    UnsupportedFileTypeError,
    load_pages,
    process_document,
    process_file,
    slugify,
    split_blocks,
)
from tests.helpers import make_pdf

REQUIRED_KEYS = {"source", "page", "department", "doc_id", "chunk_index", "file_type", "content_type"}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(CHROMA_PERSIST_DIR=tmp_path / "chroma", _env_file=None)


# ------------------------------------------------------------------ config
def test_config_defaults(settings):
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.departments == ["Genel", "İnsan Kaynakları", "Finans", "Teknik"]
    assert settings.chroma_persist_dir.is_absolute()


def test_config_rejects_bad_overlap():
    with pytest.raises(ValueError):
        Settings(CHUNK_SIZE=500, CHUNK_OVERLAP=500, _env_file=None)


# ------------------------------------------------------------------ TXT / MD
def test_markdown_pagebreaks_and_metadata(settings):
    md = "# Başlık\n\nBirinci sayfa içeriği yeterince uzun bir paragraf.\n<!-- pagebreak -->\nİkinci sayfa içeriği de yeterince uzun bir paragraf."
    doc = process_document("politika.md", md.encode(), "İnsan Kaynakları", settings)
    assert doc.page_count == 2
    assert [c.metadata["page"] for c in doc.chunks] == [1, 2]
    for c in doc.chunks:
        assert REQUIRED_KEYS <= c.metadata.keys()
        assert c.metadata["source"] == "politika.md"
        assert c.metadata["department"] == "İnsan Kaynakları"
    assert doc.chunks[0].metadata["section"] == "Başlık"


def test_txt_cp1254_fallback(settings):
    doc = process_document("eski.txt", "Şirket içi yönetmelik metni burada yer alıyor.".encode("cp1254"), "Genel", settings)
    assert "Şirket içi" in doc.chunks[0].text


def test_split_blocks_separates_tables_and_tracks_sections():
    text = "## İzinler\nGiriş metni\n| A | B |\n| --- | --- |\n| 1 | 2 |\nSon metin"
    blocks, section = split_blocks(text)
    assert [b.kind for b in blocks] == ["text", "table", "text"]
    assert all(b.section == "İzinler" for b in blocks)
    assert section == "İzinler"


def test_large_table_split_repeats_header():
    header = "| Kıdem | Süre |\n| --- | --- |"
    rows = "\n".join(f"| {i}. yıl kıdem kademesi açıklaması | {i} gün |" for i in range(60))
    pages = load_pages("t.md", f"{header}\n{rows}".encode())
    chunks = SmartChunker(400, 50).chunk(pages, source="t.md", department="Genel", doc_id="a" * 64, file_type="md")
    assert len(chunks) > 1
    for c in chunks:
        lines = c.text.split("\n")
        assert lines[0] == "| Kıdem | Süre |" and lines[1] == "| --- | --- |"
        assert all(line.startswith("|") and line.endswith("|") for line in lines)  # satır bölünmemiş
        assert c.metadata["content_type"] == "table"
    all_rows = [line for c in chunks for line in c.text.split("\n")[2:]]
    assert len(all_rows) == 60  # hiçbir satır kaybolmadı / tekrarlanmadı


def test_chunk_size_respected_and_ids_deterministic(settings):
    text = ("Bu cümle politika hakkında bilgi verir. " * 200).encode()
    a = process_document("uzun.txt", text, "Genel", settings)
    b = process_document("uzun.txt", text, "Genel", settings)
    assert len(a.chunks) > 3
    assert all(len(c.text) <= settings.chunk_size for c in a.chunks)
    assert [c.id for c in a.chunks] == [c.id for c in b.chunks]
    assert len({c.id for c in a.chunks}) == len(a.chunks)


# ------------------------------------------------------------------ XLSX
def test_xlsx_sheets_become_pages_with_markdown_tables(settings):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Onay Limitleri"
    ws.append(["Tutar", "Onaylayan", "Tarih"])
    ws.append([5000.0, "Yönetici", dt.datetime(2026, 1, 15)])
    ws.append([50000, "Direktör | CFO", None])
    ws2 = wb.create_sheet("Harcırah")
    ws2.append(["Kalem", "Limit"])
    ws2.append(["Otel", 4500])
    buf = io.BytesIO()
    wb.save(buf)

    doc = process_document("limitler.xlsx", buf.getvalue(), "Finans", settings)
    by_page = {c.metadata["page"]: c for c in doc.chunks}
    assert set(by_page) == {1, 2}
    first = by_page[1]
    assert first.metadata["sheet_name"] == "Onay Limitleri"
    assert "| Tutar | Onaylayan | Tarih |" in first.text
    assert "| 5000 | Yönetici | 2026-01-15 |" in first.text
    assert "Direktör \\| CFO" in first.text  # boru karakteri kaçırıldı
    assert by_page[2].metadata["sheet_name"] == "Harcırah"


# ------------------------------------------------------------------ DOCX
def test_docx_headings_tables_and_page_breaks(settings):
    import docx

    d = docx.Document()
    d.add_heading("Yıllık İzin", level=1)
    d.add_paragraph("Çalışanlara kıdemlerine göre yıllık izin verilir.")
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Kıdem", "Süre"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "1-5 yıl", "14 gün"
    d.add_page_break()
    d.add_heading("Uzaktan Çalışma", level=1)
    d.add_paragraph("Haftada en fazla iki gün uzaktan çalışılabilir.")
    buf = io.BytesIO()
    d.save(buf)

    doc = process_document("yonetmelik.docx", buf.getvalue(), "İnsan Kaynakları", settings)
    p1 = " ".join(c.text for c in doc.chunks if c.metadata["page"] == 1)
    p2 = " ".join(c.text for c in doc.chunks if c.metadata["page"] == 2)
    assert "## Yıllık İzin" in p1
    assert "| Kıdem | Süre |" in p1 and "| 1-5 yıl | 14 gün |" in p1
    assert "Uzaktan Çalışma" in p2 and "Yıllık İzin" not in p2


# ------------------------------------------------------------------ PDF
def test_pdf_page_numbers(settings):
    data = make_pdf(["Annual leave policy grants fourteen days.", "Expense reports are due in thirty days."])
    doc = process_document("policy.pdf", data, "Genel", settings)
    assert {c.metadata["page"]: c.text for c in doc.chunks} == {
        1: "Annual leave policy grants fourteen days.",
        2: "Expense reports are due in thirty days.",
    }


# ------------------------------------------------------------------ örnek belgeler
def test_sample_docs_are_processed_with_tables_on_correct_pages(settings):
    ik = process_file(settings.sample_docs_dir / "IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları", settings)
    fin = process_file(settings.sample_docs_dir / "Finans_Harcama_Proseduru.md", "Finans", settings)
    assert ik.page_count == 4 and fin.page_count == 4

    leave_table = next(c for c in ik.chunks if "| 15 yıl (dahil) ve daha fazla | 26 iş günü |" in c.text)
    assert leave_table.metadata["page"] == 2
    assert "| Kıdem | Yıllık İzin Süresi | Açıklama |" in leave_table.text

    approval = next(c for c in fin.chunks if "250.001 – 1.000.000 TL" in c.text)
    assert approval.metadata["page"] == 1
    assert approval.metadata["section"].startswith("3. Harcama Onay Yetkileri")


# ------------------------------------------------------------------ hatalar
def test_errors(settings):
    with pytest.raises(UnsupportedFileTypeError):
        process_document("sunum.pptx", b"x", "Genel", settings)
    with pytest.raises(DocumentProcessingError):
        process_document("bos.txt", b"", "Genel", settings)
    with pytest.raises(ValueError, match="Geçersiz departman"):
        process_document("a.txt", b"yeterince uzun bir metin icerigi burada", "Pazarlama", settings)


def test_slugify_turkish():
    assert slugify("İnsan Kaynakları") == "insan-kaynaklari"
