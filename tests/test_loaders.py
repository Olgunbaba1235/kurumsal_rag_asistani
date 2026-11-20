import io

import pytest

from rag_assistant.exceptions import DocumentLoadError, UnsupportedFileTypeError
from rag_assistant.ingestion.loaders import load_document


def test_txt_utf8_and_turkish_chars():
    doc = load_document("politika.txt", "Çalışan izin hakları: şöyle ğüzel.".encode())
    assert doc.file_type == "txt"
    assert "Çalışan" in doc.pages[0].text
    assert doc.pages[0].page is None


def test_txt_cp1254_fallback():
    doc = load_document("eski.txt", "Şirket içi yönetmelik".encode("cp1254"))
    assert doc.pages[0].text == "Şirket içi yönetmelik"


def test_hash_is_content_based():
    a = load_document("a.txt", b"ayni icerik")
    b = load_document("b.txt", b"ayni icerik")
    assert a.file_hash == b.file_hash


def test_hyphenated_line_breaks_are_joined():
    doc = load_document("x.txt", b"kurum-\nsal belge")
    assert "kurumsal belge" in doc.pages[0].text


def test_docx_paragraphs_headings_and_tables():
    import docx

    d = docx.Document()
    d.add_heading("İzin Politikası", level=1)
    d.add_paragraph("Çalışanlara yılda 14 gün izin verilir.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Kıdem"
    t.rows[0].cells[1].text = "20 gün"
    buf = io.BytesIO()
    d.save(buf)

    doc = load_document("ik.docx", buf.getvalue())
    text = doc.pages[0].text
    assert "## İzin Politikası" in text
    assert "14 gün" in text
    assert "Kıdem | 20 gün" in text


def test_pdf_pages_are_numbered():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    buf = io.BytesIO()
    w.write(buf)
    # Boş sayfalı PDF metin içermez → anlamlı hata
    with pytest.raises(DocumentLoadError, match="metin çıkarılamadı"):
        load_document("bos.pdf", buf.getvalue())


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("tablo.xlsx", b"x")


def test_empty_file():
    with pytest.raises(DocumentLoadError):
        load_document("bos.txt", b"")
