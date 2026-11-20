import pytest

from rag_assistant.ingestion.chunking import SmartChunker
from rag_assistant.models import LoadedDocument, PageContent


def _doc(pages):
    return LoadedDocument(filename="d.pdf", file_hash="a" * 64, file_type="pdf", pages=pages)


def test_respects_chunk_size_and_keeps_page_metadata():
    para = "Bu bir test cümlesidir ve politika hakkında bilgi verir. " * 30
    chunks = SmartChunker(chunk_size=300, chunk_overlap=50).split(
        _doc([PageContent(para, page=1), PageContent(para, page=2)])
    )
    assert len(chunks) > 2
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_ids_are_deterministic_and_unique():
    doc = _doc([PageContent("Paragraf bir. " * 80)])
    a = SmartChunker(300, 50).split(doc)
    b = SmartChunker(300, 50).split(doc)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_tiny_fragments_are_dropped():
    chunks = SmartChunker(300, 50).split(_doc([PageContent("12")]))
    assert chunks == []


def test_no_page_key_when_page_is_none():
    chunks = SmartChunker(300, 50).split(_doc([PageContent("Sayfa numarası olmayan yeterince uzun bir metin.")]))
    assert "page" not in chunks[0].metadata


def test_invalid_overlap():
    with pytest.raises(ValueError):
        SmartChunker(100, 100)
