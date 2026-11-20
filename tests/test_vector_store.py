import uuid

import pytest

from src.config import Settings
from src.document_loader import process_document, process_file
from src.vector_store import IN_MEMORY, VectorStore, build_where
from tests.helpers import FakeEmbedder


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(CHROMA_PERSIST_DIR=tmp_path / "chroma", _env_file=None)


@pytest.fixture
def store(settings) -> VectorStore:
    return VectorStore(FakeEmbedder(), settings, persist_dir=IN_MEMORY, collection_name=f"t_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def loaded(store, settings) -> VectorStore:
    store.add_document(
        process_file(settings.sample_docs_dir / "IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları", settings)
    )
    store.add_document(process_file(settings.sample_docs_dir / "Finans_Harcama_Proseduru.md", "Finans", settings))
    return store


def test_build_where():
    assert build_where() is None
    assert build_where(["Finans"]) == {"department": "Finans"}
    assert build_where(["Finans", "Genel"]) == {"department": {"$in": ["Finans", "Genel"]}}
    assert build_where(["Finans"], source="a.md") == {"$and": [{"department": "Finans"}, {"source": "a.md"}]}


def test_list_documents_and_stats(loaded):
    docs = loaded.list_documents()
    assert [(d.department, d.source) for d in docs] == [
        ("Finans", "Finans_Harcama_Proseduru.md"),
        ("İnsan Kaynakları", "IK_Izin_ve_Calisma_Yonetmeligi.md"),
    ]
    assert all(d.page_count == 4 and d.chunk_count > 0 for d in docs)
    assert loaded.department_stats() == {"Genel": 0, "İnsan Kaynakları": 1, "Finans": 1, "Teknik": 0}
    assert [d.source for d in loaded.list_documents("Finans")] == ["Finans_Harcama_Proseduru.md"]


def test_search_respects_department_isolation(loaded):
    q = "yıllık izin kaç iş günü kıdem"
    ik_hits = loaded.search(q, departments=["İnsan Kaynakları"])
    fin_hits = loaded.search(q, departments=["Finans"])
    assert ik_hits and all(h.department == "İnsan Kaynakları" for h in ik_hits)
    assert fin_hits and all(h.department == "Finans" for h in fin_hits)
    assert loaded.search(q, departments=["Teknik"]) == []
    # filtresiz arama her iki departmanı da görür; en alakalı sonuç İK'dan gelir
    assert loaded.search(q)[0].department == "İnsan Kaynakları"


def test_search_results_carry_citation_metadata(loaded):
    hit = loaded.search("harcama onay yetkileri tutar CFO genel müdür", departments=["Finans"])[0]
    assert hit.source == "Finans_Harcama_Proseduru.md"
    assert hit.page is not None
    assert hit.citation == f"[Finans_Harcama_Proseduru.md - Sayfa {hit.page}]"
    assert 0.0 < hit.score <= 1.0


def test_duplicate_is_skipped_and_new_version_replaces(store, settings):
    v1 = process_document("politika.md", b"Ilk surum: uzaktan calisma haftada iki gundur.", "Genel", settings)
    v2 = process_document("politika.md", b"Ikinci surum: uzaktan calisma haftada uc gundur.", "Genel", settings)

    assert store.add_document(v1).chunks_added == 1
    assert store.add_document(v1).skipped
    res = store.add_document(v2)
    assert res.replaced and res.chunks_added == 1
    assert store.count() == 1
    assert "Ikinci surum" in store.get_chunks(["Genel"])[0].text


def test_same_file_in_two_departments_is_isolated(store, settings):
    data = b"Ortak bilgi guvenligi politikasi tum calisanlar icin gecerlidir."
    store.add_document(process_document("guvenlik.md", data, "Genel", settings))
    store.add_document(process_document("guvenlik.md", data, "Teknik", settings))
    assert store.count() == 2  # farklı departmanlarda farklı kimlikler

    assert store.delete_document("guvenlik.md", department="Teknik") == 1
    assert [d.department for d in store.list_documents()] == ["Genel"]
    assert store.delete_document("guvenlik.md") == 1
    assert store.count() == 0


def test_persistence_across_instances(settings, tmp_path):
    path = tmp_path / "kalici"
    s1 = VectorStore(FakeEmbedder(), settings, persist_dir=path)
    s1.add_document(process_document("kalici.md", b"Kalici depolama testi icin yeterince uzun metin.", "Teknik", settings))
    del s1

    s2 = VectorStore(FakeEmbedder(), settings, persist_dir=path)
    assert [d.source for d in s2.list_documents("Teknik")] == ["kalici.md"]


def test_invalid_department_rejected(store, settings):
    doc = process_document("a.md", b"yeterince uzun bir metin icerigi burada", "Genel", settings)
    doc.department = "Pazarlama"
    with pytest.raises(ValueError):
        store.add_document(doc)
