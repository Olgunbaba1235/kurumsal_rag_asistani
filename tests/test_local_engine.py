"""src/local_engine.py — bellek içi hibrit indeks, çevrimdışı özetleyici ve tenant izolasyonu."""

from __future__ import annotations

import re

import pytest

from src.config import Settings
from src.document_loader import UnsupportedFileTypeError
from src.local_engine import OFFLINE_MODE, LocalWorkspace, OfflineSummarizer
from src.rag_engine import NO_INFO_ANSWER
from src.text_utils import fold

SAMPLE_QUESTIONS = [
    "Fazla mesai ücreti nasıl hesaplanır?",
    "Uzaktan çalışma kuralları nelerdir?",
    "Satın almada kaç teklif almak gerekir?",
    "Yurt içi seyahatte günlük yemek harcırahı ne kadar?",
]


def make_ws() -> LocalWorkspace:
    return LocalWorkspace(Settings(_env_file=None), api_key=None)


@pytest.fixture(scope="module")
def ws() -> LocalWorkspace:
    w = make_ws()
    assert w.load_sample_documents() == 3  # İK + Finans (md) + Hukuk sözleşmesi (pdf)
    return w


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", fold(text)).strip()


def test_sample_documents_are_indexed_with_departments(ws):
    docs = {d.filename: d for d in ws.documents}
    assert docs["IK_Izin_ve_Calisma_Yonetmeligi.md"].department == "İnsan Kaynakları"
    assert docs["Hukuk_Hizmet_Sozlesmesi.pdf"].department == "Hukuk"
    assert docs["Finans_Harcama_Proseduru.md"].department == "Finans"
    assert ws.store.chunk_count == sum(d.chunk_count for d in docs.values()) > 0
    assert ws.mode == OFFLINE_MODE


def test_reloading_samples_does_not_duplicate(ws):
    before = ws.store.chunk_count
    assert ws.load_sample_documents() == 0
    assert ws.store.chunk_count == before


@pytest.mark.parametrize("question", SAMPLE_QUESTIONS)
def test_demo_questions_get_cited_answers(ws, question):
    r = ws.ask(question)
    assert r.is_answered and r.confidence > 0
    assert r.citations, question
    for c in r.citations:
        assert c.page is not None and c.source.endswith(".md")
        assert f"[{c.label}]" in r.answer


def test_offline_answer_only_contains_document_sentences(ws):
    r = ws.ask("Fazla mesai ücreti nasıl hesaplanır?")
    corpus = norm(" ".join(c.text for c in ws.store.get_chunks()))
    bullets = [b for b in r.answer.splitlines() if b.startswith("- ")]
    assert bullets
    for b in bullets:
        sentence = re.sub(r"\s*\[[^\]]+\]\s*$", "", b[2:]).rstrip(".")
        for part in sentence.split(" · "):  # tablo satırları hücre hücre birleştirilir
            assert norm(part) in corpus, part


def test_out_of_scope_question_is_refused(ws):
    r = ws.ask("Şirketin Mars'taki ofisinin adresi nedir?")
    assert not r.is_answered and r.answer == NO_INFO_ANSWER and r.confidence == 0 and not r.citations


def test_department_filter_limits_sources(ws):
    r = ws.ask("Fazla mesai ücreti nasıl hesaplanır?", departments=["Finans"])
    assert all(c.department == "Finans" for c in r.citations)
    assert not any("IK_" in c.source for c in r.citations)


def test_history_is_recorded(ws):
    n = len(ws.history)
    ws.ask("Uzaktan çalışma kuralları nelerdir?")
    assert len(ws.history) == n + 1 and ws.history[-1].citations >= 1


def test_workspaces_are_isolated():
    a, b = make_ws(), make_ws()
    a.load_sample_documents()
    assert b.store.chunk_count == 0 and b.documents == []
    assert b.ask("Fazla mesai ücreti nasıl hesaplanır?").answer == NO_INFO_ANSWER


def test_upload_duplicate_and_remove():
    w = make_ws()
    data = "# Destek SLA\n\nKritik arızalarda müdahale süresi en fazla 4 saattir ve 7/24 destek verilir.".encode()
    doc, added = w.add_bytes("SLA.md", data, "Teknik")
    assert added and w.store.chunk_count == len(doc.chunks)
    assert w.add_bytes("SLA_kopya.md", data, "Teknik")[1] is False  # aynı içerik
    r = w.ask("Kritik arızalarda müdahale süresi nedir?")
    assert r.is_answered and r.citations[0].source == "SLA.md"
    rev = w.store.revision
    w.store.remove_document(doc.doc_id)
    assert w.store.chunk_count == 0 and w.store.revision == rev + 1
    assert w.ask("Kritik arızalarda müdahale süresi nedir?").answer == NO_INFO_ANSWER


def test_unsupported_file_and_unknown_department_are_rejected():
    w = make_ws()
    with pytest.raises(UnsupportedFileTypeError):
        w.add_bytes("virus.exe", b"MZ", "Genel")
    with pytest.raises(ValueError):
        w.add_bytes("a.md", b"# Bas\n\nYeterince uzun bir paragraf metni burada yer almaktadir.", "Yok Departman")


def test_summarizer_handles_missing_context():
    s = OfflineSummarizer()
    assert s.answer("SORU: x") == NO_INFO_ANSWER
    with pytest.raises(Exception):  # noqa: B017 - yapısal çıktı çevrimdışı yok
        s.generate_json("p", dict)
