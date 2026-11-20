from rag_assistant.pipeline import NO_CONTEXT_ANSWER, cited_indices

HR = (
    "İnsan Kaynakları Politikası\n\n"
    "Yıllık izin: Çalışanlara her yıl 14 gün yıllık izin hakkı tanınır. "
    "Beş yılı aşan kıdemde yıllık izin 20 gündür.\n\n"
    "Uzaktan çalışma: Haftada iki gün uzaktan çalışma mümkündür."
)
FIN = (
    "Satın Alma Prosedürü\n\n"
    "10.000 TL üzerindeki satın almalar finans direktörü onayı gerektirir. "
    "Teklifler en az üç tedarikçiden alınır."
)


def test_ingest_and_dedupe(pipeline):
    r1 = pipeline.ingest("ik.txt", HR.encode())
    assert r1.chunks_added > 0 and not r1.skipped
    r2 = pipeline.ingest("ik_kopya.txt", HR.encode())
    assert r2.skipped
    assert len(pipeline.store.list_documents()) == 1
    assert any(pipeline.settings.upload_dir.iterdir())


def test_retrieval_ranks_relevant_doc_first(pipeline):
    pipeline.ingest("ik.txt", HR.encode())
    pipeline.ingest("satinalma.txt", FIN.encode())
    hits = pipeline.retrieve("yıllık izin kaç gün")
    assert hits[0].source == "ik.txt"
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_file_filter(pipeline):
    pipeline.ingest("ik.txt", HR.encode())
    pipeline.ingest("satinalma.txt", FIN.encode())
    fin_hash = next(d.file_hash for d in pipeline.store.list_documents() if d.filename == "satinalma.txt")
    hits = pipeline.retrieve("yıllık izin", file_hashes=[fin_hash])
    assert all(h.source == "satinalma.txt" for h in hits)


def test_ask_includes_numbered_context_and_history(pipeline):
    pipeline.ingest("ik.txt", HR.encode())
    answer = pipeline.ask("Yıllık izin kaç gün?", history=[("user", "merhaba"), ("assistant", "selam")])
    assert answer.has_context
    assert "[1]" in answer.answer
    _, messages = pipeline.llm.calls[-1]
    assert messages[0] == ("user", "merhaba")
    assert "[1] Kaynak: ik.txt" in messages[-1][1]


def test_ask_without_documents_returns_fallback(pipeline):
    answer = pipeline.ask("Herhangi bir soru?")
    assert answer.answer == NO_CONTEXT_ANSWER
    assert pipeline.llm.calls == []


def test_delete_document(pipeline):
    pipeline.ingest("ik.txt", HR.encode())
    doc = pipeline.store.list_documents()[0]
    pipeline.delete_document(doc.file_hash)
    assert pipeline.store.count() == 0
    assert not any(pipeline.settings.upload_dir.iterdir())


def test_cited_indices():
    assert cited_indices("A [2] B [1][2] C [9]", 3) == [1, 2]
