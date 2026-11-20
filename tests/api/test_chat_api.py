from __future__ import annotations

import re

from src.rag_engine import NO_INFO_ANSWER


def h(t: dict) -> dict:
    return {"X-API-Key": t["api_key"]}


def ask(client, tenant, question, **kw):
    r = client.post("/api/v1/chat/query", json={"question": question, **kw}, headers=h(tenant))
    assert r.status_code == 200, r.text
    return r.json()


def test_grounded_answer_contract(client, tenant_a):
    body = ask(client, tenant_a, "250.000 TL üzerindeki harcamayı kim onaylar?", departments=["Finans"])
    assert body["is_answered"]
    assert re.search(r"\[Finans_Harcama_Proseduru\.md - Sayfa \d+\]", body["answer"])
    c = body["citations"][0]
    assert c["source"] == "Finans_Harcama_Proseduru.md" and c["department"] == "Finans" and c["evidence"]
    assert 0 < body["confidence"] <= 100
    assert {"rewrite_ms", "retrieval_ms", "embed_ms", "search_ms", "rerank_ms", "generation_ms", "total_ms"} <= set(
        body["timings_ms"]
    )
    assert body["total_ms"] > 0
    assert body["departments"] == ["Finans"]
    top = body["evidence"][0]
    assert "| 250.001 – 1.000.000 TL | CFO + Genel Müdür |" in top["text"]
    assert set(top["matched_by"]) == {"vector", "bm25"}


def test_department_filter_is_enforced(client, tenant_a):
    body = ask(client, tenant_a, "Yıllık izin süreleri", departments=["İnsan Kaynakları"])
    assert body["evidence"] and all(e["department"] == "İnsan Kaynakları" for e in body["evidence"])


def test_strict_grounding_refuses_unknown(client, tenant_a, llm):
    before = len(llm.calls)
    body = ask(client, tenant_a, "Şirket otoparkının aylık ücreti ne kadar?")
    assert body["answer"] == NO_INFO_ANSWER
    assert not body["is_answered"] and body["confidence"] == 0 and body["citations"] == []
    assert len(llm.calls) == before  # alaka kapısı: model hiç çağrılmadı


def test_out_of_department_question_refused(client, tenant_a):
    body = ask(client, tenant_a, "Masraf beyanı kaç gün içinde yapılmalı?", departments=["İnsan Kaynakları"])
    assert body["answer"] == NO_INFO_ANSWER


def test_hallucinated_citation_removed(client, tenant_a, llm):
    original = llm.answer_fn
    llm.answer_fn = lambda labels: f"Onay CFO'dadır [{labels[0]}]. Ek prim vardır [Uydurma.pdf - Sayfa 9]."
    try:
        body = ask(client, tenant_a, "Harcama onay yetkileri nelerdir?", departments=["Finans"])
    finally:
        llm.answer_fn = original
    assert "Uydurma.pdf" not in body["answer"]
    assert any("Uydurma.pdf" in w for w in body["warnings"])


def test_history_is_forwarded_to_model(client, tenant_a, llm):
    history = [
        {"role": "user", "content": "Harcama onay yetkileri nelerdir?"},
        {"role": "assistant", "content": "Tutara göre değişir."},
    ]
    body = ask(client, tenant_a, "Peki 1.000.000 TL üzeri?", departments=["Finans"], history=history)
    assert body["rewritten_query"].startswith("Harcama onay yetkileri nelerdir?")  # sezgisel yeniden yazma
    assert llm.calls[-1][0] == ("user", "Harcama onay yetkileri nelerdir?")


def test_invalid_department_and_validation(client, tenant_a):
    r = client.post("/api/v1/chat/query", json={"question": "izin", "departments": ["Pazarlama"]}, headers=h(tenant_a))
    assert r.status_code == 422
    r = client.post("/api/v1/chat/query", json={"question": ""}, headers=h(tenant_a))
    assert r.status_code == 422
