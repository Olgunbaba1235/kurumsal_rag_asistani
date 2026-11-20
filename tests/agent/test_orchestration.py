"""`MultiAgentOrchestrator` birim testleri: niyet → doküman ajanı ∥ SQL ajanı → sentez.

Orkestratör gerçek koddur; uzman ajanlar (RAG, Text-to-SQL) ve LLM sahtedir. Veritabanı,
ağ ve zamanlama (sleep) kullanılmaz; eşzamanlılık olaylarla (asyncio.Event) doğrulanır.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.schemas.agent import Intent
from app.services.analytics_service import data_confidence
from app.services.orchestrator_service import MultiAgentOrchestrator
from tests.agent.fakes import TENANT_A, FakeLLM, FakeRAG, FakeSQL, make_rag_response, make_sql_answer

DEPTS = ["Finans"]
DRAFT = {
    "title": "Rapor",
    "executive_summary": "Özet",
    "key_findings": ["Bulgu"],
    "recommendations": ["Öneri"],
    "caveats": [],
}


def orchestrator(rag=None, sql=None, llm=None) -> tuple[MultiAgentOrchestrator, FakeRAG, FakeSQL]:  # noqa: ANN001
    rag, sql = rag or FakeRAG(), sql or FakeSQL()
    return MultiAgentOrchestrator(rag, sql, llm), rag, sql


def run(orch, question="Soru?", departments=DEPTS, history=(), **kw):  # noqa: ANN001, ANN201
    return asyncio.run(orch.run(TENANT_A, question, departments, history, **kw))


# ------------------------------------------------------------------ niyet ajanı
def test_client_intent_skips_classifier():
    llm = FakeLLM(_IntentDecision={"intent": "DOCUMENT_ONLY", "reason": "x"})
    orch, rag, sql = orchestrator(llm=llm)
    r = run(orch, intent=Intent.SQL_ONLY)
    assert (r.intent, r.intent_method, r.intent_reason) == (Intent.SQL_ONLY, "client", "İstemci tarafından seçildi.")
    assert llm.prompts("_IntentDecision") == []
    assert rag.calls == [] and len(sql.calls) == 1


def test_llm_intent_decision_is_used():
    orch, rag, sql = orchestrator(llm=FakeLLM(_IntentDecision={"intent": "DOCUMENT_ONLY", "reason": "Politika"}))
    r = run(orch, "2026 cirosu kaç?")
    assert (r.intent, r.intent_method, r.intent_reason) == (Intent.DOCUMENT_ONLY, "llm", "Politika")
    assert len(rag.calls) == 1 and sql.calls == []


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Harcama prosedürüne göre avans nasıl alınır?", Intent.DOCUMENT_ONLY),
        ("2026 yılında toplam ciro nedir?", Intent.SQL_ONLY),
        ("Onay limitini aşan siparişler hangileri?", Intent.HYBRID),
    ],
)
def test_llm_failure_falls_back_to_heuristic_intent(question, intent):
    orch, _, _ = orchestrator(llm=FakeLLM())  # tanımsız şema → LLMError
    r = run(orch, question)
    assert (r.intent, r.intent_method) == (intent, "heuristic")


def test_without_llm_uses_heuristic_and_template_report():
    orch, _, _ = orchestrator(llm=None)
    r = run(orch, "Onay limitini aşan siparişler hangileri?")
    assert (r.intent, r.intent_method) == (Intent.HYBRID, "heuristic")
    assert r.report["synthesis_method"] == "template"


def test_classifier_prompt_contains_only_recent_history():
    llm = FakeLLM(_IntentDecision={"intent": "SQL_ONLY", "reason": "x"})
    orch, _, _ = orchestrator(llm=llm)
    history = [("user", f"eski-{i}") for i in range(6)] + [("assistant", "son-yanit")]
    run(orch, "Peki geçen ay?", history=history)
    (prompt,) = llm.prompts("_IntentDecision")
    assert "GEÇMİŞ" in prompt and "son-yanit" in prompt and "SORU: Peki geçen ay?" in prompt
    assert "eski-0" not in prompt and "eski-2" not in prompt  # yalnızca son 4 tur


# ------------------------------------------------------------------ paralel uzman ajanlar
def test_document_and_sql_agents_run_concurrently():
    """Her ajan diğerinin başlamasını bekler; sıralı çalışsaydı ikisi de zaman aşımına düşerdi."""
    docs_started, data_started = asyncio.Event(), asyncio.Event()

    async def docs_hook():
        docs_started.set()
        await asyncio.wait_for(data_started.wait(), timeout=2)

    async def data_hook():
        data_started.set()
        await asyncio.wait_for(docs_started.wait(), timeout=2)

    orch, _, _ = orchestrator(rag=FakeRAG(before=docs_hook), sql=FakeSQL(before=data_hook))
    r = run(orch, intent=Intent.HYBRID)
    assert r.warnings == []
    assert r.documents is not None and r.data is not None


@pytest.mark.parametrize(
    ("intent", "docs", "data"),
    [(Intent.DOCUMENT_ONLY, 1, 0), (Intent.SQL_ONLY, 0, 1), (Intent.HYBRID, 1, 1)],
)
def test_only_agents_required_by_intent_are_invoked(intent, docs, data):
    orch, rag, sql = orchestrator()
    run(orch, intent=intent)
    assert (len(rag.calls), len(sql.calls)) == (docs, data)


def test_tenant_and_context_reach_every_agent_unchanged():
    orch, rag, sql = orchestrator()
    history = [("user", "önceki soru"), ("assistant", "önceki yanıt")]
    run(orch, "Soru?", ["Finans", "Satış"], history, intent=Intent.HYBRID, top_k=7)
    assert rag.calls == [{"tenant": TENANT_A, "question": "Soru?", "departments": ["Finans", "Satış"],
                          "history": history, "top_k": 7}]
    assert sql.calls == [{"tenant": TENANT_A, "question": "Soru?", "history": history}]


# ------------------------------------------------------------------ hata izolasyonu
def test_document_agent_crash_keeps_sql_result():
    orch, _, _ = orchestrator(rag=FakeRAG(error=RuntimeError("vektör DB düştü")))
    r = run(orch, intent=Intent.HYBRID)
    assert r.documents is None and r.data is not None
    assert r.warnings == ["Doküman ajanı başarısız oldu (RuntimeError)."]
    assert r.report["caveats"] == ["Doküman ajanı başarısız oldu (RuntimeError)."]
    assert r.is_answered is True and r.confidence == data_confidence(r.data)


def test_sql_agent_crash_keeps_document_result():
    orch, _, _ = orchestrator(sql=FakeSQL(error=TimeoutError()))
    r = run(orch, intent=Intent.HYBRID)
    assert r.data is None and r.documents is not None
    assert r.warnings == ["Veri ajanı başarısız oldu (TimeoutError)."]
    assert r.confidence == 82
    assert any(f.startswith("Doküman:") for f in r.report["key_findings"])


def test_error_details_are_not_exposed():
    orch, _, _ = orchestrator(sql=FakeSQL(error=RuntimeError("password=s3cr3t")))
    r = run(orch, intent=Intent.SQL_ONLY)
    assert "s3cr3t" not in json.dumps([r.warnings, r.report], ensure_ascii=False)


def test_all_agents_failing_returns_honest_empty_report_without_llm_synthesis():
    llm = FakeLLM(_ReportDraft=DRAFT)
    orch, _, _ = orchestrator(rag=FakeRAG(error=RuntimeError()), sql=FakeSQL(error=RuntimeError()), llm=llm)
    r = run(orch, intent=Intent.HYBRID)
    assert (r.is_answered, r.confidence) == (False, 0)
    assert r.report["synthesis_method"] == "template" and r.report["key_findings"] == []
    assert "yeterli bilgi bulunamadı" in r.report["executive_summary"]
    assert len(r.report["caveats"]) == 2
    assert llm.prompts("_ReportDraft") == []  # bulgu yokken LLM'e rapor yazdırılmaz


def test_unanswered_documents_and_failed_sql_become_caveats():
    orch, _, _ = orchestrator(
        rag=FakeRAG(response=make_rag_response(answered=False)),
        sql=FakeSQL(answer=make_sql_answer(success=False)),
    )
    r = run(orch, intent=Intent.HYBRID)
    assert r.report["caveats"] == [
        "Dokümanlarda soruyla ilgili yeterli bilgi bulunamadı.",
        "Veri ajanı geçerli bir sorgu sonucu üretemedi.",
    ]
    assert r.is_answered is False


def test_agent_warnings_are_prefixed_by_source():
    orch, _, _ = orchestrator(
        rag=FakeRAG(response=make_rag_response(warnings=["düşük alaka"])),
        sql=FakeSQL(answer=make_sql_answer(warnings=["sonuç kısaltıldı"])),
    )
    r = run(orch, intent=Intent.HYBRID)
    assert r.warnings == ["[doküman] düşük alaka", "[veri] sonuç kısaltıldı"]


# ------------------------------------------------------------------ sentez ajanı
def test_llm_synthesis_receives_both_agent_outputs_with_row_cap():
    rows = [[f"SO-{i:04d}", 13.0 + i / 100] for i in range(30)]
    llm = FakeLLM(_ReportDraft=DRAFT)
    orch, _, _ = orchestrator(sql=FakeSQL(answer=make_sql_answer(rows=rows)), llm=llm)
    r = run(orch, intent=Intent.HYBRID)
    assert r.report["synthesis_method"] == "llm" and r.report["title"] == "Rapor"

    (prompt,) = llm.prompts("_ReportDraft")
    assert "DOKÜMAN AJANI ÇIKTISI" in prompt and "[Iskonto_Politikasi.pdf - Sayfa 2]" in prompt
    payload = json.loads(prompt.split("SQL AJANI ÇIKTISI:\n", 1)[1].split("\n\n", 1)[0])
    assert len(payload["satirlar"]) == 20 and payload["toplam_satir"] == 30
    system = next(s for n, _, s in llm.calls if n == "_ReportDraft")
    assert "UYDURMA" in system


def test_llm_synthesis_merges_and_deduplicates_caveats():
    draft = {**DRAFT, "caveats": ["Doküman ajanı başarısız oldu (RuntimeError).", "Örneklem küçük."]}
    llm = FakeLLM(_ReportDraft=draft)
    orch, _, _ = orchestrator(rag=FakeRAG(error=RuntimeError()), llm=llm)
    r = run(orch, intent=Intent.HYBRID)
    assert r.report["caveats"] == ["Doküman ajanı başarısız oldu (RuntimeError).", "Örneklem küçük."]
    assert "BİLİNEN SORUNLAR" in llm.prompts("_ReportDraft")[0]


def test_llm_synthesis_failure_falls_back_to_template():
    orch, _, _ = orchestrator(llm=FakeLLM(_ReportDraft=ValueError("bozuk JSON")))
    r = run(orch, intent=Intent.HYBRID)
    assert r.report["synthesis_method"] == "template"
    assert r.is_answered is True and r.report["key_findings"]


def test_confidence_and_timings():
    orch, _, _ = orchestrator(rag=FakeRAG(response=make_rag_response(confidence=70)))
    r = run(orch, intent=Intent.HYBRID)
    assert r.confidence == round((70 + 95) / 2)
    assert {"intent_ms", "documents_ms", "data_ms", "synthesis_ms", "total_ms"} <= set(r.timings_ms)
    assert r.departments == DEPTS and r.question == "Soru?"
