"""Multi-Agent orkestratör ve POST /api/v1/agent/query — gerçek PostgreSQL (PGlite) üzerinde."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.schemas.agent import Intent
from app.services.orchestrator_service import _template_report
from tests.api.conftest import SAMPLE_DIR, create_tenant, upload

HYBRID_Q = "Harcama onay prosedürüne göre 250.000 TL üzeri harcamayı kim onaylar ve bu tutarı aşan siparişler hangileri?"
BIG_ORDERS_SQL = {
    "sql": "SELECT o.order_no, c.name AS musteri, o.total_amount AS tutar FROM orders o "
           "JOIN customers c ON c.id = o.customer_id WHERE o.total_amount > 250000 AND o.status <> 'iptal' "
           "ORDER BY o.total_amount DESC LIMIT 20",
    "explanation": "250.000 TL üzerindeki iptal edilmemiş siparişler.",
}


def h(t: dict) -> dict:
    return {"X-API-Key": t["api_key"]}


def agent(client, tenant, question, **kw):
    r = client.post("/api/v1/agent/query", json={"question": question, **kw}, headers=h(tenant))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def sql_llm(llm) -> Iterator[list]:
    queue: list[dict] = []
    llm.json_handlers["SQLGeneration"] = lambda p: queue.pop(0) if len(queue) > 1 else queue[0]
    yield queue
    llm.json_handlers.pop("SQLGeneration", None)


@pytest.fixture
def clean_handlers(llm) -> Iterator:
    yield
    for name in ("_IntentDecision", "_ReportDraft"):
        llm.json_handlers.pop(name, None)


@pytest.fixture(scope="module")
def tenant_a(client) -> dict:
    """Bu dosyaya özel tenant: ortak `tenant_a`'nın demo verisini üretmemek için (başka testler `created=True` bekler)."""
    t = create_tenant(client, "agent-holding", "Agent Holding A.Ş.")
    for name, dept in [
        ("IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları"),
        ("Finans_Harcama_Proseduru.md", "Finans"),
    ]:
        assert upload(client, t["api_key"], name, (SAMPLE_DIR / name).read_bytes(), dept).status_code == 201
    return t


@pytest.fixture(scope="module")
def sales_a(client, tenant_a) -> None:
    assert client.post("/api/v1/analytics/demo-data", headers=h(tenant_a)).status_code == 200


def sql_calls(llm) -> int:
    return len([n for n, _ in llm.json_calls if n == "SQLGeneration"])


# ------------------------------------------------------------------ niyetler
def test_document_only_intent(client, tenant_a, llm):
    before = sql_calls(llm)
    body = agent(client, tenant_a, "Yıllık izin politikası nedir?", departments=["İnsan Kaynakları"])
    assert body["intent"] == "DOCUMENT_ONLY" and body["intent_method"] == "heuristic"
    assert body["sql"] is None and body["data"] is None
    assert body["rag_sources"] and body["rag_sources"][0]["label"].startswith("IK_Izin_ve_Calisma_Yonetmeligi.md - Sayfa")
    assert body["is_answered"] and 0 < body["confidence"] <= 100
    assert body["report"]["synthesis_method"] == "template" and body["report"]["key_findings"]
    assert sql_calls(llm) == before  # SQL ajanı çalışmadı
    assert "documents_ms" in body["timings_ms"] and "data_ms" not in body["timings_ms"]


def test_sql_only_intent_forced_by_client(client, tenant_a, sales_a, sql_llm, llm):
    sql_llm.append({"sql": "SELECT COUNT(*) AS urun_sayisi FROM products", "explanation": "Ürün sayısı."})
    body = agent(client, tenant_a, "Katalogda kaç ürün var?", intent="SQL_ONLY")
    assert body["intent"] == "SQL_ONLY" and body["intent_method"] == "client"
    assert body["rag_sources"] == []
    assert body["sql"]["success"] and "products" in body["sql"]["tables"] and body["sql"]["row_count"] == 1
    assert body["data"]["rows"][0][0] > 0
    assert "data_ms" in body["timings_ms"] and "documents_ms" not in body["timings_ms"]


def test_hybrid_intent_runs_both_agents(client, tenant_a, sales_a, sql_llm):
    sql_llm.append(BIG_ORDERS_SQL)
    body = agent(client, tenant_a, HYBRID_Q, departments=["Finans"])
    assert body["intent"] == "HYBRID" and body["intent_method"] == "heuristic"
    assert body["rag_sources"] and body["sql"]["success"]
    assert all(r[2] > 250000 for r in body["data"]["rows"])
    assert {"intent_ms", "documents_ms", "data_ms", "synthesis_ms", "total_ms"} <= set(body["timings_ms"])
    rep = body["report"]
    assert any(f.startswith("Doküman:") for f in rep["key_findings"]) and any(f.startswith("Veri:") for f in rep["key_findings"])
    assert rep["caveats"] == []


def test_agents_run_in_parallel(client, container, tenant_a, sales_a, sql_llm, monkeypatch):
    """Toplam süre, iki kolun süre toplamından kısa olmalı (asyncio.gather)."""
    import asyncio

    orch = container.orchestrator
    real_ask, real_run = orch.rag.ask, orch.sql.run

    async def slow_ask(*a, **kw):
        await asyncio.sleep(0.4)
        return await real_ask(*a, **kw)

    async def slow_run(*a, **kw):
        await asyncio.sleep(0.4)
        return await real_run(*a, **kw)

    monkeypatch.setattr(orch.rag, "ask", slow_ask)
    monkeypatch.setattr(orch.sql, "run", slow_run)
    sql_llm.append(BIG_ORDERS_SQL)
    t = agent(client, tenant_a, HYBRID_Q, departments=["Finans"])["timings_ms"]
    assert t["documents_ms"] >= 400 and t["data_ms"] >= 400
    assert t["documents_ms"] + t["data_ms"] - t["total_ms"] >= 300  # süreler üst üste biniyor


# ------------------------------------------------------------------ LLM yolları
def test_llm_intent_and_synthesis_are_used(client, tenant_a, sales_a, sql_llm, llm, clean_handlers):
    llm.json_handlers["_IntentDecision"] = lambda p: {"intent": "SQL_ONLY", "reason": "Sayısal soru."}
    llm.json_handlers["_ReportDraft"] = lambda p: {
        "title": "Ürün Kataloğu Raporu",
        "executive_summary": "Katalog dolu.",
        "key_findings": ["Ürünler mevcut."],
        "recommendations": ["Stok takibi yapın."],
        "caveats": ["Örnek uyarı."],
    }
    sql_llm.append({"sql": "SELECT COUNT(*) AS urun_sayisi FROM products", "explanation": "x"})
    body = agent(client, tenant_a, "Katalogda kaç ürün var?")
    assert body["intent"] == "SQL_ONLY" and body["intent_method"] == "llm"
    assert body["report"]["synthesis_method"] == "llm" and body["report"]["title"] == "Ürün Kataloğu Raporu"
    assert body["report"]["caveats"] == ["Örnek uyarı."]
    prompt = next(p for n, p in reversed(llm.json_calls) if n == "_ReportDraft")
    assert "SQL AJANI ÇIKTISI" in prompt and "DOKÜMAN AJANI" not in prompt


# ------------------------------------------------------------------ yedek (fallback) senaryoları
def test_intent_falls_back_to_heuristic_when_llm_fails(client, tenant_a, llm, clean_handlers):
    def boom(prompt):
        from src.llm import LLMError

        raise LLMError("kota")

    llm.json_handlers["_IntentDecision"] = boom
    body = agent(client, tenant_a, "Yıllık izin politikası nedir?", departments=["İnsan Kaynakları"])
    assert body["intent_method"] == "heuristic" and body["intent"] == "DOCUMENT_ONLY"


def test_synthesis_falls_back_to_template_when_llm_fails(client, tenant_a, llm, clean_handlers):
    def boom(prompt):
        raise ValueError("geçersiz yapısal çıktı")

    llm.json_handlers["_ReportDraft"] = boom
    body = agent(client, tenant_a, "Yıllık izin politikası nedir?", departments=["İnsan Kaynakları"])
    assert body["is_answered"] and body["report"]["synthesis_method"] == "template"


def test_sql_failure_does_not_drop_document_findings(client, tenant_a, sales_a, sql_llm):
    sql_llm.append({"sql": "SELECT * FROM tenants", "explanation": "x"})  # sürekli reddedilir
    body = agent(client, tenant_a, HYBRID_Q, departments=["Finans"])
    assert body["intent"] == "HYBRID"
    assert body["sql"]["success"] is False and body["sql"]["attempts"] == 3 and body["sql"]["error"]
    assert body["data"] is None and body["rag_sources"]
    assert body["is_answered"]
    assert any("Veri ajanı" in c for c in body["report"]["caveats"])


def test_agent_exception_is_isolated(client, container, tenant_a, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("veritabanı koptu")

    monkeypatch.setattr(container.orchestrator.sql, "run", broken)
    body = agent(client, tenant_a, HYBRID_Q, departments=["Finans"])
    assert body["is_answered"] and body["rag_sources"] and body["sql"] is None
    assert any("RuntimeError" in w for w in body["warnings"])
    assert any("Veri ajanı başarısız" in c for c in body["report"]["caveats"])


def test_nothing_found_returns_honest_empty_report(client, container, tenant_a, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr(container.orchestrator.rag, "ask", broken)
    monkeypatch.setattr(container.orchestrator.sql, "run", broken)
    body = agent(client, tenant_a, HYBRID_Q)
    assert not body["is_answered"] and body["confidence"] == 0
    assert "yeterli bilgi bulunamadı" in body["report"]["executive_summary"]
    assert body["report"]["key_findings"] == [] and len(body["report"]["caveats"]) == 2


# ------------------------------------------------------------------ güvenlik / doğrulama
def test_agent_requires_api_key(client):
    assert client.post("/api/v1/agent/query", json={"question": "kaç ürün var"}).status_code == 401


def test_agent_validates_input(client, tenant_a):
    assert client.post("/api/v1/agent/query", json={"question": "x"}, headers=h(tenant_a)).status_code == 422
    r = client.post("/api/v1/agent/query", json={"question": "geçerli soru", "intent": "NOPE"}, headers=h(tenant_a))
    assert r.status_code == 422
    r = client.post(
        "/api/v1/agent/query", json={"question": "geçerli soru", "departments": ["Olmayan"]}, headers=h(tenant_a)
    )
    assert r.status_code == 422


def test_agent_is_tenant_isolated(client, tenant_a, tenant_b, sales_a):
    body = agent(client, tenant_b, "Yıllık izin politikası nedir?")
    assert all("IK_Izin" not in s["source"] for s in body["rag_sources"])
    assert all(s["source"] == "Rakip_IK.md" for s in body["rag_sources"])


# ------------------------------------------------------------------ birim testleri
def test_template_report_without_findings():
    rep = _template_report("Soru?", None, None, ["Kaynak yok."])
    assert rep["key_findings"] == [] and rep["caveats"] == ["Kaynak yok."]
    assert rep["synthesis_method"] == "template"


def test_intent_enum_values():
    assert {i.value for i in Intent} == {"DOCUMENT_ONLY", "SQL_ONLY", "HYBRID"}
