"""Streamlit arayüzü: tenant seçici, tenant bazlı sohbet ve 3 sekmenin duman testleri.

Arayüz `streamlit.testing.AppTest` ile gerçekten çalıştırılır. `httpx.request` yönlendirilir:
`/agent/query` çağrıları gerçek FastAPI uç noktasına (sahte orkestratörle, `tests/agent`)
gider; böylece arayüz API'nin gerçek yanıt şemasıyla çizilir. Ağ, DB ve LLM kullanılmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.agent.conftest import start_agent_app  # noqa: E402
from tests.agent.fakes import KEY_A, KEY_B, TENANT_A, TENANT_B  # noqa: E402
from ui.streamlit_app import MANUAL_TENANT, load_tenant_profiles, tenant_messages  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")
SECRETS = {
    "api_url": "http://api.test",
    "tenants": {
        "ege-vadi": {"name": "Ege Vadi Gıda", "api_key": KEY_A, "departments": list(TENANT_A.departments)},
        "kuzey-ege": {"name": "Kuzey Ege Yağ", "api_key": KEY_B, "departments": list(TENANT_B.departments)},
    },
}
DOCS = {
    "items": [{
        "id": "7d7c6b2e-0000-4000-8000-000000000001", "department": "Finans", "filename": "Iskonto_Politikasi.pdf",
        "file_type": "pdf", "size_bytes": 20480, "status": "ready", "error": None, "page_count": 3,
        "chunk_count": 12, "created_at": "2026-11-20T10:00:00Z", "updated_at": "2026-11-20T10:00:05Z",
    }],
    "total": 1, "limit": 200, "offset": 0,
}
ANALYTICS = {
    "question": "q", "route": "data", "route_reason": "x", "route_method": "client",
    "answer": "📊 **Veri analizi**\n3 müşteri listelendi.", "is_answered": True, "confidence": 95, "sources": [],
    "documents": None, "timings_ms": {}, "warnings": [],
    "data": {
        "success": True, "sql": "SELECT name, SUM(total_amount) AS ciro FROM orders GROUP BY name",
        "explanation": "Müşteri cirosu", "summary": "", "columns": ["musteri", "ciro"],
        "rows": [["Körfez Market", 1025466.42], ["Efes Gıda", 566819.5], ["Torbalı Toptan", 587940.0]],
        "row_count": 3, "truncated": False, "tables": ["orders"], "numeric_stats": {}, "insights": [],
        "chart": None, "attempts": [{"sql": "SELECT 1", "error": None}], "timings_ms": {}, "warnings": [],
    },
}


# ------------------------------------------------------------------ saf yardımcılar
def test_load_tenant_profiles_skips_entries_without_key():
    profiles = load_tenant_profiles({
        "ege-vadi": {"name": "Ege", "api_key": " rag_a ", "departments": ["Finans"]},
        "anahtarsiz": {"name": "Eksik"},
        "kuzey": {"api_key": "rag_b"},
    })
    assert list(profiles) == ["ege-vadi", "kuzey"]
    assert profiles["ege-vadi"] == {"name": "Ege", "api_key": "rag_a", "departments": ["Finans"]}
    assert profiles["kuzey"] == {"name": "kuzey", "api_key": "rag_b", "departments": []}
    assert load_tenant_profiles(None) == {} and MANUAL_TENANT not in profiles


def test_tenant_messages_are_isolated_per_tenant():
    state: dict = {}
    tenant_messages(state, "a").append({"role": "user", "content": "A'nın sorusu"})
    assert tenant_messages(state, "b") == []
    assert tenant_messages(state, "a") == [{"role": "user", "content": "A'nın sorusu"}]


# ------------------------------------------------------------------ arayüz (AppTest)
@pytest.fixture
def api(monkeypatch):  # noqa: ANN001, ANN201
    app = start_agent_app()
    seen: list[dict[str, Any]] = []

    def route(method, url, headers=None, timeout=None, **kw):  # noqa: ANN001, ANN202
        path = httpx.URL(url).path
        seen.append({"method": method, "path": path, "headers": dict(headers or {}), **kw})
        if path == "/api/v1/agent/query":
            return app.client.request(method, path, headers=headers, **kw)
        if path == "/api/v1/documents" and method == "GET":
            return httpx.Response(200, json=DOCS, request=httpx.Request(method, url))
        if path == "/api/v1/analytics/query":
            return httpx.Response(200, json=ANALYTICS, request=httpx.Request(method, url))
        return httpx.Response(404, json={"detail": "yok"}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", route)
    yield app, seen
    app.client.__exit__(None, None, None)


def new_app(secrets: dict | None = SECRETS) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in (secrets or {}).items():
        at.secrets[k] = v
    return at.run()


def agent_calls(seen: list[dict]) -> list[dict]:
    return [s for s in seen if s["path"] == "/api/v1/agent/query"]


def test_without_profiles_and_key_shows_onboarding(api):
    at = new_app(secrets=None)
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].value == MANUAL_TENANT
    assert any("Başlamak için" in i.value for i in at.info)
    assert len(at.tabs) == 0


def test_tenant_profile_renders_three_tabs(api):
    _, seen = api
    at = new_app()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].value == "ege-vadi"
    assert [t.label for t in at.tabs] == ["🤖 Akıllı Kurumsal Asistan", "📁 Doküman Yönetimi", "📊 Canlı Veri Analitiği"]
    assert any("ege-vadi" in c.value for c in at.caption)
    docs_call = next(s for s in seen if s["path"] == "/api/v1/documents")
    assert docs_call["headers"] == {"X-API-Key": KEY_A}
    assert any(df.value["filename"].tolist() == ["Iskonto_Politikasi.pdf"] for df in at.dataframe)


def test_chat_shows_rag_sources_and_sql_in_separate_expanders(api):
    app, seen = api
    at = new_app()
    at.chat_input[0].set_value("İskonto tavanını aşan siparişler hangileri?").run()
    assert not at.exception, at.exception

    (call,) = agent_calls(seen)
    assert call["headers"] == {"X-API-Key": KEY_A}
    assert call["json"]["departments"] is None  # boş seçim = tenant'ın tüm departmanları
    assert app.container.orchestrator.calls[0]["tenant"] == TENANT_A
    labels = [e.label for e in at.expander]
    assert "📄 Doküman kaynakları (1)" in labels
    assert "🗄️ SQL özeti" in labels
    assert any("SELECT order_no" in c.value for c in at.code)
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_switching_tenant_isolates_chat_history(api):
    app, seen = api
    at = new_app()
    at.chat_input[0].set_value("A şirketinin gizli sorusu").run()
    assert len(at.chat_message) == 2

    at.sidebar.selectbox[0].set_value("kuzey-ege").run()
    assert not at.exception, at.exception
    assert len(at.chat_message) == 0  # A'nın sohbeti B ekranında görünmez

    at.chat_input[0].set_value("B şirketinin sorusu").run()
    b_call = agent_calls(seen)[-1]
    assert b_call["headers"] == {"X-API-Key": KEY_B}
    assert b_call["json"]["history"] == []  # A'nın geçmişi B'nin API çağrısına gitmez
    assert app.container.orchestrator.calls[-1]["tenant"] == TENANT_B
    # Tenant B'nin kendi departmanları (Genel, Üretim) varsayılan seçimle 422'ye düşmez
    assert app.container.orchestrator.calls[-1]["departments"] == list(TENANT_B.departments)

    at.sidebar.selectbox[0].set_value("ege-vadi").run()
    assert len(at.chat_message) == 2


def test_department_selection_is_sent(api):
    app, seen = api
    at = new_app()
    at.sidebar.multiselect[0].set_value(["Finans"])
    at.chat_input[0].set_value("Finans sorusu").run()
    assert agent_calls(seen)[0]["json"]["departments"] == ["Finans"]
    assert app.container.orchestrator.calls[0]["departments"] == ["Finans"]


def test_analytics_tab_renders_table_and_chart_builder(api):
    _, seen = api
    at = new_app()
    next(t for t in at.text_input if t.label == "Soru").input("Müşteri bazında ciro?")
    next(b for b in at.button if b.label == "Sorgula").click().run()
    assert not at.exception, at.exception
    call = next(s for s in seen if s["path"] == "/api/v1/analytics/query")
    assert call["json"]["question"] == "Müşteri bazında ciro?" and call["json"]["mode"] == "data"
    assert any(df.value.shape == (3, 2) for df in at.dataframe)
    assert "🎛️ Grafiği özelleştir" in [e.label for e in at.expander]
    assert any(s.label == "Y ekseni" and s.value == "ciro" for s in at.selectbox)
