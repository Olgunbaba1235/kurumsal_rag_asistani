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


# ------------------------------------------------------------------ kullanıcı girişi
SESSION = {
    "token": "rs_" + "a" * 32 + "." + "b" * 43, "expires_at": "2026-11-22T12:00:00Z",
    "email": "calisan@ornek.com", "full_name": "Ayşe Yılmaz",
    "me": {"via": "session", "company": {"id": "7d7c6b2e-0000-4000-8000-0000000000aa", "slug": "ornek-holding",
           "name": "Örnek Holding"}, "user_id": None, "membership_id": None, "role": "user", "is_owner": False,
           "departments": ["İK"]},
}


def test_user_login_uses_bearer_token_and_hides_analytics(api, monkeypatch):
    _, seen = api
    logins: list[dict] = []

    def fake_post(url, json=None, timeout=None, **kw):  # noqa: ANN001, ANN202, A002
        logins.append(json)
        return httpx.Response(200, json=SESSION, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    at = new_app(secrets=None)
    from ui.streamlit_app import LOGIN_TENANT

    at.sidebar.selectbox[0].set_value(LOGIN_TENANT).run()
    assert len(at.tabs) == 0  # giriş yapılmadan sekme yok
    inputs = {t.label: t for t in at.sidebar.text_input}
    inputs["Şirket kısa adı"].input("ornek-holding")
    inputs["E-posta"].input("calisan@ornek.com")
    inputs["Parola"].input("Guclu-Parola-2026!")
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert logins == [{"email": "calisan@ornek.com", "password": "Guclu-Parola-2026!", "company_slug": "ornek-holding"}]
    assert len(at.tabs) == 3
    docs_call = [s for s in seen if s["path"] == "/api/v1/documents"][-1]
    assert docs_call["headers"] == {"Authorization": f"Bearer {SESSION['token']}"}
    assert any("yalnızca şirket sahibi ve yöneticilere" in i.value for i in at.info)
    assert at.sidebar.multiselect[0].options == ["İK"]
    assert SESSION["token"] not in " ".join(m.value for m in at.markdown)  # belirteç ekrana yazılmaz


# ------------------------------------------------------------------ "Bunu mu sormak istediniz?" (/chat/query)
TYPO_Q = "yıllık izn kaç gün"
FIXED_Q = "Yıllık izin kaç gündür?"
INTENTS = ["Yıllık Ücretli İzin nelerdir?", "Hastalık İzni ile ilgili kurallar nelerdir?",
           "Mazeret İzni ile ilgili kurallar nelerdir?"]


def chat_response(question: str, suggestion: dict | None) -> dict:
    """Gerçek `ChatResponse` şemasından geçirilmiş /chat/query yanıtı (şema sapması testte yakalanır)."""
    from app.schemas.chat import ChatResponse

    body = {
        "answer": f"Yanıt: {question} [IK.md - Sayfa 1]", "is_answered": True, "confidence": 88,
        "citations": [{"label": "IK.md - Sayfa 1", "source": "IK.md", "page": 1, "department": "Finans",
                       "relevance": 91.0, "evidence": "Yıllık izin 14 gündür."}],
        "evidence": [], "rewritten_query": question, "search_queries": [question], "departments": ["Finans"],
        "reranker": "lexical", "timings_ms": {"total_ms": 12.0}, "total_ms": 12.0, "warnings": [],
        "suggestion": suggestion,
    }
    return ChatResponse.model_validate(body).model_dump(mode="json")


@pytest.fixture
def chat_api(api, monkeypatch):  # noqa: ANN001, ANN201
    """`api` yönlendirmesine /chat/query ekler; `suggestions[soru]` → API'nin döndüreceği öneri."""
    _, seen = api
    suggestions: dict[str, dict | None] = {}
    inner = httpx.request

    def route(method, url, headers=None, timeout=None, **kw):  # noqa: ANN001, ANN202
        path = httpx.URL(url).path
        if path == "/api/v1/chat/query":
            seen.append({"method": method, "path": path, "headers": dict(headers or {}), **kw})
            body = kw["json"]
            sug = suggestions.get(body["question"]) if body.get("suggest", True) else None
            return httpx.Response(200, json=chat_response(body["question"], sug), request=httpx.Request(method, url))
        return inner(method, url, headers=headers, timeout=timeout, **kw)

    monkeypatch.setattr(httpx, "request", route)
    return seen, suggestions


def chat_calls(seen: list[dict]) -> list[dict]:
    return [s for s in seen if s["path"] == "/api/v1/chat/query"]


def boxes(at) -> list[str]:  # noqa: ANN001
    return [i.value.replace("\\", "") for i in at.info if "Bunu mu sormak istediniz" in i.value]


def suggest_buttons(at) -> list:  # noqa: ANN001
    return [b for b in at.button if b.key and b.key.startswith("b2b_suggest_")]


def doc_only_app() -> AppTest:
    from ui.streamlit_app import INTENT_LABELS

    at = new_app()
    next(s for s in at.selectbox if s.label == "Niyet").set_value(INTENT_LABELS["DOCUMENT_ONLY"]).run()
    return at


def test_typo_suggestion_box_and_one_click_reask_without_loop(chat_api):
    seen, suggestions = chat_api
    suggestions[TYPO_Q] = {"has_suggestion": True, "suggested_query": FIXED_Q, "alternatives": [], "kind": "typo"}
    suggestions[FIXED_Q] = {"has_suggestion": True, "suggested_query": "döngü olmamalı", "alternatives": [],
                            "kind": "typo"}
    at = doc_only_app()
    at.chat_input[0].set_value(TYPO_Q).run()
    assert not at.exception, at.exception
    (call,) = chat_calls(seen)
    assert call["json"]["suggest"] is True and call["headers"] == {"X-API-Key": KEY_A}
    assert not agent_calls(seen)  # yalnızca doküman niyeti /chat/query ile yanıtlanır
    assert boxes(at) == ["Bunu mu sormak istediniz: **Yıllık izin kaç gündür**?"]
    assert any(f"Yanıt: {TYPO_Q}" in m.value for m in at.markdown)
    (button,) = suggest_buttons(at)
    assert button.label == "🔁 Önerilen soruyla sor"

    button.click().run()
    assert not at.exception, at.exception
    second = chat_calls(seen)[-1]
    assert second["json"]["question"] == FIXED_Q and second["json"]["suggest"] is False
    assert [h["content"] for h in second["json"]["history"] if h["role"] == "user"] == [TYPO_Q]
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert not suggest_buttons(at) and len(boxes(at)) == 1  # eski kutu butonsuz kalır, yeni kutu yok
    assert "döngü olmamalı" not in " ".join(boxes(at))


def test_ambiguous_question_offers_intent_buttons(chat_api):
    seen, suggestions = chat_api
    suggestions["izin"] = {"has_suggestion": True, "suggested_query": INTENTS[0], "alternatives": INTENTS,
                           "kind": "ambiguous"}
    at = doc_only_app()
    at.chat_input[0].set_value("izin").run()
    assert not at.exception, at.exception
    assert len(boxes(at)) == 1 and "birden fazla konuya" in boxes(at)[0]
    buttons = suggest_buttons(at)
    assert [b.label for b in buttons] == INTENTS
    buttons[1].click().run()
    assert not at.exception, at.exception
    assert chat_calls(seen)[-1]["json"]["question"] == INTENTS[1] and chat_calls(seen)[-1]["json"]["suggest"] is False
    assert not suggest_buttons(at)
    assert all(o in boxes(at)[0] for o in INTENTS)  # geçmişte butonsuz liste


def test_no_suggestion_means_no_box_and_sources_are_shown(chat_api):
    at = doc_only_app()
    at.chat_input[0].set_value("Yıllık izin kaç gündür?").run()
    assert not at.exception, at.exception
    assert not boxes(at) and not suggest_buttons(at)
    assert "📄 Doküman kaynakları (1)" in [e.label for e in at.expander]
    assert any("güven %88" in c.value for c in at.caption)


def test_suggestion_text_is_markdown_escaped(chat_api):
    _, suggestions = chat_api
    suggestions[TYPO_Q] = {"has_suggestion": True, "suggested_query": "**kalın** [x](http://y) <b>",
                           "alternatives": [], "kind": "typo"}
    at = doc_only_app()
    at.chat_input[0].set_value(TYPO_Q).run()
    raw = next(i.value for i in at.info if "Bunu mu" in i.value)
    assert "\\*\\*kalın\\*\\*" in raw and "\\[x\\]" in raw and "\\<b\\>" in raw


def test_automatic_intent_for_admin_still_uses_agent(chat_api):
    seen, _ = chat_api
    at = new_app()
    at.chat_input[0].set_value("İskonto tavanını aşan siparişler hangileri?").run()
    assert not at.exception, at.exception
    assert agent_calls(seen) and not chat_calls(seen)


def test_logged_in_user_without_analytics_gets_document_chat_with_suggestions(chat_api, monkeypatch):
    seen, suggestions = chat_api
    suggestions[TYPO_Q] = {"has_suggestion": True, "suggested_query": FIXED_Q, "alternatives": [], "kind": "typo"}
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(200, json=SESSION,
                                                                          request=httpx.Request("POST", url)))
    from ui.streamlit_app import LOGIN_TENANT

    at = new_app(secrets=None)
    at.sidebar.selectbox[0].set_value(LOGIN_TENANT).run()
    inputs = {t.label: t for t in at.sidebar.text_input}
    inputs["Şirket kısa adı"].input("ornek-holding")
    inputs["E-posta"].input("calisan@ornek.com")
    inputs["Parola"].input("x")
    at.sidebar.button[0].click().run()
    at.chat_input[0].set_value(TYPO_Q).run()
    assert not at.exception, at.exception
    (call,) = chat_calls(seen)
    assert call["headers"] == {"Authorization": f"Bearer {SESSION['token']}"} and not agent_calls(seen)
    assert len(boxes(at)) == 1 and len(suggest_buttons(at)) == 1


def test_suggestion_options_helper_is_defensive():
    from ui.streamlit_app import suggestion_options

    assert suggestion_options(None) == [] and suggestion_options({"has_suggestion": False}) == []
    assert suggestion_options({"has_suggestion": True, "suggested_query": " a ", "alternatives": []}) == ["a"]
    assert suggestion_options({"has_suggestion": True, "suggested_query": "a",
                               "alternatives": ["a", "b", "b", "c", "d", " "]}) == ["a", "b", "c"]
