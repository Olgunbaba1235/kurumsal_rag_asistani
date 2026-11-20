"""Streamlit arayüzünün saf (Streamlit'ten bağımsız) yardımcıları ve API istemcisi."""

from __future__ import annotations

import httpx
import pytest

from ui.streamlit_app import ApiClient, ApiError, history_payload, rows_to_df


def test_history_payload_uses_summary_for_assistant_turns():
    msgs = [
        {"role": "user", "content": "Soru 1"},
        {"role": "assistant", "content": "x", "report": {"executive_summary": "Özet 1"}},
        {"role": "assistant", "content": "y", "report": {"executive_summary": ""}},  # boş → atlanır
    ]
    assert history_payload(msgs) == [
        {"role": "user", "content": "Soru 1"},
        {"role": "assistant", "content": "Özet 1"},
    ]


def test_history_payload_is_limited():
    msgs = [{"role": "user", "content": f"s{i}"} for i in range(30)]
    assert len(history_payload(msgs, limit=10)) == 10


def test_rows_to_df():
    df = rows_to_df(["a", "b"], [[1, "x"], [2, "y"]])
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_api_client_sends_key_and_parses_errors(monkeypatch):
    seen = {}

    def fake(method, url, headers=None, timeout=None, **kw):
        seen.update(method=method, url=url, headers=headers, json=kw.get("json"))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake)
    api = ApiClient("http://x:8000/", "KEY")
    assert api.agent_query("soru?", ["Finans"], [], "HYBRID") == {"ok": True}
    assert seen["url"] == "http://x:8000/api/v1/agent/query" and seen["headers"] == {"X-API-Key": "KEY"}
    assert seen["json"] == {"question": "soru?", "departments": ["Finans"], "history": [], "intent": "HYBRID"}

    monkeypatch.setattr(
        httpx, "request",
        lambda m, u, **kw: httpx.Response(401, json={"detail": "Geçersiz anahtar"}, request=httpx.Request(m, u)),
    )
    with pytest.raises(ApiError, match="401.*Geçersiz anahtar"):
        api.analytics_query("q", [], "auto")

    def refuse(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", refuse)
    with pytest.raises(ApiError, match="ulaşılamadı"):
        api.list_documents()
