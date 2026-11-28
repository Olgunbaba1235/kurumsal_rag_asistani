""""Bunu mu sormak istediniz?" — sorgu öneri mantığı (birim) ve vitrin arayüzü (AppTest).

Öneri ana aramadan önce `fast_model` ile üretilir; soru zaten netse gereksiz uyarı çıkmaz, model hata verirse
arama hiçbir koşulda engellenmez.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.llm import LLMError
from src.local_engine import LocalWorkspace
from src.rag_engine import NO_SUGGESTION, QuerySuggestion, _QuerySuggestionOut, suggest_query

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SAMPLE = "Örnek Holding A.Ş."
TYPO = "fazla mesay ücreti nasıl hesplanır"
FIXED = "Fazla mesai ücreti nasıl hesaplanır?"


class FakeSuggestLLM:
    """Yalnızca `generate_json` çağrılarını kaydeden sahte model; `reply` dict veya istisna olabilir."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        self.calls.append({"prompt": prompt, "schema": schema, "system": system, "fast": fast})
        if isinstance(self.reply, Exception):
            raise self.reply
        return schema.model_validate(self.reply)

    def stream(self, messages, **_):
        yield "kullanılmamalı"


# ------------------------------------------------------------------ suggest_query (birim)
def test_typo_question_gets_clear_suggestion_from_fast_model():
    llm = FakeSuggestLLM({"has_suggestion": True, "suggested_query": FIXED})
    out = suggest_query(TYPO, llm)
    assert out == QuerySuggestion(True, FIXED)
    assert out.to_dict() == {"has_suggestion": True, "suggested_query": FIXED}
    call = llm.calls[0]
    assert call["fast"] is True and call["schema"] is _QuerySuggestionOut and TYPO in call["prompt"]
    assert "YANITLAMA" in call["system"] and "EKLEME" in call["system"]


def test_clear_question_returns_no_suggestion():
    llm = FakeSuggestLLM({"has_suggestion": False, "suggested_query": ""})
    out = suggest_query(FIXED, llm)
    assert out == NO_SUGGESTION and out.to_dict() == {"has_suggestion": False, "suggested_query": ""}


@pytest.mark.parametrize("suggested", [
    "fazla mesai ücreti nasıl hesaplanır",      # yalnızca büyük/küçük harf ve noktalama farkı
    "Fazla mesai ucreti nasil hesaplanir?",     # yalnızca Türkçe karakter farkı
    "   ",                                      # boş öneri
    "x" * 301,                                  # makul olmayan uzunluk
])
def test_trivial_or_invalid_suggestions_are_suppressed(suggested):
    llm = FakeSuggestLLM({"has_suggestion": True, "suggested_query": suggested})
    assert suggest_query(FIXED, llm) == NO_SUGGESTION


def test_suggestion_text_is_normalized():
    llm = FakeSuggestLLM({"has_suggestion": True, "suggested_query": "  Yıllık izin\n kaç   gündür? "})
    assert suggest_query("yılık izn kaç", llm).suggested_query == "Yıllık izin kaç gündür?"


@pytest.mark.parametrize("error", [LLMError("upstream 503", code=503), ValueError("bozuk JSON"), RuntimeError("x")])
def test_model_errors_never_block_search(error):
    assert suggest_query(TYPO, FakeSuggestLLM(error)) == NO_SUGGESTION


def test_without_llm_or_empty_question_no_model_call():
    llm = FakeSuggestLLM({"has_suggestion": True, "suggested_query": FIXED})
    assert suggest_query(TYPO, None) == NO_SUGGESTION
    assert suggest_query("   ", llm) == NO_SUGGESTION and not llm.calls


# ------------------------------------------------------------------ motor / çalışma alanı
def _ws(llm, *, enabled: bool = True) -> LocalWorkspace:
    ws = LocalWorkspace(Settings(_env_file=None), api_key=None)
    ws.engine.llm = llm
    ws.engine.settings = ws.engine.settings.model_copy(update={"query_suggestions": enabled})
    return ws


def test_engine_suggest_respects_setting():
    llm = FakeSuggestLLM({"has_suggestion": True, "suggested_query": FIXED})
    assert _ws(llm).suggest(TYPO).suggested_query == FIXED
    off = FakeSuggestLLM({"has_suggestion": True, "suggested_query": FIXED})
    assert _ws(off, enabled=False).suggest(TYPO) == NO_SUGGESTION and not off.calls


def test_offline_workspace_does_not_call_model_for_suggestions():
    ws = LocalWorkspace(Settings(_env_file=None), api_key=None)
    assert ws.settings.query_suggestions is False and ws.suggest(TYPO) == NO_SUGGESTION


# ------------------------------------------------------------------ arayüz (AppTest)
@pytest.fixture
def app_test(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        import streamlit.dataframe_util as du

        monkeypatch.setattr(du, "convert_anything_to_arrow_bytes", lambda *a, **k: b"")
    at = st_testing.AppTest.from_file(APP, default_timeout=90)
    at.session_state["stream_delay"] = 0.0
    at.session_state["demo_delay"] = 0.0
    return at.run()


def _infos(at) -> list[str]:
    return [i.value for i in at.info]


def test_ui_shows_suggestion_and_reasks_with_one_click(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    asked: list[str] = []
    monkeypatch.setattr(ws, "suggest", lambda q: asked.append(q) or (
        QuerySuggestion(True, FIXED) if q == TYPO else NO_SUGGESTION))

    at.chat_input[0].set_value(TYPO).run()
    assert not at.exception, at.exception
    boxes = [t for t in _infos(at) if "Bunu mu sormak istediniz" in t]
    assert len(boxes) == 1 and "Fazla mesai ücreti nasıl hesaplanır" in boxes[0] and boxes[0].endswith("?")
    assert asked == [TYPO]
    chat = at.session_state["chats"][SAMPLE]
    assert chat[-1]["suggestion"].has_suggestion and chat[-1]["response"] is not None

    button = next(b for b in at.button if b.key and b.key.startswith("suggest_"))
    button.click().run()
    assert not at.exception, at.exception
    chat = at.session_state["chats"][SAMPLE]
    assert [m["content"] for m in chat if m["role"] == "user"] == [TYPO, FIXED]
    assert chat[-1]["response"].is_answered and not chat[-1]["suggestion"].has_suggestion
    assert asked == [TYPO, FIXED]


def test_ui_clear_question_shows_no_suggestion_box(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    monkeypatch.setattr(ws, "suggest", lambda q: NO_SUGGESTION)
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    assert not [t for t in _infos(at) if "Bunu mu sormak istediniz" in t]
    assert not [b for b in at.button if b.key and b.key.startswith("suggest_")]
