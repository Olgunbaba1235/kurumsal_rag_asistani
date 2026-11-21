"""Vitrin arayüzü: tema bileşenleri, grafikler ve uçtan uca Streamlit (AppTest) duman testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from ui import showcase_charts as charts
from ui import showcase_theme as theme

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SAMPLE = "Örnek Holding A.Ş."
OTHER = "Ege Vadi Gıda A.Ş."


# ------------------------------------------------------------------ tema
def test_citation_card_escapes_document_text():
    html = theme.citation_card("x.pdf - Sayfa 2", "<b>x.pdf</b>", 2, "Finans", 87.4, "<script>alert(1)</script> metin")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;b&gt;x.pdf&lt;/b&gt;" in html and "Sayfa 2" in html and "%87" in html


def test_citation_card_truncates_long_evidence_and_clamps_relevance():
    html = theme.citation_card("l", "s", None, "", 250, "kelime " * 400)
    assert "…" in html and "width:100%" in html and "Sayfa" not in html


def test_kpi_and_hero_escape():
    assert "&lt;i&gt;" in theme.kpi_card("x", "<i>", "1") and "&lt;i&gt;" in theme.hero("<i>", "s", [("b", "")])


# ------------------------------------------------------------------ grafikler
def test_sample_usage_is_deterministic():
    import datetime as dt

    day = dt.date(2026, 11, 21)
    a, b = charts.sample_usage(today=day), charts.sample_usage(today=day)
    assert a == b and len(a) == 14 and all(r["answered"] <= r["queries"] for r in a)


def test_chart_builders_return_figures():
    class Doc:
        def __init__(self, dep, n):
            self.department, self.chunk_count = dep, n

    class Q:
        question, total_ms, confidence, answered = "soru", 12.5, 70, True

    counts = charts.department_counts([Doc("Finans", 3), Doc("İK", 2), Doc("Finans", 1)])
    assert counts == {"Finans": 4, "İK": 2}
    pts = charts.quality_points([Q(), Q()])
    assert [p["n"] for p in pts] == [1, 2]
    for fig in (charts.usage_figure(charts.sample_usage(), 3), charts.department_figure(counts),
                charts.department_figure({}), charts.quality_figure(pts)):
        assert fig is not None
        if not charts.PLOTLY:
            assert fig["data"]["values"] is not None


# ------------------------------------------------------------------ uygulama (AppTest)
@pytest.fixture
def app_test(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    try:
        import pyarrow  # noqa: F401
    except ImportError:  # yalnızca pyarrow'suz test ortamları için
        import streamlit.dataframe_util as du

        monkeypatch.setattr(du, "convert_anything_to_arrow_bytes", lambda *a, **k: b"")
    at = st_testing.AppTest.from_file(APP, default_timeout=90)
    at.session_state["stream_delay"] = 0.0
    at.session_state["demo_delay"] = 0.0
    return at.run()


def test_app_starts_with_dashboard(app_test):
    at = app_test
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["💬 Asistan", "📈 Analitik", "📚 Doküman Kütüphanesi"]
    html = " ".join(m.value for m in at.markdown)
    assert "Doküman" in html and "Aktif tenant" in html and "Sistem sağlığı" in html and "Sağlıklı" in html
    assert sum(bool(b.key and b.key.startswith("sample_")) for b in at.button) == 4


def test_sample_question_button_shows_citation_cards(app_test):
    at = app_test
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert any(e.label.startswith("📎 Kaynaklar") for e in at.expander)
    assert any('class="glass cite"' in m.value and "Sayfa" in m.value for m in at.markdown)


def test_live_demo_mode_runs_full_script(app_test):
    at = app_test
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    msgs = at.session_state["chats"][SAMPLE]
    assert len(msgs) == 12 and at.session_state["demo_queue"] == []
    assert all(m["response"].is_answered for m in msgs[1:10:2])  # 4 politika sorusu + sözleşme PDF
    assert not msgs[-1]["response"].is_answered  # kapsam dışı soru: uydurma yok


def test_tenants_are_isolated_in_ui(app_test):
    at = app_test
    at.button(key="sample_1").click().run()
    at.sidebar.selectbox[0].set_value(OTHER).run()
    assert not at.exception, at.exception
    assert len(at.chat_message) == 0
    at.chat_input[0].set_value("Uzaktan çalışma kuralları nelerdir?").run()
    assert not at.session_state["chats"][OTHER][-1]["response"].is_answered  # B'nin belgesi yok
    at.sidebar.selectbox[0].set_value(SAMPLE).run()
    assert len(at.chat_message) == 2


def test_library_lists_and_removes_documents(app_test):
    at = app_test
    delete = [b for b in at.button if b.key and b.key.startswith("del_")]
    assert len(delete) == 3
    delete[0].click().run()
    assert not at.exception, at.exception
    assert len([b for b in at.button if b.key and b.key.startswith("del_")]) == 2
