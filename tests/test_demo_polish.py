"""Canlı demo iyileştirmelerinin testleri.

1. Akışlı yanıt: model çıktısı parça parça akar; Denetçi akıştan sonra çalışır, kayıt tam bir kez olur.
2. Güven skoru: Denetçi onaylı, iyi kaynaklı yanıt "Yüksek güvenilirlik" bandında; red/servis hatası %0.
3. Atıflar ekranda [1], [2]; Denetçi raporunda ham Markdown/etiket boşluğu yok, başlıklar iddia sayılmaz.
4. Mobil: kenar çubuğu "auto", ana alanda demo butonu, KPI şeridi için duyarlı CSS.
5. Örnek belgeler sunucu başına önbelleklenir; uyanık tutma iş akışı mevcut.
6. Analitikte simüle veri açıkça işaretli.
7. Tablo içeren örnek sözleşme PDF'i Hukuk departmanında, demo senaryosunda.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from src.auditor import AuditResult, RuleBasedAuditor, display_text
from src.config import Settings
from src.document_loader import _unwrap_lines
from src.local_engine import LocalWorkspace, calibrated_confidence
from ui import showcase_charts as charts
from ui import showcase_theme as theme
from ui.citations import number_citations, streaming_view

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SAMPLE = "Örnek Holding A.Ş."
OVERTIME = "Fazla mesai ücreti nasıl hesaplanır?"
CONTRACT = "Bakım sözleşmesinde P1 çözüm süresi aşılırsa hizmet kredisi ne kadar?"
IK = "IK_Izin_ve_Calisma_Yonetmeligi.md"


@pytest.fixture(scope="module")
def ws() -> LocalWorkspace:
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    w.load_sample_documents()
    return w


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


class ChunkedLLM:
    """Gemini gibi parça parça yazan sahte model; bağlamdaki ilk etiketle atıflı yanıt üretir."""

    def __init__(self, extra: str = ""):
        self.extra = extra

    def stream(self, messages, **_):  # noqa: ANN001, ANN003, ANN201
        prompt = messages[-1][1]
        label = re.search(r'etiket="\[([^\]]+)\]"', prompt).group(1)
        text = (f"Fazla çalışma ücreti normal saatlik ücretin **%50 fazlasıyla** ödenir [{label}]. "
                f"Haftalık 45 saati aşan çalışmalar fazla çalışma sayılır [{label}].{self.extra}")
        for i in range(0, len(text), 12):
            yield text[i:i + 12]

    def generate_json(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        from src.llm import LLMError

        raise LLMError("yok")


# ====================================================================== 1. akış
def test_answer_streams_in_chunks_then_audits_once():
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    w.load_sample_documents()
    w.engine.llm = ChunkedLLM()
    pending = w.start(OVERTIME)
    chunks = list(pending.tokens)
    assert len(chunks) > 5 and len(w.history) == 0  # akış bitene kadar kayıt yok
    turn = pending.finish()
    assert pending.finish() is turn and len(w.history) == 1  # tam bir kez
    assert turn.audit.status == "approved" and "".join(chunks).startswith("Fazla çalışma")


def test_unsupported_streamed_claim_is_removed_after_audit():
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    w.load_sample_documents()
    w.engine.llm = ChunkedLLM(extra=f" Bu ücret 2031 yılında %75'e çıkacaktır [{IK} - Sayfa 1].")
    turn = w.run(OVERTIME)
    assert turn.audit.status == "revised" and "2031" not in turn.audit.answer


def test_refused_and_compound_questions_still_work_through_start(ws):
    assert ws.start("Şirketin Mars'taki ofisinin adresi nedir?").finish().audit.status == "rejected"
    t = ws.start("Fazla mesai ücreti nasıl hesaplanır ve şirketin Mars'taki ofisinin adresi nedir?").finish()
    assert t.response.is_answered and [p.answered for p in t.insight.parts] == [True, False]


def test_ui_streams_and_records_audited_text(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    monkeypatch.setattr(ws.engine, "llm", ChunkedLLM())
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    msg = at.session_state["chats"][SAMPLE][-1]
    assert msg["audit"].status == "approved" and msg["content"] == msg["audit"].answer
    shown = " ".join(m.value for m in at.markdown)
    assert "[1]" in shown and f"[{IK} - Sayfa" not in shown.split('class="cite-label"')[0]


# ====================================================================== 2. güven
def test_verified_answer_gets_high_confidence(ws):
    t = ws.run(OVERTIME)
    assert t.audit.status == "approved"
    assert t.insight.confidence >= 85 and t.insight.confidence_level == "Yüksek güvenilirlik"
    assert t.insight.confidence <= 97
    assert ws.history[-1].confidence == t.insight.confidence  # KPI ortalaması aynı skoru kullanır


def test_confidence_levels_follow_verification(ws):
    resp = ws.ask(OVERTIME)
    ok = AuditResult("approved", resp.answer, "", "", 0, [])
    revised = AuditResult("revised", resp.answer, "", "", 0, [])
    rejected = AuditResult("rejected", "", "", "", 0, [])
    assert calibrated_confidence(resp, ok)[0] > calibrated_confidence(resp, revised)[0] > 0
    assert calibrated_confidence(resp, rejected) == (0, "Bilgi yok")
    assert calibrated_confidence(dataclasses.replace(resp, error="503"), ok)[0] == 0


def test_ui_shows_confidence_level_pill(app_test):
    app_test.button(key="sample_0").click().run()
    html = " ".join(m.value for m in app_test.markdown)
    assert 'class="conf high"' in html and "Yüksek güvenilirlik" in html


# ====================================================================== 3. atıf ve rapor temizliği
class _Cit:
    def __init__(self, label: str, source: str, page: int):
        self.label, self.source, self.page = label, source, page


def test_number_citations_replaces_long_labels():
    a, b = _Cit("A.md - Sayfa 1", "A.md", 1), _Cit("B.pdf - Sayfa 2", "B.pdf", 2)
    text = "Bir [A.md - Sayfa 1]. İki [B.pdf - Sayfa 2]. Üç [A.md - Sayfa 1; B.pdf - Sayfa 2]. Not [x]."
    out, order = number_citations(text, [a, b])
    assert out == "Bir [1]. İki [2]. Üç [1][2]. Not [x]."
    assert order == [a, b]


def test_number_citations_matches_label_without_extension():
    a = _Cit("A.md - Sayfa 1", "A.md", 1)
    out, order = number_citations("Bilgi [a - sayfa 1].", [a])
    assert out == "Bilgi [1]." and order == [a]


def test_streaming_view_hides_half_written_label():
    a = _Cit("A.md - Sayfa 1", "A.md", 1)
    assert streaming_view("Bir [A.md - Sayfa 1]. İki [A.md - Say", [a]) == "Bir [1]. İki "


def test_citation_card_shows_number():
    card = theme.citation_card("A.md - Sayfa 1", "A.md", 1, "Genel", 80, "metin metin metin", number=2)
    assert '<span class="cite-num">[2]</span>' in card


def test_report_claims_are_clean_and_headings_not_counted(ws):
    base = ws.ask(OVERTIME)
    label = base.citations[0].label
    answer = (f"### Ücret Kuralları\n\n**Fazla Mesai**\n\n- **Ücret:** normal saatlik ücretin **%50 fazlasıyla** "
              f"ödenir [{label}].")
    a = RuleBasedAuditor().audit(dataclasses.replace(base, answer=answer))
    assert len(a.checks) == 1  # başlık ve kalın başlık satırı iddia değil
    text = a.checks[0].text
    assert "**" not in text and "[" not in text and " ." not in text
    assert text.startswith("Ücret: normal saatlik ücretin %50 fazlasıyla ödenir")


def test_bold_line_with_number_is_still_checked():
    assert display_text("**Ücret %75 fazlasıyla ödenir.** [x.md - Sayfa 1]") == "Ücret %75 fazlasıyla ödenir."


# ====================================================================== 4. mobil
def test_mobile_layout_rules():
    app_src = Path(APP).read_text(encoding="utf-8")
    assert 'initial_sidebar_state="auto"' in app_src
    assert "@media (max-width: 640px)" in theme.CSS and ':has(.kpi)' in theme.CSS


def test_demo_button_is_in_main_area(app_test):
    main_demo = [b for b in app_test.main.button if b.label.startswith("▶")]
    assert main_demo and main_demo[0].key == "demo_main"
    main_demo[0].click().run()
    assert not app_test.exception and len(app_test.session_state["chats"][SAMPLE]) == 12


# ====================================================================== 5. açılış performansı
def test_sample_documents_are_cached_across_sessions(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit.dataframe_util as du

    monkeypatch.setattr(du, "convert_anything_to_arrow_bytes", lambda *a, **k: b"")
    calls = {"n": 0}
    original = LocalWorkspace.sample_documents

    def counting(self):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(LocalWorkspace, "sample_documents", counting)
    import streamlit as st

    st.cache_resource.clear()
    for _ in range(2):  # iki ayrı oturum
        at = st_testing.AppTest.from_file(APP, default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        assert len(at.session_state["workspaces"][SAMPLE].documents) == 3
    assert calls["n"] == 1


def test_keep_alive_workflow_targets_live_app():
    workflow = (ROOT / ".github" / "workflows" / "keep-alive.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "keep_alive.py").read_text(encoding="utf-8")
    assert "schedule:" in workflow and "scripts/keep_alive.py" in workflow
    assert "streamlitApp" in script and "get this app back up" in script


# ====================================================================== 6. simüle veri
def test_simulated_data_is_clearly_labelled(monkeypatch):
    class Go:
        class Figure:
            def __init__(self, data=None):  # noqa: ANN001
                self.data, self.layout = [], {}

            def add_trace(self, t):  # noqa: ANN001
                self.data.append(t)

            def update_layout(self, **kw):  # noqa: ANN003
                self.layout.update(kw)

        @staticmethod
        def Scatter(**kw):  # noqa: ANN003, ANN205, N802
            return kw

    monkeypatch.setattr(charts, "go", Go)
    monkeypatch.setattr(charts, "PLOTLY", True)
    fig = charts.usage_figure(charts.sample_usage())
    assert "SİMÜLE VERİ" in fig.layout["title"]["text"]
    assert fig.layout["annotations"][0]["text"] == "SİMÜLE VERİ"  # filigran
    assert "SİMÜLE VERİ" in theme.SIMULATED_BANNER


def test_analytics_tab_shows_simulation_banner(app_test):
    assert any("DEMO ORTAMI · SİMÜLE VERİ" in m.value for m in app_test.markdown)


# ====================================================================== 7. örnek sözleşme
def test_contract_pdf_is_indexed_under_legal_with_tables(ws):
    doc = next(d for d in ws.documents if d.filename == "Hukuk_Hizmet_Sozlesmesi.pdf")
    assert doc.department == "Hukuk" and doc.page_count == 2
    texts = [c.text for c in ws.store.get_chunks(["Hukuk"])]
    assert any("| P1 – Kritik |" in t and "4 saat" in t for t in texts)  # SLA tablosu tablo olarak korunur
    assert any("480.000 TL" in t for t in texts)


def test_contract_question_is_answered_from_pdf(ws):
    t = ws.run(CONTRACT)
    assert t.audit.status == "approved" and "%5" in t.audit.answer
    assert {c.source for c in t.response.citations} == {"Hukuk_Hizmet_Sozlesmesi.pdf"}
    assert "YAZILIM BAKIM" not in t.audit.answer  # büyük harfli başlık yanıta girmez


def test_pdf_soft_line_wraps_are_joined():
    text = "Madde 5 – Hizmet Kredisi\nBir takvim ayında P1 süresi aşılırsa bedelin\n%5'i oranında kredi tanınır.\nYeni cümle."
    assert _unwrap_lines(text).splitlines() == [
        "Madde 5 – Hizmet Kredisi",
        "Bir takvim ayında P1 süresi aşılırsa bedelin %5'i oranında kredi tanınır.",
        "Yeni cümle.",
    ]


def test_contract_generator_script_exists():
    script = ROOT / "scripts" / "make_sample_contract.py"
    assert script.exists() and "Hukuk_Hizmet_Sozlesmesi.pdf" in script.read_text(encoding="utf-8")


def test_timing_meta_is_in_seconds(app_test):
    app_test.button(key="sample_0").click().run()
    assert re.search(r"\d+\.\d sn · \d+ kaynak", " ".join(m.value for m in app_test.markdown))
