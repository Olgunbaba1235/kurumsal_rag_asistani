"""Uçtan uca kullanıcı testlerinde bulunan 8 hata/UX eksiğinin regresyon testleri.

1. Departman filtresi varsayılan boş; filtre kaynaklı redde neden açıkça gösterilir
2. "X sorgu bu oturumda" yalnızca gerçek yeni sorguda artar (rerun/kesilen çizim sayacı şişirmez)
3. Red rozeti raporda tekrar etmez; "Doğrulama notları" tek "Denetçi Kontrol Raporu"nda birleşir
4. Kaynak kartları kapalı gelir; ham Markdown başlıkları/tabloları karta taşmaz, atıf etiketi ayrı kaptadır
5. KPI kartları eşit yükseklikte; hızlı başlangıç buton metinleri kesilmez
6. Analitik: Türkçe tarih etiketleri, "Güven (%)" ekseni kesilmez
7. Streamlit Deploy/menü gizli; dosya yükleme alanı Türkçe
8. Bileşik sorularda kısmi yanıt + not; "dk" / "dakika" kısaltma eşitliği
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from pathlib import Path

import pytest

from src.auditor import RuleBasedAuditor
from src.compound import split_compound
from src.config import Settings
from src.local_engine import LocalWorkspace
from src.rag_engine import is_refusal
from src.text_utils import tokenize
from ui import showcase_charts as charts
from ui import showcase_theme as theme

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SAMPLE = "Örnek Holding A.Ş."
OTHER = "Ege Vadi Gıda A.Ş."
HR, FIN = "İnsan Kaynakları", "Finans"
SAMPLE_QUESTIONS = [
    "Fazla mesai ücreti nasıl hesaplanır?",
    "Uzaktan çalışma kuralları nelerdir?",
    "Satın almada kaç teklif almak gerekir?",
    "Yurt içi seyahatte günlük yemek harcırahı ne kadar?",
]
MARS = "Şirketin Mars'taki ofisinin adresi nedir?"
COMPOUND = "Fazla mesai ücreti nasıl hesaplanır ve şirketin Mars'taki ofisinin adresi nedir?"
PURCHASE = "Satın almada kaç teklif almak gerekir?"
FILTER_NOTE = (f"Aktif departman filtresi ({HR}) nedeniyle diğer belgeler taranmadı. "
               "Tüm belgelerde aramak için filtreyi kaldırın.")


def make_ws() -> LocalWorkspace:
    return LocalWorkspace(Settings(_env_file=None), api_key=None)


@pytest.fixture(scope="module")
def ws() -> LocalWorkspace:
    w = make_ws()
    w.load_sample_documents()
    return w


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


def html(at) -> str:  # noqa: ANN001
    return " ".join(m.value for m in at.markdown)


def session_count(at) -> int:  # noqa: ANN001
    kpi = next(m.value for m in at.markdown if "sorgu bu oturumda" in m.value)
    return int(re.search(r"(\d+) sorgu bu oturumda", kpi).group(1))


def chats(at, tenant: str = SAMPLE) -> list[dict]:  # noqa: ANN001
    return at.session_state["chats"][tenant]


def history_len(at, tenant: str = SAMPLE) -> int:  # noqa: ANN001
    return len(at.session_state["workspaces"][tenant].history)


# ====================================================================== 1. departman filtresi
def test_department_filter_defaults_to_all_departments(app_test):
    ms = app_test.sidebar.multiselect[0]
    assert ms.label == "Departman filtresi" and ms.value == []
    assert "Tüm departmanlar" in " ".join(c.value for c in app_test.caption)


def test_stale_filter_value_is_dropped(app_test):
    at = app_test
    at.session_state["dept_filter"] = ["Kapatılmış Departman"]
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.multiselect[0].value == []


def test_filter_resets_on_tenant_switch_and_demo_start(app_test):
    at = app_test
    at.sidebar.multiselect[0].set_value([HR]).run()
    at.sidebar.selectbox[0].set_value(OTHER).run()
    assert at.sidebar.multiselect[0].value == []
    at.sidebar.multiselect[0].set_value([FIN]).run()
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    assert at.session_state["tenant"] == SAMPLE and at.sidebar.multiselect[0].value == []


def test_filtered_refusal_explains_reason_engine(ws):
    t = ws.run(PURCHASE, departments=[HR])
    assert not t.response.is_answered and t.audit.status == "rejected"
    assert t.insight.filter_note == FILTER_NOTE
    assert t.insight.outside_departments == [FIN] and FIN in t.insight.filter_probe
    assert ws.run(PURCHASE).insight.filter_note == ""  # filtre yoksa not yok
    assert ws.run(SAMPLE_QUESTIONS[0], departments=[HR]).insight.filter_note == ""  # yanıtlandıysa not yok


def test_filtered_refusal_reason_and_one_click_retry_in_ui(app_test):
    at = app_test
    at.sidebar.multiselect[0].set_value([HR]).run()
    at.chat_input[0].set_value(PURCHASE).run()
    assert not at.exception, at.exception
    assert not chats(at)[-1]["response"].is_answered
    page = html(at)
    assert FILTER_NOTE in page and f"{FIN} belgelerinde" in page
    report = next(e for e in at.expander if e.label == "🛡️ Denetçi Kontrol Raporu")
    assert "Filtre uyarısı" in " ".join(m.value for m in report.markdown)

    next(b for b in at.button if b.key and b.key.startswith("retry_")).click().run()
    assert not at.exception, at.exception
    assert at.sidebar.multiselect[0].value == []
    last = chats(at)[-1]["response"]
    assert last.is_answered and last.citations and all(c.department == FIN for c in last.citations)
    assert session_count(at) == 2


# ====================================================================== 2. sorgu sayacı
def test_counter_counts_each_question_once_and_ignores_reruns(app_test):
    at = app_test
    assert session_count(at) == 0
    at.button(key="sample_0").click().run()
    assert session_count(at) == 1  # KPI aynı çalıştırmada güncel (sorgudan sonra çizilir)
    for _ in range(3):
        at.run()
    next(t for t in at.toggle if "Denetçi" in t.label).set_value(False).run()
    assert session_count(at) == 1 and history_len(at) == 1 and len(chats(at)) == 2
    at.chat_input[0].set_value("Uzaktan çalışma kuralları nelerdir?").run()
    assert session_count(at) == 2 and history_len(at) == 2


def test_answer_is_recorded_before_rendering_exactly_once(app_test, monkeypatch):
    """Çizim yarıda kesilse (rerun/bağlantı kopması) bile soru tam bir kez kaydedilir ve tekrar sorulmaz."""
    at = app_test
    original, calls = theme.audit_badge, {"n": 0}

    def interrupted(status):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("çizim yarıda kesildi")
        return original(status)

    monkeypatch.setattr(theme, "audit_badge", interrupted)
    at.button(key="sample_0").click().run()
    assert at.exception  # ilk çizim kesildi…
    # …ama kayıt tamam: sohbet + oturum metriği yazıldı, gelen kutusu boşaldı
    assert len(chats(at)) == 2 and history_len(at) == 1 and at.session_state["inbox"] == []
    at.run()
    at.run()
    assert not at.exception, at.exception
    assert len(chats(at)) == 2 and history_len(at) == 1 and session_count(at) == 1


def test_demo_restart_and_clear_reset_session_metrics(app_test):
    at = app_test
    start = next(b for b in at.button if b.label.startswith("▶"))
    start.click().run()
    assert session_count(at) == 5 and len(chats(at)) == 10
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert session_count(at) == 5 and len(chats(at)) == 10  # 10 değil: demo yeniden başladı
    next(b for b in at.button if b.label.startswith("🧹")).click().run()
    assert session_count(at) == 0 and chats(at) == []


# ====================================================================== 3. rozet ve rapor
def test_report_has_no_duplicate_badge_and_merges_notes(app_test):
    at = app_test
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    page = html(at)
    assert page.count("✕ Kaynakta Bulunamadı — Uydurma Engellendi") == 1  # yalnızca yanıtın üstünde
    reports = [e for e in at.expander if e.label == "🛡️ Denetçi Kontrol Raporu"]
    assert len(reports) == 5
    for report in reports:
        inner = " ".join(m.value for m in report.markdown)
        assert 'class="audit-badge' not in inner
        assert "En iyi alaka skoru" in inner and "Eşik değeri" in inner and "%30" in inner
    assert not any(e.label == "ℹ️ Doğrulama notları" for e in at.expander)  # ayrı kutu yok


def test_notes_box_is_used_only_when_auditor_is_off(app_test):
    at = app_test
    next(t for t in at.toggle if "Denetçi" in t.label).set_value(False).run()
    at.chat_input[0].set_value(MARS).run()
    notes = [e for e in at.expander if e.label == "ℹ️ Doğrulama notları"]
    assert len(notes) == 1 and "Eşik değeri" in " ".join(m.value for m in notes[0].markdown)


# ====================================================================== 4. kaynak kartları
def test_sources_expander_is_collapsed_by_default(app_test):
    at = app_test
    at.button(key="sample_3").click().run()
    sources = [e for e in at.expander if e.label.startswith("📎 Kaynaklar")]
    assert sources and not any(e.proto.expanded for e in sources)


def test_citation_card_renders_markdown_safely():
    evidence = ("## 5. İş Seyahati Harcamaları\n\n### 5.2 Konaklama ve Harcırah Limitleri\n\n"
                "| Kalem | Yurt İçi (Büyükşehir) | Yurt Dışı |\n| --- | --- | --- |\n"
                "| Günlük yemek harcırahı | 1.200 TL | 60 EUR |\n\n- Uçak: **ekonomi** sınıfı")
    card = theme.citation_card("Finans.md - Sayfa 3", "Finans.md", 3, "Finans", 77, evidence)
    assert "\n" not in card  # tek satır: Markdown ayrıştırıcısı HTML bloğunu bölemez
    assert "##" not in card and "# " not in card  # ham başlık diyezi yok
    assert '<div class="cite-h">5.2 Konaklama ve Harcırah Limitleri</div>' in card
    assert "<th>Kalem</th>" in card and "<td>1.200 TL</td>" in card and "<strong>ekonomi</strong>" in card
    quote = card[card.index('class="quote"'):card.index('class="cite-label"')]
    assert "Atıf etiketi" not in quote and "|" not in quote  # etiket tablonun/parçanın dışında
    assert card.index("</table>") < card.index("Atıf etiketi")


def test_citation_card_long_table_is_truncated_by_rows():
    rows = "\n".join(f"| Kalem {i} | {i * 100} TL |" for i in range(200))
    card = theme.citation_card("x", "x.md", 1, "", 50, f"| A | B |\n| --- | --- |\n{rows}")
    assert card.count("<tr>") < 200 and "cite-more" in card and card.count("<table>") == 1


# ====================================================================== 5. yerleşim
def test_kpi_cards_share_a_fixed_height():
    rule = re.search(r"\.kpi \{([^}]*)\}", theme.CSS).group(1)
    assert re.search(r"(?<!-)height:\s*\d+px", rule) and "flex-direction: column" in rule
    card = theme.kpi_card("🏢", "Aktif tenant", "Çok Uzun Bir Şirket Adı Anonim Şirketi", "alt", small=True)
    assert 'title="Çok Uzun Bir Şirket Adı Anonim Şirketi"' in card  # kırpılan metin ipucunda tam görünür


def test_sample_buttons_use_short_labels_and_send_full_question(app_test):
    at = app_test
    buttons = [b for b in at.button if b.key and b.key.startswith("sample_")]
    assert [b.help for b in buttons] == SAMPLE_QUESTIONS
    assert all(len(b.label) <= 32 for b in buttons)
    at.button(key="sample_3").click().run()
    assert chats(at)[0]["content"] == SAMPLE_QUESTIONS[3]


# ====================================================================== 6. analitik
class _FakeGo:
    """Plotly kurulu olmasa da Plotly yolunu test etmek için en küçük `graph_objects` taklidi."""

    class Figure:
        def __init__(self, data=None):  # noqa: ANN001
            self.data, self.layout = ([data] if data else []), {}

        def add_trace(self, trace):  # noqa: ANN001
            self.data.append(trace)

        def update_layout(self, **kw):  # noqa: ANN003
            self.layout.update(kw)

    @staticmethod
    def Scatter(**kw):  # noqa: ANN003, ANN205, N802
        return {"type": "scatter", **kw}

    @staticmethod
    def Pie(**kw):  # noqa: ANN003, ANN205, N802
        return {"type": "pie", **kw}


@pytest.fixture
def fake_plotly(monkeypatch):
    monkeypatch.setattr(charts, "go", _FakeGo)
    monkeypatch.setattr(charts, "PLOTLY", True)


def test_turkish_date_labels():
    assert charts.tr_day_label("2026-11-07") == "7 Kas"
    assert charts.tr_day_label("2026-01-31") == "31 Oca"
    assert charts.tr_long_date("2026-11-07") == "7 Kasım 2026, Cumartesi"


def test_usage_chart_has_turkish_axis_labels(fake_plotly):
    fig = charts.usage_figure(charts.sample_usage(today=dt.date(2026, 11, 21)))
    x = fig.data[0]["x"]
    assert x[0] == "7 Kas" and x[-1] == "20 Kas"
    xaxis = fig.layout["xaxis"]
    assert xaxis["type"] == "category" and xaxis["ticktext"][:3] == ["7 Kas", "9 Kas", "11 Kas"]
    english = r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
    assert not re.search(english, str(fig.data) + str(fig.layout))
    assert fig.data[0]["customdata"][0] == "7 Kasım 2026, Cumartesi"


def test_quality_chart_leaves_room_for_y_axis_title(fake_plotly):
    fig = charts.quality_figure([{"n": 1, "ms": 5.0, "confidence": 70, "answered": True, "question": "q"}])
    assert fig.layout["margin"]["l"] >= 80 and fig.layout["yaxis"]["automargin"] is True
    assert fig.layout["yaxis"]["title"]["text"] == "Güven (%)"


def test_vega_fallback_uses_turkish_labels(monkeypatch):
    monkeypatch.setattr(charts, "PLOTLY", False)
    spec = charts.usage_figure(charts.sample_usage(today=dt.date(2026, 11, 21)))
    assert spec["data"]["values"][0]["gün"] == "7 Kas" and spec["encoding"]["x"]["field"] == "gün"


# ====================================================================== 7. araç çubuğu ve yükleme alanı
def test_streamlit_deploy_and_menu_are_hidden():
    for selector in ('[data-testid="stAppDeployButton"]', '[data-testid="stMainMenu"]',
                     '[data-testid="stToolbarActions"]'):
        assert selector in theme.CSS
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r'^\[client\][^\[]*^toolbarMode\s*=\s*"minimal"', config, re.MULTILINE)


def test_file_uploader_texts_are_turkish(app_test):
    css = theme.uploader_css(25, ["pdf", "docx", "md"])
    assert "Dosya seç" in css and "Dosya başına en fazla 25 MB" in css and "PDF · DOCX · MD" in css
    assert "Dosyaları buraya bırakın" in css and "per file" not in css
    sidebar_html = " ".join(m.value for m in app_test.sidebar.markdown)
    assert "Dosya seç" in sidebar_html


# ====================================================================== 8. bileşik sorular ve kısaltmalar
def test_split_compound_is_conservative():
    assert split_compound(COMPOUND) == [
        "Fazla mesai ücreti nasıl hesaplanır", "şirketin Mars'taki ofisinin adresi nedir?"]
    assert split_compound("Evlilik izni kaç gün, doğum izni kaç hafta?") == [
        "Evlilik izni kaç gün", "doğum izni kaç hafta?"]
    assert split_compound("Fazla mesai nasıl hesaplanır? Mars ofisi nerede?") == [
        "Fazla mesai nasıl hesaplanır?", "Mars ofisi nerede?"]
    # Ortak yüklemli tek soru bölünmez
    for single in ("Uzaktan çalışma kuralları ve çekirdek saatler nelerdir?",
                   "Yurt içi ve yurt dışı yemek harcırahı ne kadar?", MARS, *SAMPLE_QUESTIONS):
        assert split_compound(single) == [single]
    # Kurtarma modu (tüm soru reddedildiğinde) soru sözcüğü şartı aramaz
    assert split_compound("Mars ofisinin adresi ve fazla mesai ücreti nedir?", relaxed=True) == [
        "Mars ofisinin adresi", "fazla mesai ücreti nedir?"]


def test_compound_question_answers_supported_part_and_notes_the_rest(ws):
    n = len(ws.history)
    t = ws.run(COMPOUND)
    assert len(ws.history) == n + 1  # alt sorular ayrı sorgu sayılmaz
    r = t.response
    assert r.is_answered and r.citations and not is_refusal(r.answer) and r.confidence > 0
    assert t.audit.status == "approved"
    assert "%50" in t.audit.answer and "Mars" in t.audit.answer and "yer almamaktadır" in t.audit.answer
    assert [p.answered for p in t.insight.parts] == [True, False]
    assert any(w.startswith("Bileşik soru") for w in r.warnings)


def test_mixed_compound_question_calls_model_only_for_supported_part():
    w = make_ws()
    w.load_sample_documents()
    prompts: list[str] = []
    original = w.engine.llm.stream

    def counting(messages, **kw):  # noqa: ANN001, ANN003, ANN202
        prompts.append(messages[-1][1])
        return original(messages, **kw)

    w.engine.llm.stream = counting
    w.run(COMPOUND)
    assert len(prompts) == 1  # yalnızca kaynaklı alt soru modele gider; bütün soru boşuna çağrılmaz
    assert "Mars" not in prompts[0].split("SORU:")[1]


def test_rescue_mode_answers_when_whole_question_was_refused(ws):
    t = ws.run("Mars ofisinin adresi ve fazla mesai ücreti nedir?")
    assert t.response.is_answered and "%50" in t.audit.answer
    assert [p.answered for p in t.insight.parts] == [False, True]


def test_fully_unsupported_compound_question_is_still_refused(ws):
    t = ws.run("Mars ofisinin adresi nedir ve Jüpiter şubesi ne zaman açılacak?")
    assert not t.response.is_answered and t.audit.status == "rejected"


def test_abbreviations_are_equivalent_to_full_words(ws):
    assert tokenize("15 Dk") == tokenize("15 dakika") and tokenize("2 sa") == tokenize("2 saat")
    assert tokenize("8 km") == tokenize("8 kilometre")
    short, full = ws.run("Dinlenme arası kaç dk?"), ws.run("Dinlenme arası kaç dakika?")
    assert short.response.is_answered and short.insight.best_relevance == full.insight.best_relevance


def test_abbreviated_document_matches_full_word_question():
    w = make_ws()
    w.add_bytes("Mola.md", "# Mola\n\nÜretim hattında her vardiyada 15 Dk çay molası ve 45 Dk yemek molası "
                           "verilir.".encode(), "Genel")
    w.add_bytes("Diger.md", "# Diğer\n\nOtopark kullanımı için güvenlik birimine başvurulur ve kart "
                            "verilir.".encode(), "Genel")
    a, b = w.run("Yemek molası kaç dakika?"), w.run("Yemek molası kaç dk?")
    assert a.response.is_answered and a.response.citations[0].source == "Mola.md"
    assert a.insight.best_relevance == b.insight.best_relevance


def test_auditor_keeps_scope_notes_but_not_hidden_claims(ws):
    base = ws.ask(SAMPLE_QUESTIONS[0])
    label = base.citations[0].label
    note = "Bu konudaki ayrıntı yüklenen dokümanlarda yer almamaktadır."
    ok = RuleBasedAuditor().audit(dataclasses.replace(base, answer=f"{base.answer}\n\n{note}"))
    assert ok.status == "approved"
    sneaky = f"{base.answer}\n- Ücret %75 fazlasıyla ödenir; ayrıntı dokümanlarda yer almamaktadır. [{label}]"
    bad = RuleBasedAuditor().audit(dataclasses.replace(base, answer=sneaky))
    assert bad.status == "revised" and "%75" not in bad.answer


def test_revised_answer_keeps_paragraph_structure(ws):
    base = ws.ask(SAMPLE_QUESTIONS[0])
    label = base.citations[0].label
    note = "ℹ️ **Not:** “Mars ofisi” kısmına ilişkin bilgi yüklenen dokümanlarda yer almamaktadır."
    fake = f"{base.answer}\n- Ücret %75 fazlasıyla ödenir. [{label}]\n\n{note}"
    a = RuleBasedAuditor().audit(dataclasses.replace(base, answer=fake))
    assert a.status == "revised" and "%75" not in a.answer
    assert a.answer.endswith(f"\n\n{note}")  # not, listenin son maddesine yapışmaz


def test_compound_question_in_ui(app_test):
    at = app_test
    at.chat_input[0].set_value(COMPOUND).run()
    assert not at.exception, at.exception
    msg = chats(at)[-1]
    assert msg["response"].is_answered and msg["audit"].status == "approved"
    page = html(at)
    assert "🧩 Kısmi kaynaklı yanıt" in page and "yer almamaktadır" in page
    assert any(e.label.startswith("📎 Kaynaklar") for e in at.expander)
    assert session_count(at) == 1
