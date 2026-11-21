"""Araştırmacı + Denetçi ikili ajan katmanı: onay, düzeltme, ret, atlama ve arayüz rozetleri."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.auditor import GeminiAuditor, RuleBasedAuditor, normalize_number, numbers_in
from src.config import Settings
from src.llm import LLMError
from src.local_engine import LocalWorkspace
from src.rag_engine import NO_INFO_ANSWER

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SAMPLE = "Örnek Holding A.Ş."
OVERTIME = "Fazla mesai ücreti nasıl hesaplanır?"
OUT_OF_SCOPE = "Şirketin Mars'taki ofisinin adresi nedir?"


@pytest.fixture(scope="module")
def ws() -> LocalWorkspace:
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    w.load_sample_documents()
    return w


@pytest.fixture(scope="module")
def researched(ws):
    """Araştırmacı'nın gerçek (kaynaklı) yanıtı."""
    r = ws.ask(OVERTIME)
    assert r.is_answered and r.citations
    return r


def with_answer(resp, answer):  # noqa: ANN001, ANN201
    return dataclasses.replace(resp, answer=answer)


# ------------------------------------------------------------------ yardımcılar
@pytest.mark.parametrize(("raw", "norm"), [("1.200", "1200"), ("1,5", "1.5"), ("10:00", "10:00"),
                                           ("15.02.2026", "15.02.2026"), ("45", "45")])
def test_number_normalization(raw, norm):
    assert normalize_number(raw) == norm


def test_numbers_in_text():
    assert numbers_in("Ücret %50 fazlası, 1.200 TL ve 10:00–16:00") == {"50", "1200", "10:00", "16:00"}


# ------------------------------------------------------------------ kural tabanlı denetçi
def test_supported_answer_is_approved(researched):
    a = RuleBasedAuditor().audit(researched)
    assert a.status == "approved" and a.answer == researched.answer
    assert a.checks and all(c.supported for c in a.checks)
    assert a.duration_ms >= 0 and "destekleniyor" in a.summary


def test_hallucinated_number_is_removed_and_answer_revised(researched):
    label = researched.citations[0].label
    fake = researched.answer + f"\n- Fazla çalışma ücreti normal saatlik ücretin %75 fazlasıyla ödenir. [{label}]"
    a = RuleBasedAuditor().audit(with_answer(researched, fake))
    assert a.status == "revised"
    assert "%75" not in a.answer and "75" not in "".join(c.text for c in a.checks if c.supported)
    assert any("75" in c.reason for c in a.removed)
    assert "%50" in a.answer  # doğru iddia korunur


def test_unrelated_sentence_is_removed(researched):
    label = researched.citations[0].label
    fake = researched.answer + f"\n- Şirket çalışanlarına her yıl ücretsiz Maldivler tatili hediye edilir. [{label}]"
    a = RuleBasedAuditor().audit(with_answer(researched, fake))
    assert a.status == "revised" and "Maldivler" not in a.answer
    assert any("örtüşme" in c.reason for c in a.removed)


def test_wrong_quote_is_removed(researched):
    label = researched.citations[0].label
    fake = researched.answer + f'\n- Yönetmelik "fazla mesai yasaktır ve ödenmez" der. [{label}]'
    a = RuleBasedAuditor().audit(with_answer(researched, fake))
    assert a.status == "revised" and any("Alıntı" in c.reason for c in a.removed)


def test_fully_unsupported_answer_is_rejected(researched):
    label = researched.citations[0].label
    fake = f"- Fazla mesai saat başına 999 TL sabit ücretle ödenir. [{label}]"
    a = RuleBasedAuditor().audit(with_answer(researched, fake))
    assert a.status == "rejected" and a.answer == NO_INFO_ANSWER


def test_out_of_scope_question_is_rejected(ws):
    r, a = ws.ask_audited(OUT_OF_SCOPE)
    assert not r.is_answered and a.status == "rejected" and a.answer == NO_INFO_ANSWER
    assert "bilgi yok" in a.summary


def test_toggle_off_skips_audit(ws):
    r, a = ws.ask_audited(OVERTIME, audit=False)
    assert a.status == "skipped" and a.answer == r.answer and a.checks == []
    assert ws.history[-1].audit_status == "skipped"


def test_workspace_audits_by_default_and_logs_status(ws):
    r, a = ws.ask_audited("Uzaktan çalışma kuralları nelerdir?")
    assert a.status == "approved" and ws.history[-1].audit_status == "approved"


# ------------------------------------------------------------------ Gemini denetçi (sahte LLM)
class FakeLLM:
    def __init__(self, verdict=None, error=False):  # noqa: ANN001
        self.verdict, self.error, self.calls = verdict, error, []

    def generate_json(self, prompt, schema, *, system=None, fast=True):  # noqa: ANN001, ANN201
        self.calls.append({"prompt": prompt, "fast": fast, "system": system})
        if self.error:
            raise LLMError("kota")
        return schema.model_validate(self.verdict)

    def stream(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        yield ""


def test_gemini_auditor_approves_and_uses_fast_model(researched):
    llm = FakeLLM({"verdict": "approved", "corrected_answer": researched.answer, "issues": []})
    a = GeminiAuditor(llm).audit(researched)
    assert a.status == "approved" and a.method == "Gemini Flash denetçi"
    assert llm.calls[0]["fast"] is True and "KAYNAK PARÇALARI" in llm.calls[0]["prompt"]


def test_gemini_auditor_revision_is_reported(researched):
    first = researched.answer.splitlines()
    corrected = "\n".join(first[:2])
    llm = FakeLLM({"verdict": "revised", "corrected_answer": corrected, "issues": ["%75 oranı kaynakta yok"]})
    a = GeminiAuditor(llm).audit(researched)
    assert a.status == "revised" and a.answer == corrected and "%75" in a.summary


def test_gemini_auditor_safety_net_catches_approved_hallucination(researched):
    label = researched.citations[0].label
    bad = researched.answer + f"\n- Ücret %90 fazlasıyla ödenir. [{label}]"
    llm = FakeLLM({"verdict": "approved", "corrected_answer": bad, "issues": []})
    a = GeminiAuditor(llm).audit(researched)
    assert a.status == "revised" and "%90" not in a.answer


def test_gemini_auditor_reject_and_error_fallback(researched):
    rej = GeminiAuditor(FakeLLM({"verdict": "rejected", "corrected_answer": "", "issues": []})).audit(researched)
    assert rej.status == "rejected" and rej.answer == NO_INFO_ANSWER
    fb = GeminiAuditor(FakeLLM(error=True)).audit(researched)
    assert fb.status == "approved" and "kural tabanlı" in fb.summary


# ------------------------------------------------------------------ arayüz
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


def html(at) -> str:  # noqa: ANN001
    return " ".join(m.value for m in at.markdown)


def test_auditor_toggle_is_on_by_default(app_test):
    toggle = next(t for t in app_test.toggle if "Denetçi" in t.label)
    assert toggle.value is True


def test_demo_shows_approved_badges_and_rejection_badge(app_test):
    at = app_test
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    msgs = at.session_state["chats"][SAMPLE]
    statuses = [m["audit"].status for m in msgs if m["role"] == "assistant"]
    assert statuses == ["approved"] * 5 + ["rejected"]  # 4 politika + sözleşme PDF, Mars reddi
    page = html(at)
    assert page.count("✓ Denetçi Onaylı") >= 4
    assert "✕ Kaynakta Bulunamadı — Uydurma Engellendi" in page
    reports = [e for e in at.expander if e.label == "🛡️ Denetçi Kontrol Raporu"]
    assert len(reports) == 6
    assert any("Denetim süresi" in c.value for c in at.caption)


def test_revised_answer_badge_is_rendered(app_test, monkeypatch):
    at = app_test
    from src import auditor as aud

    original = aud.RuleBasedAuditor.audit

    def tamper(self, response):  # noqa: ANN001, ANN202
        if response.is_answered and response.citations:
            label = response.citations[0].label
            response = dataclasses.replace(
                response, answer=response.answer + f"\n- Bu kural 2031 yılında yürürlükten kalkacaktır. [{label}]")
        return original(self, response)

    monkeypatch.setattr(aud.RuleBasedAuditor, "audit", tamper)
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    audit = at.session_state["chats"][SAMPLE][-1]["audit"]
    assert audit.status == "revised" and "2031" not in audit.answer
    assert "✎ Denetçi Tarafından Düzeltildi" in html(at)


def test_toggle_off_hides_badges_and_report(app_test):
    at = app_test
    next(t for t in at.toggle if "Denetçi" in t.label).set_value(False).run()
    at.button(key="sample_1").click().run()
    assert not at.exception, at.exception
    assert at.session_state["chats"][SAMPLE][-1]["audit"].status == "skipped"
    assert "class=\"audit-badge" not in html(at)
    assert not any(e.label == "🛡️ Denetçi Kontrol Raporu" for e in at.expander)
    assert any(e.label.startswith("📎 Kaynaklar") for e in at.expander)
