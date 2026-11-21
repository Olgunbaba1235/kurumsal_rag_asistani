"""Uçtan uca denetimde bulunan sağlamlık sorunlarının regresyon testleri.

* Sıcak yenileme: `src` modülleri sys.modules'tan yarım silinse veya başka bir `src` paketi yüklü olsa da
  uygulama açılır; çalışma anında fonksiyon içinden `src.*` import edilmez (`KeyError: 'src'`).
* Model adları: eski `gemini-2.5-*` adları hiçbir yapılandırmada kalmadı.
* Gemini 429/503: akış ve JSON çağrıları tekrar denenir; kalıcı hatalar beklemeden Türkçe mesajla döner;
  servis hatası arayüzde "bilgi yok" reddi gibi değil, tekrar denenebilir uyarı olarak görünür.
* Boş tenant, demo, filtre, Denetçi aç/kapa, sekmeler, sohbeti temizleme akışları hatasız çalışır.
* Yükleme: Türkçe karakterli UTF-8/BOM/cp1254/UTF-16 metin, DOCX, PDF; boş/bozuk dosyada anlaşılır hata.
* Atıf: aynı adlı farklı uzantılı belgeler (tr.pdf / tr.docx) birbirine karışmaz.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import types
from pathlib import Path

import pytest

from src.config import Settings
from src.document_loader import DocumentProcessingError
from src.llm import GeminiLLM, LLMError, friendly_error
from src.local_engine import LocalWorkspace

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SAMPLE, EMPTY = "Örnek Holding A.Ş.", "Ege Vadi Gıda A.Ş."
OVERTIME = "Fazla mesai ücreti nasıl hesaplanır?"


def make_ws(samples: bool = False) -> LocalWorkspace:
    ws = LocalWorkspace(Settings(_env_file=None), api_key=None)
    if samples:
        ws.load_sample_documents()
    return ws


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


@pytest.fixture
def restore_core_modules():
    """Modül tahliyesini taklit eden testlerden sonra sys.modules'u eski haline getirir; aksi halde sonraki
    testler farklı sınıf kimlikleriyle (ör. iki ayrı LLMError) çalışır."""
    saved = {k: v for k, v in sys.modules.items() if k in ("src", "ui") or k.startswith(("src.", "ui."))}
    yield
    for k in [k for k in sys.modules if k in ("src", "ui") or k.startswith(("src.", "ui."))]:
        del sys.modules[k]
    sys.modules.update(saved)


def page(at) -> str:  # noqa: ANN001
    return " ".join(m.value for m in at.markdown)


# ====================================================================== import / sıcak yenileme
def test_no_runtime_src_imports_inside_functions():
    """Fonksiyon içinden `from src…` yapmak, Streamlit modülleri boşaltırken `KeyError: 'src'` üretir."""
    allowed = {("rag_engine.py", "src.embeddings"), ("rag_engine.py", "src.llm"), ("rag_engine.py", "src.vector_store")}
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"):
                    if (path.name, node.module) not in allowed:  # yalnızca Chroma fabrika yolu
                        offenders.append(f"{path.name}:{node.lineno} {node.module}")
    assert not offenders, offenders


def test_app_recovers_from_half_evicted_src_package(app_test, restore_core_modules):
    sys.modules.pop("src", None)  # eski Streamlit: ebeveyn silindi, alt modüller kaldı
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert "src" in sys.modules


def test_app_replaces_foreign_src_package(app_test, monkeypatch, restore_core_modules):
    foreign = types.ModuleType("src")
    foreign.__file__ = "/site-packages/src/__init__.py"  # başka bir kütüphanenin üst düzey `src` paketi
    monkeypatch.setitem(sys.modules, "src", foreign)
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert Path(sys.modules["src"].__file__).resolve().is_relative_to(ROOT)


def test_stale_workspace_from_old_code_is_rebuilt(app_test):
    class OldWorkspace:  # sıcak yenilemeden önceki sınıfın örneği gibi
        pass

    app_test.session_state["workspaces"][SAMPLE] = OldWorkspace()
    app_test.run()
    assert not app_test.exception, app_test.exception
    assert type(app_test.session_state["workspaces"][SAMPLE]).__name__ == "LocalWorkspace"


# ====================================================================== model adları
def test_no_retired_gemini_model_names_left():
    pattern = re.compile(r"gemini-2\.[05]-flash")
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".py", ".toml", ".example", ".yml", ".yaml"}
             and ".git" not in p.parts and ".venv" not in p.parts and "node_modules" not in p.parts]
    files += [ROOT / ".env.example"]
    hits = [f"{p.relative_to(ROOT)}" for p in files if p.exists() and pattern.search(p.read_text(encoding="utf-8"))]
    assert not hits, hits
    s = Settings(_env_file=None)
    assert (s.llm_model, s.fast_llm_model) == ("gemini-3.5-flash", "gemini-3.5-flash-lite")


# ====================================================================== Gemini hata/tekrar politikası
class _APIError(Exception):
    def __init__(self, code: int, message: str = "boom"):
        super().__init__(f"{code} {message}")
        self.code = code


@pytest.fixture
def fake_genai(monkeypatch):
    """`google.genai` kurulu olmasa da GeminiLLM'i test etmek için en küçük taklit."""
    genai = types.ModuleType("google.genai")
    errors = types.ModuleType("google.genai.errors")
    errors.APIError = _APIError
    gtypes = types.ModuleType("google.genai.types")
    for name in ("GenerateContentConfig", "Content", "Part", "EmbedContentConfig"):
        setattr(gtypes, name, lambda *a, **k: (a, k))
    genai.errors, genai.types = errors, gtypes
    genai.Client = lambda api_key: types.SimpleNamespace(models=types.SimpleNamespace())
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", genai, raising=False)
    for name, mod in (("google.genai", genai), ("google.genai.errors", errors), ("google.genai.types", gtypes)):
        monkeypatch.setitem(sys.modules, name, mod)
    return genai


def _llm(script: list) -> tuple[GeminiLLM, dict]:
    """script: her çağrı için ya bir istisna ya da üretilecek parça listesi."""
    llm = GeminiLLM("test-key")
    llm.retry_base_delay = 0.0
    calls = {"stream": 0, "json": 0}

    def stream(**_):
        step = script[calls["stream"]]
        calls["stream"] += 1
        for item in step:
            if isinstance(item, Exception):
                raise item
            yield types.SimpleNamespace(text=item)

    def generate(**_):
        step = script[calls["json"]]
        calls["json"] += 1
        if isinstance(step, Exception):
            raise step
        return types.SimpleNamespace(text=step)

    llm._client.models.generate_content_stream = stream
    llm._client.models.generate_content = generate
    return llm, calls


def test_stream_retries_transient_503_before_first_chunk(fake_genai):
    llm, calls = _llm([[_APIError(503)], [_APIError(429)], ["Merhaba ", "dünya"]])
    assert "".join(llm.stream([("user", "soru")])) == "Merhaba dünya"
    assert calls["stream"] == 3


def test_stream_does_not_retry_after_output_started(fake_genai):
    llm, calls = _llm([["Yarım ", _APIError(503)], ["tekrar"]])
    out: list[str] = []
    with pytest.raises(LLMError) as exc:
        for tok in llm.stream([("user", "soru")]):
            out.append(tok)
    assert out == ["Yarım "] and calls["stream"] == 1 and exc.value.code == 503


def test_stream_gives_up_with_turkish_message_after_retries(fake_genai):
    llm, calls = _llm([[_APIError(429)]] * 3)
    with pytest.raises(LLMError) as exc:
        list(llm.stream([("user", "soru")]))
    assert calls["stream"] == 3 and exc.value.code == 429 and "istek sınırı" in exc.value.user_message


def test_permanent_errors_are_not_retried(fake_genai):
    llm, calls = _llm([[_APIError(404)], ["olmamalı"]])
    with pytest.raises(LLMError) as exc:
        list(llm.stream([("user", "soru")]))
    assert calls["stream"] == 1 and "LLM_MODEL" in exc.value.user_message


def test_network_errors_are_retried(fake_genai):
    llm, calls = _llm([[ConnectionError("reset")], ["tamam"]])
    assert list(llm.stream([("user", "soru")])) == ["tamam"] and calls["stream"] == 2


def test_json_generation_retries_503(fake_genai):
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    llm, calls = _llm([_APIError(503), '{"ok": true}'])
    assert llm.generate_json("p", Out).ok is True and calls["json"] == 2


@pytest.mark.parametrize("code,needle", [(429, "429"), (503, "yoğun"), (401, "API anahtarı"), (None, "bağlanılamadı")])
def test_friendly_messages_are_turkish(code, needle):
    assert needle in friendly_error(code)


class _FailingLLM:
    def stream(self, messages, **_):  # noqa: ANN001, ANN003, ANN201
        raise LLMError("upstream 503", code=503)
        yield  # pragma: no cover

    def generate_json(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        raise LLMError("upstream 503", code=503)


def _failing_ws() -> LocalWorkspace:
    ws = make_ws(samples=True)
    ws.engine.llm = _FailingLLM()
    return ws


def test_service_error_is_not_reported_as_missing_information():
    ws = _failing_ws()
    turn = ws.run(OVERTIME)
    r = turn.response
    assert r.error and "yoğun" in r.error and not r.is_answered
    assert turn.audit.status == "skipped"  # "Kaynakta bulunamadı" kararı verilmez
    assert "yer almamaktadır" not in r.answer and len(ws.history) == 1


def test_service_error_in_compound_question_does_not_become_scope_note():
    ws = _failing_ws()
    turn = ws.run("Fazla mesai ücreti nasıl hesaplanır ve şirketin Mars'taki ofisinin adresi nedir?")
    assert turn.response.error and not turn.insight.parts


def test_service_error_shows_retry_in_ui(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    monkeypatch.setattr(ws.engine, "llm", _FailingLLM())
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    html = page(at)
    assert "Yanıt şu anda üretilemedi" in html and "Kaynakta Bulunamadı" not in html
    retry = next(b for b in at.button if b.key and b.key.startswith("again_"))
    monkeypatch.setattr(ws.engine, "llm", ws.llm)  # servis düzeldi
    retry.click().run()
    assert not at.exception, at.exception
    assert at.session_state["chats"][SAMPLE][-1]["response"].is_answered


# ====================================================================== arayüz kenar durumları
def test_empty_tenant_query_shows_clear_turkish_hint(app_test):
    at = app_test
    at.sidebar.selectbox[0].set_value(EMPTY).run()
    at.chat_input[0].set_value("Yıllık izin kaç gün?").run()
    assert not at.exception, at.exception
    assert any("bilgi tabanında henüz belge yok" in i.value for i in at.info)


def test_demo_from_empty_tenant_switches_to_sample(app_test):
    at = app_test
    at.sidebar.selectbox[0].set_value(EMPTY).run()
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    assert at.session_state["tenant"] == SAMPLE and len(at.session_state["chats"][SAMPLE]) == 12
    assert at.session_state["chats"].get(EMPTY, []) == []


def test_upload_on_empty_tenant_then_answer():
    ws = make_ws()
    doc, added = ws.add_bytes("Güvenlik.md", "# Güvenlik\n\nZiyaretçiler resepsiyonda kimlik bırakır ve refakatçiyle "
                                             "dolaşır; kayıt 2 yıl saklanır.".encode(), "Genel")
    assert added and ws.ask("Ziyaretçiler kimliği nereye bırakır?").is_answered


def test_full_tenant_flows_do_not_crash(app_test):
    at = app_test
    at.sidebar.multiselect[0].set_value(["Finans"]).run()
    at.chat_input[0].set_value("Satın almada kaç teklif almak gerekir?").run()
    toggle = next(t for t in at.toggle if "Denetçi" in t.label)
    toggle.set_value(False).run()
    at.chat_input[0].set_value("Uzaktan çalışma kuralları nelerdir?").run()
    toggle.set_value(True).run()
    assert not at.exception, at.exception
    assert len(at.session_state["chats"][SAMPLE]) == 4
    # Analitik ve Doküman Kütüphanesi sekmeleri aynı çalıştırmada çizilir
    assert any("Doküman Kütüphanesi" in t.label for t in at.tabs)
    assert len([b for b in at.button if b.key and b.key.startswith("del_")]) == 3
    next(b for b in at.button if b.label.startswith("🧹")).click().run()
    assert not at.exception and at.session_state["chats"][SAMPLE] == []


def test_upload_button_without_files_warns(app_test):
    at = app_test
    form_submit = next(b for b in at.button if b.label == "İndeksle")
    form_submit.click().run()
    assert not at.exception, at.exception
    assert any("dosya seçin" in w.value for w in at.warning)


# ====================================================================== belge yükleme
TR_TEXT = ("# Şirket Güvenlik Yönetmeliği\n\nÇalışanlar ofise giriş kartını her gün yanında taşımalıdır. Kayıp kart "
           "İK birimine 24 saat içinde bildirilir ve yenisi ücretsiz verilir.\n")


@pytest.mark.parametrize("name,encoding", [("a.md", "utf-8"), ("b.txt", "utf-8-sig"), ("c.txt", "cp1254"),
                                            ("d.txt", "utf-16"), ("e.md", "utf-16-be")])
def test_turkish_text_encodings(name, encoding):
    data = TR_TEXT.encode(encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data
    doc, _ = make_ws().add_bytes(name, data, "Genel")
    text = doc.chunks[0].text
    assert "Şirket Güvenlik Yönetmeliği" in text and "Çalışanlar" in text and "\x00" not in text


@pytest.mark.parametrize("name,data,needle", [
    ("bos.txt", b"", "boş bir dosya"),
    ("bosluk.md", b"   \n\t ", "yalnızca boşluk"),
    ("bozuk.docx", b"zip degil", "DOCX okunamadı"),
    ("bozuk.xlsx", b"zip degil", "XLSX okunamadı"),
    ("bozuk.pdf", b"%PDF-1.4 bozuk", "PDF okunamadı"),
    ("ikili.txt", b"abc\x00\x01\x02def" * 10, "ikili"),
])
def test_bad_uploads_raise_clear_errors(name, data, needle):
    with pytest.raises(DocumentProcessingError, match=needle):
        make_ws().add_bytes(name, data, "Genel")


def test_turkish_docx_upload():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("İzin Prosedürü", 1)
    d.add_paragraph("Yıllık izin talepleri en az iki hafta önce İK Portalı üzerinden yapılır; şef onayı gerekir.")
    buf = io.BytesIO()
    d.save(buf)
    ws = make_ws()
    doc, _ = ws.add_bytes("İzin_Prosedürü.docx", buf.getvalue(), "İnsan Kaynakları")
    assert "İzin Prosedürü" in doc.chunks[0].text
    assert ws.ask("Yıllık izin talebi ne zaman yapılır?").citations[0].source == "İzin_Prosedürü.docx"


def _turkish_pdf(lines: list[str]) -> bytes:
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("Türkçe glif içeren TTF yazı tipi yok")
    pdfmetrics.registerFont(TTFont("DV", str(font)))
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFont("DV", 12)
    for i, line in enumerate(lines):
        c.drawString(72, 750 - 16 * i, line)
    c.save()
    return buf.getvalue()


def test_same_stem_different_extension_citations_do_not_collide():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Yıllık izin talepleri en az iki hafta önce İK Portalı üzerinden yapılır; şef onayı gerekir.")
    buf = io.BytesIO()
    d.save(buf)
    ws = make_ws()
    ws.add_bytes("tr.docx", buf.getvalue(), "Genel")
    ws.add_bytes("tr.pdf", _turkish_pdf(["Araç Kullanım Politikası: şehir dışı görevlerde",
                                         "kilometre başına 8 TL yakıt gideri ödenir ve fiş zorunludur."]), "Genel")
    r = ws.ask("Kilometre başına yakıt gideri ne kadar?")
    assert r.is_answered and [c.source for c in r.citations] == ["tr.pdf"]


def test_blank_pdf_is_rejected_with_scan_hint():
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    buf = io.BytesIO()
    canvas.Canvas(buf).save()
    with pytest.raises(DocumentProcessingError, match="taranmış"):
        make_ws().add_bytes("tarama.pdf", buf.getvalue(), "Genel")
