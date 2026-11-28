""""Bunu mu sormak istediniz?" — kurumsal bağlama duyarlı öneri motoru (birim) ve vitrin arayüzü (AppTest).

* İpucu (belge adları, departmanlar, başlıklar, dağarcık) YALNIZCA kullanıcının erişebildiği kapsamdan gelir.
* Net soru → model çağrılmaz; yazım hatası → tek düzeltme; belirsiz soru → 2–3 tıklanabilir niyet.
* Önbellek kapsam (tenant + departmanlar) anahtarlıdır: geniş yetkilinin önerisi dar yetkiliye dönmez.
* Arayüz: öneri ana aramayla eşzamanlı; öneriden/örnekten gelen soru tekrar denetlenmez (döngü yok).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.config import Settings
from src.llm import LLMError
from src.local_engine import LocalWorkspace
from src.suggestions import (
    EMPTY_HINT,
    NO_SUGGESTION,
    QuerySuggester,
    QuerySuggestion,
    _QuerySuggestionOut,
    analyze_query,
    build_corpus_hint,
    hint_from_index,
    same_question,
    suggest_query,
)

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SAMPLE = "Örnek Holding A.Ş."
TYPO = "fazla mesay ücreti nasıl hesplanır"
FIXED = "Fazla mesai ücreti nasıl hesaplanır?"
IK, FIN, HUKUK = "İnsan Kaynakları", "Finans", "Hukuk"
IK_DOC, FIN_DOC = "IK_Izin_ve_Calisma_Yonetmeligi.md", "Finans_Harcama_Proseduru.md"
CLEAN_QUESTIONS = [
    "Fazla mesai ücreti nasıl hesaplanır?",
    "Uzaktan çalışma kuralları nelerdir?",
    "Satın almada kaç teklif almak gerekir?",      # çekimli sözcükler ("almada", "almak") hata sayılmaz
    "Yurt içi seyahatte günlük yemek harcırahı ne kadar?",
    "Bakım sözleşmesinde P1 çözüm süresi aşılırsa hizmet kredisi ne kadar?",
    "Şirketin Mars'taki ofisinin adresi nedir?",    # özel isim + kesme işareti: düzeltilmez
    "yillik izin kac gun",                          # Türkçe karakter eksikliği öneri gerektirmez
]


class FakeSuggestLLM:
    """Yalnızca `generate_json` çağrılarını kaydeden sahte model; `reply` dict, istisna veya prompt → dict."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        with self.lock:
            self.calls.append({"prompt": prompt, "schema": schema, "system": system, "fast": fast})
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate(reply)

    def stream(self, messages, **_):
        yield "kullanılmamalı"


def _reply(suggested: str = "", alternatives: list[str] | None = None, has: bool = True) -> dict:
    return {"has_suggestion": has, "suggested_query": suggested, "alternatives": alternatives or []}


@pytest.fixture(scope="module")
def ws() -> LocalWorkspace:
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    w.load_sample_documents()
    return w


def _hint(ws: LocalWorkspace, departments: list[str] | None = None):
    return hint_from_index(ws.engine.retriever.index_for(departments))


# ====================================================================== kapsam ipucu
def test_hint_lists_documents_departments_and_clean_topics(ws):
    hint = _hint(ws)
    assert {IK_DOC, FIN_DOC} <= set(hint.documents) and {IK, FIN} <= set(hint.departments)
    assert "Yıllık Ücretli İzin" in hint.topics and "Masraf Beyanı ve İadesi" in hint.topics
    # Başlık numarası ve kurum öneki ("Örnek Holding A.Ş. — …") ayıklanır
    assert not [t for t in hint.topics if t[:1].isdigit() or "Holding" in t or "—" in t]
    rendered = hint.render()
    assert IK_DOC in rendered and "Konu başlıkları:" in rendered and len(rendered) <= 2000
    assert len(hint.render(max_chars=120)) <= 120


def test_hint_is_built_only_from_the_given_scope(ws):
    fin = _hint(ws, [FIN])
    assert fin.documents == (FIN_DOC,) and fin.departments == (FIN,)
    assert not [t for t in fin.topics if "İzin" in t or "Çalışma" in t]
    assert "izin" not in fin.words  # İK dağarcığı Finans kapsamına sızmaz
    ik = _hint(ws, [IK])
    assert ik.documents == (IK_DOC,) and "Masraf Beyanı ve İadesi" not in ik.topics
    assert len({fin.fingerprint, ik.fingerprint, _hint(ws).fingerprint}) == 3


def test_hint_is_cached_on_the_scoped_index(ws):
    index = ws.engine.retriever.index_for([IK])
    assert hint_from_index(index) is hint_from_index(index)
    assert build_corpus_hint([]) is EMPTY_HINT and EMPTY_HINT.empty


def test_hint_refreshes_when_documents_change():
    w = LocalWorkspace(Settings(_env_file=None), api_key=None)
    assert _hint(w).empty and w.suggest("izin") == NO_SUGGESTION
    w.load_sample_documents()
    assert not _hint(w).empty and w.suggest("izin").has_suggestion


# ====================================================================== ön analiz (LLM'siz)
@pytest.mark.parametrize("question", CLEAN_QUESTIONS)
def test_clear_questions_skip_the_model(ws, question):
    llm = FakeSuggestLLM(_reply(FIXED))
    out = QuerySuggester(llm).suggest(question, _hint(ws), scope=("t",))
    assert out == NO_SUGGESTION and out.method == "skipped" and not llm.calls


@pytest.mark.parametrize(("question", "expected"), [
    ("yıllık izn kaç gün", "yıllık izin kaç gün"),
    ("fazla mesay ücreti nasıl hesaplanır", "fazla mesai ücreti nasıl hesaplanır"),
    ("Harcrah limiti nedir", "Harcırah limiti nedir"),
])
def test_offline_typo_correction_uses_document_vocabulary(ws, question, expected):
    out = QuerySuggester(None).suggest(question, _hint(ws))
    assert out.has_suggestion and out.kind == "typo" and out.method == "heuristic"
    assert out.suggested_query == expected and out.options == [expected]


def test_analysis_flags_unknown_words_but_not_inflections(ws):
    hint = _hint(ws)
    a = analyze_query("Satın almada kaç teklif almak gerekir?", hint)
    assert a.clean and not a.unknown and a.has_cue
    b = analyze_query("izn", hint)
    assert b.ambiguous and b.corrections == {"izn": "izin"}
    c = analyze_query("Şirketin Mars'taki ofisi nerede?", hint)
    assert "mars" in c.unknown and not c.corrections  # özel isim "maaş"a düzeltilmez
    d = analyze_query("izin", _hint(ws, [FIN]))
    assert not d.corrections  # kapsamda olmayan gerçek sözcük benzer sözcüğe ("için") çevrilmez


@pytest.mark.parametrize(("question", "topic_word"), [
    ("izin", "İzin"), ("yıllık izn", "İzin"), ("hastalık izni", "Hastalık"),
    ("fatura", None), ("masraf formu", "Masraf"), ("teklif kurallari", "Teklif"),
])
def test_ambiguous_queries_get_2_to_3_intent_buttons(ws, question, topic_word):
    out = QuerySuggester(None).suggest(question, _hint(ws))
    assert out.has_suggestion and out.kind == "ambiguous"
    assert 2 <= len(out.options) <= 3 and out.suggested_query == out.options[0]
    assert len({o.casefold() for o in out.options}) == len(out.options)
    assert all(o.endswith("?") and len(o) <= 200 for o in out.options)
    if topic_word:
        assert topic_word in out.options[0]


@pytest.mark.parametrize(("question", "department"), [("ik prosedürü", IK), ("İK", IK), ("finans", FIN)])
def test_department_mentions_narrow_intents_to_that_department(ws, question, department):
    hint = _hint(ws)
    out = QuerySuggester(None).suggest(question, hint)
    assert out.kind == "ambiguous" and len(out.options) >= 2
    dept_topics = {t for t, d in hint.topic_department.items() if d == department}
    assert all(any(o.startswith(t) for t in dept_topics) for o in out.options), out.options


def test_intent_options_never_leave_the_users_scope(ws):
    fin_topics = set(_hint(ws, [FIN]).topics)
    for q in ("izin", "yıllık izn", "fatura", "masraf formu", "prosedür"):
        out = QuerySuggester(None).suggest(q, _hint(ws, [FIN]), scope=("t", FIN))
        assert not [o for o in out.options if "zin" in o or "Çalışma" in o], (q, out.options)
        for option in out.options:
            assert any(option.startswith(t) for t in fin_topics), option
            assert not option.startswith(("Amaç", "Genel İlkeler", "Yaptırımlar"))  # genel bölümler niyet değil
    assert QuerySuggester(None).suggest("izin", _hint(ws, [FIN])) == NO_SUGGESTION


# ====================================================================== model katmanı
def test_typo_question_gets_single_document_aligned_fix(ws):
    llm = FakeSuggestLLM(_reply(FIXED))
    out = QuerySuggester(llm).suggest(TYPO, _hint(ws), scope=("t",))
    assert out == QuerySuggestion(True, FIXED, (), "typo") and out.method == "llm"
    assert out.to_dict() == {"has_suggestion": True, "suggested_query": FIXED, "alternatives": [], "kind": "typo"}
    call = llm.calls[0]
    assert call["fast"] is True and call["schema"] is _QuerySuggestionOut
    # Model kurum bağlamını ve ön analiz sinyallerini görür
    prompt = call["prompt"]
    assert TYPO in prompt and IK_DOC in prompt and "Fazla Çalışma (Fazla Mesai)" in prompt
    assert "mesay → mesai" in prompt and "hesplanır" in prompt
    assert "YANITLAMA" in call["system"] and "EKLEME" in call["system"]


def test_ambiguous_question_gets_multiple_intents_from_model(ws):
    alts = ["Yıllık izin kaç gündür?", "Hastalık izni nasıl kullanılır?", "Mazeret izni hangi durumlarda verilir?"]
    llm = FakeSuggestLLM(_reply(alts[0], alts))
    out = QuerySuggester(llm).suggest("izin", _hint(ws))
    assert out.kind == "ambiguous" and out.options == alts and out.suggested_query == alts[0]
    assert "birden fazla niyete karşılık gelebilir" in llm.calls[0]["prompt"]


def test_model_options_are_deduplicated_and_capped(ws):
    alts = ["Yıllık izin kaç gündür?", "yıllık izin kaç gündür", "Hastalık izni?", "Mazeret izni?", "Doğum izni?",
            "x" * 201, "  ", "izin"]
    out = QuerySuggester(FakeSuggestLLM(_reply("Yıllık izin kaç gündür?", alts))).suggest("izin", _hint(ws))
    assert out.options == ["Yıllık izin kaç gündür?", "Hastalık izni?", "Mazeret izni?"]


@pytest.mark.parametrize("suggested", [
    "fazla mesay ücreti nasıl hesplanır",   # özgün soruyla aynı
    "FAZLA MESAY ÜCRETİ NASIL HESPLANIR?",  # yalnızca büyük/küçük harf ve noktalama farkı
    "fazla mesay ucreti nasil hesplanir",   # yalnızca Türkçe karakter farkı
    "   ",
    "x" * 201,
])
def test_trivial_or_invalid_model_suggestions_are_suppressed(ws, suggested):
    out = QuerySuggester(FakeSuggestLLM(_reply(suggested))).suggest(TYPO, _hint(ws))
    assert out == NO_SUGGESTION


def test_model_says_no_suggestion(ws):
    out = QuerySuggester(FakeSuggestLLM(_reply(has=False))).suggest(TYPO, _hint(ws))
    assert out == NO_SUGGESTION and out.method == "llm"


def test_out_of_scope_filenames_are_never_suggested(ws):
    fin = _hint(ws, [FIN])
    leak = FakeSuggestLLM(_reply(f"{IK_DOC} belgesine göre yıllık izin kaç gün?"))
    assert QuerySuggester(leak).suggest("izin", fin, scope=("t", FIN)) == NO_SUGGESTION
    ok = FakeSuggestLLM(_reply(f"{FIN_DOC} belgesine göre fatura onayı nasıl yapılır?"))
    assert QuerySuggester(ok).suggest("fatura", fin, scope=("t", FIN)).has_suggestion


@pytest.mark.parametrize("error", [LLMError("upstream 503", code=503), ValueError("bozuk JSON"),
                                   LLMError("şema", retryable=False)])
def test_model_errors_fall_back_to_heuristics_and_are_not_cached(ws, error):
    llm = FakeSuggestLLM(error)
    sug = QuerySuggester(llm)
    first = sug.suggest("yıllık izn kaç gün", _hint(ws), scope=("t",))
    assert first.suggested_query == "yıllık izin kaç gün" and first.method == "heuristic"
    sug.suggest("yıllık izn kaç gün", _hint(ws), scope=("t",))
    assert len(llm.calls) == 2  # geçici hata önbelleğe alınmaz: model tekrar denenir


def test_unexpected_errors_never_propagate(ws):
    out = QuerySuggester(FakeSuggestLLM(RuntimeError("beklenmeyen"))).suggest(TYPO, _hint(ws))
    assert out.method == "heuristic" and out.suggested_query == "fazla mesai ücreti nasıl hesplanır"
    assert QuerySuggester(None).suggest(TYPO, None) == NO_SUGGESTION  # type: ignore[arg-type]


def test_disabled_empty_long_or_unscoped_questions_skip_the_model(ws):
    llm = FakeSuggestLLM(_reply(FIXED))
    assert QuerySuggester(llm, enabled=False).suggest(TYPO, _hint(ws)).method == "disabled"
    sug = QuerySuggester(llm)
    assert sug.suggest("   ", _hint(ws)) == NO_SUGGESTION
    assert sug.suggest("izn " * 80, _hint(ws)) == NO_SUGGESTION  # > 300 karakter
    assert sug.suggest(TYPO, EMPTY_HINT) == NO_SUGGESTION       # erişilebilir belge yok
    assert suggest_query(TYPO, llm) == NO_SUGGESTION            # ipucusuz yardımcı da öneri vermez
    assert not llm.calls


def test_turkish_text_and_special_characters_are_safe(ws):
    llm = FakeSuggestLLM(_reply("İşe giriş ve çıkış prosedürü nedir? <b>x</b> `kod` **kalın**"))
    out = QuerySuggester(llm).suggest("işe girş çıkş prosedürü ğüşiöç", _hint(ws))
    assert out.has_suggestion and "<b>" in out.suggested_query  # kaçış arayüzün işidir; metin korunur
    assert same_question("İŞE GİRİŞ", "ise giris") and not same_question("izin", "izn")


# ====================================================================== önbellek ve kapsam yalıtımı
def test_cache_hits_within_scope_and_isolates_scopes(ws):
    prompts: list[str] = []

    def reply(prompt: str) -> dict:
        prompts.append(prompt)
        return _reply("Yıllık izin kaç gündür?" if IK_DOC in prompt else "Masraf beyanı nasıl yapılır?")

    sug = QuerySuggester(FakeSuggestLLM(reply))
    wide = sug.suggest("izn", _hint(ws), scope=("tenant-1", "*"))
    again = sug.suggest("İZN", _hint(ws), scope=("tenant-1", "*"))
    assert again.method == "cache" and again == wide and len(prompts) == 1
    narrow = sug.suggest("izn", _hint(ws, [FIN]), scope=("tenant-1", FIN))
    assert len(prompts) == 2 and IK_DOC not in prompts[1]  # dar kapsam kendi ipucuyla, yeniden hesaplanır
    assert narrow.suggested_query != wide.suggested_query
    other = sug.suggest("izn", _hint(ws), scope=("tenant-2", "*"))
    assert other.method == "llm" and len(prompts) == 3  # başka tenant aynı soruyu önbellekten almaz


def test_cache_is_bounded_and_thread_safe(ws):
    sug = QuerySuggester(FakeSuggestLLM(lambda p: _reply("Yıllık izin kaç gündür?")), cache_size=8)
    hint = _hint(ws)
    questions = [f"izn {chr(97 + i % 26)}{i}" for i in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: sug.suggest(q, hint, scope=("t",)), questions))
    assert all(isinstance(r, QuerySuggestion) for r in results) and len(sug._cache) <= 8
    sug.clear()
    assert not sug._cache


def test_response_schema_requires_every_field_without_defaults():
    schema = _QuerySuggestionOut.model_json_schema()
    assert set(schema["required"]) == {"has_suggestion", "suggested_query", "alternatives"}
    assert "default" not in str(schema)


# ====================================================================== motor / çalışma alanı
def test_offline_workspace_uses_heuristic_suggester(ws):
    assert ws.settings.query_suggestions is True and ws.engine.suggester.llm is None
    assert ws.suggest("yıllık izn kaç gün").suggested_query == "yıllık izin kaç gün"
    assert not [o for o in ws.suggest("izin", [FIN]).options if "İzin" in o]


def test_workspace_suggest_respects_setting():
    w = LocalWorkspace(Settings(_env_file=None, QUERY_SUGGESTIONS=False), api_key=None)
    w.load_sample_documents()
    assert w.suggest("izin") == NO_SUGGESTION and w.suggest("izin").method == "disabled"


def test_workspace_engine_uses_model_when_available(ws, monkeypatch):
    llm = FakeSuggestLLM(_reply(FIXED))
    monkeypatch.setattr(ws.engine, "suggester", QuerySuggester(llm))
    assert ws.suggest(TYPO, [IK]).suggested_query == FIXED
    assert FIN_DOC not in llm.calls[0]["prompt"]  # yalnızca İK kapsamı modele gider


def test_suggestions_run_concurrently_with_retrieval(ws):
    """Öneri ayrı iş parçacığında aramayla aynı anda çalışabilir (BM25 önbelleği kilitli)."""
    questions = ["izin", "fatura", TYPO, "masraf formu"] * 6
    with ThreadPoolExecutor(max_workers=4) as pool:
        sug = [pool.submit(ws.suggest, q, [IK, FIN]) for q in questions]
        turns = [ws.run(q, (), [IK, FIN]) for q in CLEAN_QUESTIONS[:3]]  # ana akış (Streamlit iş parçacığı)
        assert all(isinstance(f.result(timeout=60), QuerySuggestion) for f in sug)
    assert all(t.response.answer for t in turns)


# ====================================================================== arayüz (AppTest)
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


def _boxes(at) -> list[str]:
    return [i.value.replace("\\", "") for i in at.info if "Bunu mu sormak istediniz" in i.value]


def _suggest_buttons(at) -> list:
    return [b for b in at.button if b.key and b.key.startswith("suggest_")]


def _user_questions(at) -> list[str]:
    return [m["content"] for m in at.session_state["chats"][SAMPLE] if m["role"] == "user"]


def _patch_suggest(monkeypatch, ws, answers: dict[str, QuerySuggestion], *, delay: float = 0.0) -> list:
    asked: list[tuple[str, list, str]] = []

    def fake(question, departments=None):
        asked.append((question, list(departments or []), threading.current_thread().name))
        if delay:
            time.sleep(delay)
        return answers.get(question, NO_SUGGESTION)

    monkeypatch.setattr(ws, "suggest", fake)
    return asked


def test_ui_typo_suggestion_reasks_once_without_loop(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    asked = _patch_suggest(monkeypatch, ws, {TYPO: QuerySuggestion(True, FIXED, (), "typo"),
                                             FIXED: QuerySuggestion(True, "Başka bir öneri?", (), "typo")})
    at.chat_input[0].set_value(TYPO).run()
    assert not at.exception, at.exception
    boxes = _boxes(at)
    assert len(boxes) == 1 and "Fazla mesai ücreti nasıl hesaplanır" in boxes[0] and boxes[0].endswith("?")
    assert not boxes[0].endswith("??")
    assert [q for q, *_ in asked] == [TYPO]
    assert asked[0][2].startswith("oneri")  # ana akıştan ayrı iş parçacığında
    chat = at.session_state["chats"][SAMPLE]
    assert chat[-1]["suggestion"].has_suggestion and chat[-1]["response"] is not None

    buttons = _suggest_buttons(at)
    assert len(buttons) == 1 and "Önerilen soruyla sor" in buttons[0].label
    buttons[0].click().run()
    assert not at.exception, at.exception
    assert _user_questions(at) == [TYPO, FIXED]
    chat = at.session_state["chats"][SAMPLE]
    # Önerilen soru kabul edilmiş sayılır: yeniden denetlenmez, yeni kutu/buton çıkmaz
    assert [q for q, *_ in asked] == [TYPO] and not chat[-1]["suggestion"].has_suggestion
    assert chat[-1]["response"].is_answered
    assert not _suggest_buttons(at)
    # Eski mesajın kutusu metin olarak kalır ama butonsuzdur (tekrar tıklanamaz)
    assert len(_boxes(at)) == 1


def test_ui_ambiguous_question_offers_intent_buttons(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    alts = ("Yıllık Ücretli İzin nelerdir?", "Hastalık İzni ile ilgili kurallar nelerdir?",
            "Mazeret ve Özel İzinler ile ilgili kurallar nelerdir?")
    asked = _patch_suggest(monkeypatch, ws, {"izin": QuerySuggestion(True, alts[0], alts, "ambiguous")})
    at.chat_input[0].set_value("izin").run()
    assert not at.exception, at.exception
    assert len(_boxes(at)) == 1 and "birden fazla konuya" in _boxes(at)[0]
    buttons = _suggest_buttons(at)
    assert [b.label for b in buttons] == list(alts)
    buttons[1].click().run()
    assert not at.exception, at.exception
    assert _user_questions(at) == ["izin", alts[1]] and len(asked) == 1
    assert not _suggest_buttons(at)
    # Geçmişteki çoklu öneri, butonsuz liste olarak görünür
    box = _boxes(at)[0]
    assert all(a in box for a in alts)


def test_ui_real_engine_suggestions_end_to_end(app_test):
    """Taklitsiz: çevrimdışı motor belirsiz soruya belge başlıklarından niyet butonları üretir."""
    at = app_test
    at.chat_input[0].set_value("masraf formu").run()
    assert not at.exception, at.exception
    buttons = _suggest_buttons(at)
    assert 2 <= len(buttons) <= 3 and "Masraf" in buttons[0].label
    buttons[0].click().run()
    assert not at.exception, at.exception
    chat = at.session_state["chats"][SAMPLE]
    assert chat[-1]["response"].is_answered and not chat[-1]["suggestion"].has_suggestion


def test_ui_passes_department_filter_to_suggestions(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    asked = _patch_suggest(monkeypatch, ws, {})
    at.session_state["dept_filter"] = [FIN]
    at.run()
    at.chat_input[0].set_value("fatura onayı").run()
    assert not at.exception, at.exception
    assert asked and asked[-1][1] == [FIN]


def test_ui_sample_and_demo_questions_skip_suggestion(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    asked = _patch_suggest(monkeypatch, ws, {})
    at.button(key="sample_0").click().run()
    assert not at.exception, at.exception
    assert not asked and not _boxes(at) and not _suggest_buttons(at)


def test_ui_slow_suggestion_never_delays_the_answer(app_test, monkeypatch):
    at = app_test
    at.session_state["suggest_wait"] = 0.05
    ws = at.session_state["workspaces"][SAMPLE]
    _patch_suggest(monkeypatch, ws, {TYPO: QuerySuggestion(True, FIXED, (), "typo")}, delay=3.0)
    t0 = time.perf_counter()
    at.chat_input[0].set_value(TYPO).run()
    elapsed = time.perf_counter() - t0
    assert not at.exception, at.exception
    chat = at.session_state["chats"][SAMPLE]
    assert chat[-1]["response"] is not None and not chat[-1]["suggestion"].has_suggestion
    assert not _boxes(at) and elapsed < 3.0


def test_ui_survives_failing_or_stale_suggest(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    monkeypatch.setattr(ws, "suggest", lambda q: QuerySuggestion(True, FIXED))  # eski imza (sıcak yenileme)
    at.chat_input[0].set_value(TYPO).run()
    assert not at.exception, at.exception
    monkeypatch.setattr(ws, "suggest", lambda q, d=None: (_ for _ in ()).throw(RuntimeError("x")))
    at.chat_input[0].set_value("izin nedir").run()
    assert not at.exception, at.exception
    assert len(_user_questions(at)) == 2 and not _boxes(at)


def test_ui_escapes_markdown_in_suggestions(app_test, monkeypatch):
    at = app_test
    ws = at.session_state["workspaces"][SAMPLE]
    _patch_suggest(monkeypatch, ws, {TYPO: QuerySuggestion(True, "**kalın** [bağlantı](http://x) <b>?", (), "typo")})
    at.chat_input[0].set_value(TYPO).run()
    assert not at.exception, at.exception
    raw = next(i.value for i in at.info if "Bunu mu sormak istediniz" in i.value)
    assert "\\*\\*kalın\\*\\*" in raw and "\\[bağlantı\\]" in raw and "\\<b\\>" in raw


def test_ui_chat_input_limits_question_length(app_test):
    assert app_test.chat_input[0].proto.max_chars == 2000
