""""Bunu mu sormak istediniz?" — REST API + RBAC (PGlite üzerinde gerçek PostgreSQL + RLS).

* Öneri ipucu yalnızca kullanıcının yetkili olduğu departmanların belgelerinden kurulur: yetkisiz departmanın
  belge adı, başlığı veya içeriği ne modele giden istemde ne de yanıttaki öneride görünür.
* Önbellek kapsam anahtarlıdır: tüm departmanlara yetkili sahibin önbelleğe aldığı öneri İK kullanıcısına dönmez.
* Başka şirketin belgeleri hiçbir koşulda ipucuna girmez.
* `suggest=false` (önerilen soruyla tekrar sorma) ve departmansız kullanıcıda öneri üretilmez; net soruda model
  çağrılmaz; yavaş öneri yanıtı geciktirmez.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

API = "/api/v1"
PW = "Guclu-Parola-2026!"
DEPTS = ["İK", "Finans", "Hukuk"]
SUFFIX = uuid.uuid4().hex[:6]
SCHEMA = "_QuerySuggestionOut"
IK_DOC, FIN_DOC, B_DOC = "IK_Izin_Yonetmeligi.md", "Finans_Butce_Gizli.md", "Rakip_Izin.md"
FIN_TOPIC, FIN_SECRET, B_SECRET, B_TOPIC = "Bütçe Onay Süreci", "FINGIZLI-4242", "RAKIPGIZLI-9090", "Rakip Kreş Yardımı"
IK_BODY = (
    "# İzin Yönetmeliği\n\n"
    "## 1. Yıllık İzin\n\nYıllık izin kıdeme göre 14 ile 26 iş günü arasındadır. İzin talebi portal üzerinden yapılır.\n\n"
    "## 2. Hastalık İzni\n\nHastalık izni için doktor raporu gerekir. Rapor iki gün içinde İK'ya iletilir.\n\n"
    "## 3. Mazeret İzni\n\nEvlilik için 3 gün, doğum için 5 gün mazeret izni verilir.\n"
)
FIN_BODY = (
    "# Bütçe Prosedürü\n\n"
    f"## 1. {FIN_TOPIC}\n\nYıllık bütçe CFO onayına sunulur. Fatura onayı Finans müdüründedir. Kod: {FIN_SECRET}.\n\n"
    "## 2. Fatura Kontrolü\n\nTedarikçi faturaları üç iş günü içinde kontrol edilir.\n"
)
B_BODY = f"# Rakip İzin\n\n## 1. {B_TOPIC}\n\nRakip çalışanlarına izin ve kreş yardımı verilir. Kod: {B_SECRET}.\n"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def register(client, slug: str, email: str, name: str) -> dict:
    r = client.post(
        f"{API}/auth/register",
        json={"company": {"name": name, "slug": slug, "departments": DEPTS}, "full_name": "Sahip Kişi",
              "email": email, "password": PW},
    )
    assert r.status_code == 201, r.text
    return r.json()


def join(client, owner: str, email: str, role: str, departments: list[str]) -> str:
    r = client.post(f"{API}/company/invitations", json={"email": email, "role": role, "departments": departments},
                    headers=bearer(owner))
    assert r.status_code == 201, r.text
    r = client.post(f"{API}/auth/accept-invitation",
                    json={"token": r.json()["token"], "email": email, "password": PW, "full_name": "Davetli Kişi"})
    assert r.status_code == 201, r.text
    return r.json()["token"]


def upload(client, token: str, filename: str, content: str, department: str) -> None:
    r = client.post(f"{API}/documents/upload", params={"wait": "true"},
                    files={"file": (filename, content.encode(), "application/octet-stream")},
                    data={"department": department}, headers=bearer(token))
    assert r.status_code == 201, r.text


def ask(client, token: str, question: str, **kw) -> dict:
    r = client.post(f"{API}/chat/query", json={"question": question, **kw}, headers=bearer(token))
    assert r.status_code == 200, r.text
    return r.json()


def suggestion_calls(llm) -> list[str]:
    return [p for name, p in list(llm.json_calls) if name == SCHEMA]


@pytest.fixture(scope="module")
def s(client) -> dict:
    a = register(client, f"oneri-a-{SUFFIX}", f"sahip-oa-{SUFFIX}@a.com", "Öneri Alfa A.Ş.")
    b = register(client, f"oneri-b-{SUFFIX}", f"sahip-ob-{SUFFIX}@b.com", "Öneri Beta A.Ş.")
    upload(client, a["token"], IK_DOC, IK_BODY, "İK")
    upload(client, a["token"], FIN_DOC, FIN_BODY, "Finans")
    upload(client, b["token"], B_DOC, B_BODY, "İK")
    return {
        "owner": a["token"], "b_owner": b["token"],
        "ik_user": join(client, a["token"], f"ik-oneri-{SUFFIX}@a.com", "user", ["İK"]),
        "fin_user": join(client, a["token"], f"fin-oneri-{SUFFIX}@a.com", "user", ["Finans"]),
        "no_dept": join(client, a["token"], f"bos-oneri-{SUFFIX}@a.com", "user", []),
    }


@pytest.fixture
def handler(llm):
    """`_QuerySuggestionOut` için betik; test sonunda kaldırılır (diğer testler sezgisel yedeği görür)."""

    def install(fn) -> None:
        llm.json_handlers[SCHEMA] = fn

    yield install
    llm.json_handlers.pop(SCHEMA, None)


def _out(suggested: str = "", alternatives: list[str] | None = None, has: bool = True) -> dict:
    return {"has_suggestion": has, "suggested_query": suggested, "alternatives": alternatives or []}


# ================================================================ sözleşme ve sezgisel yedek
def test_typo_suggestion_contract_with_heuristic_fallback(client, s):
    body = ask(client, s["ik_user"], "yıllık izn kaç gün")
    sug = body["suggestion"]
    assert sug == {"has_suggestion": True, "suggested_query": "yıllık izin kaç gün", "alternatives": [],
                   "kind": "typo"}
    assert body["departments"] == ["İK"]


def test_ambiguous_query_returns_intents_from_own_departments_only(client, s):
    sug = ask(client, s["ik_user"], "izin")["suggestion"]
    assert sug["kind"] == "ambiguous" and 2 <= len(sug["alternatives"]) <= 3
    assert sug["suggested_query"] == sug["alternatives"][0]
    joined = " ".join(sug["alternatives"])
    assert "İzin" in joined and FIN_TOPIC not in joined and "Fatura" not in joined and B_TOPIC not in joined


def test_clear_question_skips_the_model(client, s, llm, handler):
    handler(lambda p: _out("kullanılmamalı"))
    before = len(suggestion_calls(llm))
    body = ask(client, s["ik_user"], "Hastalık izni için doktor raporu gerekir mi?")
    assert body["suggestion"] is None and len(suggestion_calls(llm)) == before


def test_suggest_false_and_users_without_departments_get_no_suggestion(client, s, llm, handler):
    handler(lambda p: _out("Yıllık izin kaç gündür?"))
    before = len(suggestion_calls(llm))
    assert ask(client, s["ik_user"], "izn", suggest=False)["suggestion"] is None
    assert ask(client, s["no_dept"], "izn")["suggestion"] is None
    assert len(suggestion_calls(llm)) == before


# ================================================================ RBAC: istem ve yanıtta sızıntı yok
def test_model_prompt_contains_only_authorized_department_material(client, s, llm, handler):
    handler(lambda p: _out("Yıllık izin kaç gündür?"))
    ask(client, s["ik_user"], "bütçe")
    prompt = suggestion_calls(llm)[-1]
    assert IK_DOC in prompt and "Yıllık İzin" in prompt
    for leaked in (FIN_DOC, FIN_TOPIC, FIN_SECRET, "Fatura Kontrolü", "Finans", B_DOC, B_TOPIC, B_SECRET):
        assert leaked not in prompt, leaked

    ask(client, s["fin_user"], "izn")
    prompt = suggestion_calls(llm)[-1]
    assert FIN_DOC in prompt and IK_DOC not in prompt and "Hastalık İzni" not in prompt


def test_model_output_naming_unauthorized_documents_is_dropped(client, s, handler):
    handler(lambda p: _out(f"{FIN_DOC} belgesine göre bütçe kim onaylar?",
                           [f"{FIN_DOC} bütçe onayı nedir?", "Yıllık izin kaç gündür?", "Hastalık izni nasıl alınır?"]))
    sug = ask(client, s["ik_user"], "bütçe onayı")["suggestion"]
    assert sug is not None and FIN_DOC not in str(sug)
    assert sug["alternatives"] == ["Yıllık izin kaç gündür?", "Hastalık izni nasıl alınır?"]


def test_cached_suggestion_of_wider_scope_never_reaches_narrower_user(client, s, llm, handler):
    def reply(prompt: str) -> dict:
        if FIN_DOC in prompt:
            return _out(f"{FIN_TOPIC} için hangi onaylar gerekir?")
        return _out("Yıllık izin kaç gündür?")

    handler(reply)
    question = "onay süreci"  # kısa ve soru kalıbı yok → model denetler (önbellek süreç başına yenidir)
    owner = ask(client, s["owner"], question)["suggestion"]
    assert owner and FIN_TOPIC in owner["suggested_query"]
    n = len(suggestion_calls(llm))
    again = ask(client, s["owner"], question)["suggestion"]
    assert again == owner and len(suggestion_calls(llm)) == n  # aynı kapsam: önbellekten
    narrow = ask(client, s["ik_user"], question)["suggestion"]
    assert len(suggestion_calls(llm)) == n + 1  # farklı kapsam: yeniden, kendi ipucuyla hesaplanır
    assert narrow is None or FIN_TOPIC not in str(narrow)
    assert FIN_DOC not in suggestion_calls(llm)[-1]


def test_other_company_documents_never_appear(client, s, llm, handler):
    handler(lambda p: _out(f"{B_TOPIC} nedir?"))
    sug = ask(client, s["ik_user"], "kreş")["suggestion"]
    assert B_DOC not in suggestion_calls(llm)[-1] and B_SECRET not in suggestion_calls(llm)[-1]
    assert sug is None or B_DOC not in str(sug)
    handler(lambda p: _out("", has=False))
    ask(client, s["b_owner"], "izn")
    prompt = suggestion_calls(llm)[-1]
    assert B_DOC in prompt and IK_DOC not in prompt and FIN_DOC not in prompt


def test_department_filter_narrows_the_hint(client, s, llm, handler):
    handler(lambda p: _out("Yıllık izin kaç gündür?"))
    ask(client, s["owner"], "izn", departments=["İK"])
    prompt = suggestion_calls(llm)[-1]
    assert IK_DOC in prompt and FIN_DOC not in prompt
    r = client.post(f"{API}/chat/query", json={"question": "izn", "departments": ["Finans"]},
                    headers=bearer(s["ik_user"]))
    assert r.status_code == 403  # yetkisiz departman öneri katmanına hiç ulaşmaz


# ================================================================ dayanıklılık
@pytest.mark.parametrize("error", [RuntimeError("beklenmeyen"), ValueError("bozuk JSON")])
def test_model_errors_never_break_the_answer(client, s, handler, error):
    def boom(prompt: str) -> dict:
        raise error

    handler(boom)
    body = ask(client, s["ik_user"], "yıllık izn kaç gün")
    assert body["answer"] and body["suggestion"]["suggested_query"] == "yıllık izin kaç gün"  # sezgisel yedek


def test_slow_suggestion_does_not_delay_the_answer(client, s, handler, container):
    release = threading.Event()

    def slow(prompt: str) -> dict:
        release.wait(10)
        return _out("Yıllık izin kaç gündür?")

    handler(slow)
    wait = container.settings.suggestion_wait_seconds
    t0 = time.perf_counter()
    try:
        body = ask(client, s["ik_user"], "izn talebi")
    finally:
        release.set()
    assert body["answer"] and body["suggestion"] is None
    assert time.perf_counter() - t0 < wait + 5


def test_very_long_and_unicode_questions_are_safe(client, s):
    body = ask(client, s["ik_user"], ("İzin " + "ğüşıöç " * 150)[:1990])
    assert body["suggestion"] is None
    r = client.post(f"{API}/chat/query", json={"question": "x" * 2001}, headers=bearer(s["ik_user"]))
    assert r.status_code == 422
