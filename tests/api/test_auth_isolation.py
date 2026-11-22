"""Kullanıcı kimliği ve yetkisiz erişim testleri (PGlite üzerinde gerçek PostgreSQL).

Teknik plandaki senaryolar (numaralar plandaki tabloyla aynı):
1–6 şirket ve departman izolasyonu, rol kapıları · 7–9 oturum askıya alma/süre/iptal ·
10, 20 davet tek kullanımlık ve süreli · 11 rol yükseltme ve üye yönetimi · 12–13 ham SQL ile RLS ·
14 iki kimlik birden · 15 log'da gizli veri yok · 16 migration geri alma döngüsü ·
17–19 girişte şirket izolasyonu · 21 iptal edilen oturum · 22 veritabanında düz metin gizli yok.

Testler kendi şirketlerini kayıtla oluşturur; mevcut `tenant_a`/`tenant_b` verisine dokunmaz.
16 numaralı migration döngüsü bu dosyanın en sonundadır.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from tests.api.conftest import ROOT, run

API = "/api/v1"
PW = "Guclu-Parola-2026!"
DEPTS = ["İK", "Finans", "Hukuk"]
SECRET_IK, SECRET_FIN, SECRET_B = "IK-GIZLI-111", "FIN-GIZLI-222", "B-GIZLI-999"
SUFFIX = uuid.uuid4().hex[:6]


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


def invite(client, token: str, email: str, role: str, departments: list[str] | None = None) -> str:
    r = client.post(
        f"{API}/company/invitations",
        json={"email": email, "role": role, "departments": departments or []},
        headers=bearer(token),
    )
    assert r.status_code == 201, r.text
    return r.json()["token"]


def accept(client, inv_token: str, email: str, password: str = PW):
    return client.post(
        f"{API}/auth/accept-invitation",
        json={"token": inv_token, "email": email, "password": password, "full_name": "Davetli Kişi"},
    )


def join(client, owner_token: str, email: str, role: str, departments: list[str] | None = None) -> dict:
    r = accept(client, invite(client, owner_token, email, role, departments), email)
    assert r.status_code == 201, r.text
    return r.json()


def login(client, email: str, slug: str, password: str = PW):
    return client.post(f"{API}/auth/login", json={"email": email, "password": password, "company_slug": slug})


def upload(client, token: str, filename: str, content: str, department: str):
    return client.post(
        f"{API}/documents/upload",
        params={"wait": "true"},
        files={"file": (filename, content.encode(), "application/octet-stream")},
        data={"department": department},
        headers=bearer(token),
    )


def doc_names(client, token: str) -> set[str]:
    r = client.get(f"{API}/documents", headers=bearer(token))
    assert r.status_code == 200, r.text
    return {d["filename"] for d in r.json()["items"]}


def evidence_sources(body: dict) -> set[str]:
    return {e["source"] for e in body.get("evidence", [])}


@pytest.fixture(scope="module")
def w(client) -> dict:
    """İki şirket: A (sahip, admin, İK manager'ı, İK user'ı, departmansız user) ve B (sahip)."""
    a_slug, b_slug = f"auth-a-{SUFFIX}", f"auth-b-{SUFFIX}"
    a = register(client, a_slug, f"sahip-a-{SUFFIX}@a.com", "Alfa Hukuk Bürosu")
    b = register(client, b_slug, f"sahip-b-{SUFFIX}@b.com", "Beta Muhasebe")
    owner = a["token"]
    docs = {}
    for fn, body, dept in [
        ("IK_Yonetmelik.md", f"# İzin\n\nYıllık izin 20 iş günüdür. Kod: {SECRET_IK}.", "İK"),
        ("Finans_Butce.md", f"# Bütçe\n\nYıllık bütçe onayı CFO'dadır. Kod: {SECRET_FIN}.", "Finans"),
    ]:
        r = upload(client, owner, fn, body, dept)
        assert r.status_code == 201, r.text
        docs[fn] = r.json()["document"]["id"]
    r = upload(client, b["token"], "B_IK.md", f"# İzin\n\nYıllık izin 30 iş günüdür. Kod: {SECRET_B}.", "İK")
    assert r.status_code == 201, r.text
    docs["B_IK.md"] = r.json()["document"]["id"]
    return {
        "a_slug": a_slug, "b_slug": b_slug, "a_id": a["me"]["company"]["id"], "b_id": b["me"]["company"]["id"],
        "owner": owner, "b_owner": b["token"], "docs": docs,
        "admin": join(client, owner, f"admin-{SUFFIX}@a.com", "admin")["token"],
        "manager": join(client, owner, f"mudur-{SUFFIX}@a.com", "manager", ["İK"])["token"],
        "user": join(client, owner, f"calisan-{SUFFIX}@a.com", "user", ["İK"])["token"],
        "user_none": join(client, owner, f"bos-{SUFFIX}@a.com", "user", [])["token"],
    }


def members(client, token: str) -> dict[str, dict]:
    r = client.get(f"{API}/company/members", headers=bearer(token))
    assert r.status_code == 200, r.text
    return {m["email"]: m for m in r.json()}


# ================================================================ 1–6 izolasyon ve roller
def test_register_creates_owner_admin_with_all_departments(client, w):
    me = client.get(f"{API}/auth/me", headers=bearer(w["owner"])).json()
    assert me["via"] == "session" and me["role"] == "admin" and me["is_owner"] is True
    assert me["departments"] == DEPTS


def test_1_other_company_documents_are_invisible(client, w):
    assert doc_names(client, w["owner"]) == {"IK_Yonetmelik.md", "Finans_Butce.md"}
    assert doc_names(client, w["b_owner"]) == {"B_IK.md"}
    b_doc = w["docs"]["B_IK.md"]
    for token in (w["owner"], w["admin"], w["user"]):
        assert client.get(f"{API}/documents/{b_doc}", headers=bearer(token)).status_code == 404
    assert client.delete(f"{API}/documents/{b_doc}", headers=bearer(w["admin"])).status_code == 404
    body = client.post(f"{API}/chat/query", json={"question": f"{SECRET_B} kodu nedir?"}, headers=bearer(w["owner"])).json()
    assert "B_IK.md" not in evidence_sources(body) and SECRET_B not in body["answer"]
    assert client.get(f"{API}/documents/{b_doc}", headers=bearer(w["b_owner"])).status_code == 200


def test_2_forged_token_with_other_company_id_is_rejected(client, w):
    a_hex, b_hex = uuid.UUID(w["a_id"]).hex, uuid.UUID(w["b_id"]).hex
    forged = w["user"].replace(a_hex, b_hex)
    assert forged != w["user"]
    assert client.get(f"{API}/documents", headers=bearer(forged)).status_code == 401
    assert client.get(f"{API}/auth/me", headers=bearer(w["user"] + "x")).status_code == 401
    assert client.get(f"{API}/auth/me", headers=bearer("rs_" + "0" * 32 + "." + "a" * 43)).status_code == 401


def test_3_user_sees_only_assigned_department(client, w):
    assert doc_names(client, w["user"]) == {"IK_Yonetmelik.md"}
    fin = w["docs"]["Finans_Butce.md"]
    assert client.get(f"{API}/documents/{fin}", headers=bearer(w["user"])).status_code == 404
    r = client.post(f"{API}/chat/query", json={"question": "Bütçe onayı?", "departments": ["Finans"]},
                    headers=bearer(w["user"]))
    assert r.status_code == 403
    assert client.get(f"{API}/documents", params={"department": "Finans"}, headers=bearer(w["user"])).status_code == 403
    body = client.post(f"{API}/chat/query", json={"question": f"Yıllık bütçe onayı ve {SECRET_FIN}?"},
                       headers=bearer(w["user"])).json()
    # Gizli kod soruda geçtiği için yanıtın soru alanında görünür; içerik alanlarında görünmemeli
    leaked = [body["answer"], *map(str, body["citations"]), *(e["text"] for e in body["evidence"])]
    assert "Finans_Butce.md" not in evidence_sources(body) and not any(SECRET_FIN in x for x in leaked)
    me = client.get(f"{API}/auth/me", headers=bearer(w["user"])).json()
    assert me["departments"] == ["İK"] and me["role"] == "user"


def test_4_user_cannot_upload_or_delete(client, w):
    assert upload(client, w["user"], "x.md", "# X\n\nmetin", "İK").status_code == 403
    ik = w["docs"]["IK_Yonetmelik.md"]
    assert client.delete(f"{API}/documents/{ik}", headers=bearer(w["user"])).status_code == 403
    assert client.get(f"{API}/documents/{ik}", headers=bearer(w["owner"])).status_code == 200


def test_5_manager_limited_to_own_department(client, w):
    assert upload(client, w["manager"], "fin2.md", "# F\n\nmetin", "Finans").status_code == 403
    fin = w["docs"]["Finans_Butce.md"]
    assert client.delete(f"{API}/documents/{fin}", headers=bearer(w["manager"])).status_code == 404
    assert client.get(f"{API}/documents/{fin}", headers=bearer(w["owner"])).status_code == 200
    r = upload(client, w["manager"], "IK_Ek.md",
               "# Ek İzin\n\nDoğum günü izni bir iş günüdür ve yöneticinin onayıyla kullanılır.", "İK")
    assert r.status_code == 201, r.text
    assert client.delete(f"{API}/documents/{r.json()['document']['id']}", headers=bearer(w["manager"])).status_code == 204


def test_6_user_without_departments_sees_nothing(client, w):
    assert doc_names(client, w["user_none"]) == set()
    body = client.post(f"{API}/chat/query", json={"question": "Yıllık izin kaç gün?"}, headers=bearer(w["user_none"])).json()
    assert not body["is_answered"] and not body["evidence"]


def test_sales_analytics_only_for_owner_and_admin(client, w):
    for token in (w["owner"], w["admin"]):
        assert client.get(f"{API}/analytics/schema", headers=bearer(token)).status_code == 200
    for token in (w["manager"], w["user"]):
        assert client.get(f"{API}/analytics/schema", headers=bearer(token)).status_code == 403
        assert client.post(f"{API}/analytics/demo-data", headers=bearer(token)).status_code == 403
        r = client.post(f"{API}/agent/query", json={"question": "Ciro?", "intent": "SQL_ONLY"}, headers=bearer(token))
        assert r.status_code == 403
    r = client.post(f"{API}/agent/query", json={"question": "Yıllık izin kaç gün?"}, headers=bearer(w["user"]))
    assert r.status_code == 200 and r.json()["intent"] == "DOCUMENT_ONLY" and r.json()["sql"] is None


# ================================================================ 7–9, 21 oturumlar
def test_7_suspended_member_loses_access_immediately(client, w):
    email = f"askiya-{SUFFIX}@a.com"
    token = join(client, w["owner"], email, "user", ["İK"])["token"]
    mid = members(client, w["owner"])[email]["membership_id"]
    r = client.patch(f"{API}/company/members/{mid}", json={"status": "suspended"}, headers=bearer(w["admin"]))
    assert r.status_code == 204, r.text
    assert client.get(f"{API}/documents", headers=bearer(token)).status_code == 401
    assert login(client, email, w["a_slug"]).status_code == 401
    client.patch(f"{API}/company/members/{mid}", json={"status": "active"}, headers=bearer(w["admin"]))
    assert client.get(f"{API}/documents", headers=bearer(token)).status_code == 401  # iptal geri alınmaz
    assert login(client, email, w["a_slug"]).status_code == 200


@pytest.fixture
def shift_clock(container, monkeypatch):
    real = container.auth.clock

    def _shift(**delta):  # noqa: ANN003
        monkeypatch.setattr(container.auth, "clock", lambda: real() + dt.timedelta(**delta))

    return _shift


def test_8_absolute_and_idle_expiry(client, w, shift_clock):
    email = f"calisan-{SUFFIX}@a.com"
    t1 = login(client, email, w["a_slug"]).json()["token"]
    shift_clock(hours=12, minutes=1)
    assert client.get(f"{API}/auth/me", headers=bearer(t1)).status_code == 401
    shift_clock(minutes=0)
    t2 = login(client, email, w["a_slug"]).json()["token"]
    shift_clock(hours=2, minutes=1)
    assert client.get(f"{API}/auth/me", headers=bearer(t2)).status_code == 401
    shift_clock(minutes=0)
    t3 = login(client, email, w["a_slug"]).json()["token"]
    shift_clock(hours=1, minutes=50)  # hareketli oturum: son görülme güncellenir
    assert client.get(f"{API}/auth/me", headers=bearer(t3)).status_code == 200


def test_9_logout_logout_all_and_password_change(client, w):
    email = f"parola-{SUFFIX}@a.com"
    first = join(client, w["owner"], email, "user", ["İK"])["token"]
    second = login(client, email, w["a_slug"]).json()["token"]
    assert client.post(f"{API}/auth/logout", headers=bearer(first)).status_code == 204
    assert client.get(f"{API}/auth/me", headers=bearer(first)).status_code == 401
    assert client.get(f"{API}/auth/me", headers=bearer(second)).status_code == 200
    third = login(client, email, w["a_slug"]).json()["token"]
    assert client.post(f"{API}/auth/logout-all", headers=bearer(second)).json()["revoked"] == 2
    assert client.get(f"{API}/auth/me", headers=bearer(third)).status_code == 401

    old = login(client, email, w["a_slug"]).json()["token"]
    other = login(client, email, w["a_slug"]).json()["token"]
    r = client.post(f"{API}/auth/password", json={"current_password": PW, "new_password": "Yeni-Guclu-Parola-99"},
                    headers=bearer(old))
    assert r.status_code == 200, r.text
    new = r.json()["token"]
    assert client.get(f"{API}/auth/me", headers=bearer(old)).status_code == 401
    assert client.get(f"{API}/auth/me", headers=bearer(other)).status_code == 401
    assert client.get(f"{API}/auth/me", headers=bearer(new)).status_code == 200
    assert login(client, email, w["a_slug"]).status_code == 401
    assert login(client, email, w["a_slug"], "Yeni-Guclu-Parola-99").status_code == 200


def test_21_revoked_token_rejected_everywhere(client, w):
    token = login(client, f"calisan-{SUFFIX}@a.com", w["a_slug"]).json()["token"]
    client.post(f"{API}/auth/logout", headers=bearer(token))
    for method, path in [("get", "/documents"), ("post", "/chat/query"), ("get", "/auth/me"), ("get", "/company/members")]:
        kw = {"json": {"question": "izin?"}} if method == "post" else {}
        assert getattr(client, method)(f"{API}{path}", headers=bearer(token), **kw).status_code == 401, path


def test_login_lockout_after_repeated_failures(client, w):
    email = f"kilit-{SUFFIX}@a.com"
    join(client, w["owner"], email, "user", ["İK"])
    for _ in range(5):
        assert login(client, email, w["a_slug"], "yanlis-parola-123").status_code == 401
    r = login(client, email, w["a_slug"])  # doğru parola da kilit süresince reddedilir
    assert r.status_code == 401 and r.json()["detail"] == "E-posta, parola veya şirket hatalı."


# ================================================================ 10, 20 davetler
def test_10_invitation_single_use_expiry_and_binding(client, w, container, shift_clock):
    email = f"davet-{SUFFIX}@a.com"
    tok = invite(client, w["owner"], email, "user", ["İK"])
    assert accept(client, tok, f"baska-{SUFFIX}@a.com").status_code == 401  # e-posta eşleşmeli
    forged = tok.replace(uuid.UUID(w["a_id"]).hex, uuid.UUID(w["b_id"]).hex)
    assert accept(client, forged, email).status_code == 401  # başka şirketin bağlamında eşleşmez
    assert accept(client, tok, email).status_code == 201
    assert accept(client, tok, email).status_code == 401  # tek kullanımlık
    assert email in members(client, w["owner"]) and email not in members(client, w["b_owner"])

    late = f"gec-{SUFFIX}@a.com"
    tok2 = invite(client, w["owner"], late, "user", ["İK"])
    shift_clock(hours=72, minutes=1)
    assert accept(client, tok2, late).status_code == 401
    shift_clock(minutes=0)
    assert late not in members(client, w["owner"])

    revoked = f"iptal-{SUFFIX}@a.com"
    r = client.post(f"{API}/company/invitations", json={"email": revoked, "role": "user"}, headers=bearer(w["owner"]))
    assert client.delete(f"{API}/company/invitations/{r.json()['id']}", headers=bearer(w["owner"])).status_code == 204
    assert accept(client, r.json()["token"], revoked).status_code == 401


def test_20_concurrent_acceptance_only_one_wins(client, w, container):
    email = f"yaris-{SUFFIX}@a.com"
    tok = invite(client, w["owner"], email, "user", ["İK"])

    async def both():
        async def one():
            try:
                await container.auth.accept_invitation(token=tok, email=email, password=PW, full_name="Yarış")
                return True
            except Exception:  # noqa: BLE001
                return False

        return await asyncio.gather(one(), one())

    assert sorted(run(client, both)) == [False, True]
    assert list(members(client, w["owner"])).count(email) == 1


def test_member_and_manager_cannot_invite(client, w):
    for token in (w["manager"], w["user"]):
        r = client.post(f"{API}/company/invitations", json={"email": f"x-{SUFFIX}@a.com", "role": "user"},
                        headers=bearer(token))
        assert r.status_code == 403
        assert client.get(f"{API}/company/members", headers=bearer(token)).status_code == 403


# ================================================================ 11 rol yükseltme
def test_11_no_self_escalation_and_owner_protection(client, w):
    m = members(client, w["owner"])
    owner_id = next(v["membership_id"] for v in m.values() if v["is_owner"])
    user_id = m[f"calisan-{SUFFIX}@a.com"]["membership_id"]
    admin_id = m[f"admin-{SUFFIX}@a.com"]["membership_id"]
    patch = lambda tok, mid, body: client.patch(f"{API}/company/members/{mid}", json=body, headers=bearer(tok))  # noqa: E731

    assert patch(w["user"], user_id, {"role": "admin"}).status_code == 403  # kendini yükseltemez
    assert patch(w["admin"], admin_id, {"departments": DEPTS}).status_code == 403  # kendini yönetemez
    assert patch(w["admin"], owner_id, {"status": "suspended"}).status_code == 403  # sahip korunur
    assert patch(w["owner"], owner_id, {"role": "user"}).status_code == 403
    assert patch(w["admin"], user_id, {"role": "admin"}).status_code == 403  # admin'i yalnızca sahip verir
    r = client.post(f"{API}/company/invitations", json={"email": f"adm2-{SUFFIX}@a.com", "role": "admin"},
                    headers=bearer(w["admin"]))
    assert r.status_code == 403
    b_member = next(iter(members(client, w["b_owner"]).values()))["membership_id"]
    assert patch(w["owner"], b_member, {"status": "suspended"}).status_code == 404  # başka şirket
    assert client.post(f"{API}/company/members/{b_member}/revoke-sessions", headers=bearer(w["owner"])).status_code == 404
    assert client.get(f"{API}/auth/me", headers=bearer(w["b_owner"])).status_code == 200
    assert members(client, w["owner"])[f"calisan-{SUFFIX}@a.com"]["role"] == "user"


def test_api_key_rotation_and_ownership_transfer(client, w):
    assert client.post(f"{API}/company/api-key/rotate", headers=bearer(w["admin"])).status_code == 403
    r = client.post(f"{API}/company/api-key/rotate", headers=bearer(w["owner"]))
    assert r.status_code == 200
    key = r.json()["api_key"]
    names = {d["filename"] for d in client.get(f"{API}/documents", headers={"X-API-Key": key}).json()["items"]}
    assert names == {"IK_Yonetmelik.md", "Finans_Butce.md"}
    # API anahtarı admin gibi davranır ama admin daveti oluşturamaz
    r = client.post(f"{API}/company/invitations", json={"email": f"k-{SUFFIX}@a.com", "role": "admin"},
                    headers={"X-API-Key": key})
    assert r.status_code == 403


# ================================================================ 12–13 ham SQL ile RLS
def test_12_rls_on_new_tables(client, container, w):
    a, b = uuid.UUID(w["a_id"]), uuid.UUID(w["b_id"])

    async def check():
        async with container.db.tenant_session(a) as s:
            for table in ("memberships", "invitations", "user_sessions"):
                total = await s.scalar(text(f"SELECT count(*) FROM {table}"))
                other = await s.scalar(text(f"SELECT count(*) FROM {table} WHERE tenant_id = :b"), {"b": b})
                assert total > 0 and other == 0, table
        async with container.db.system_session() as s:
            for table in ("memberships", "invitations", "user_sessions"):
                assert await s.scalar(text(f"SELECT count(*) FROM {table}")) == 0, table
        async with container.db.tenant_session(a) as s:
            uid = await s.scalar(text("SELECT user_id FROM memberships LIMIT 1"))
            with pytest.raises((ProgrammingError, DBAPIError)):
                await s.execute(
                    text("INSERT INTO memberships (id, tenant_id, user_id, role) VALUES (:i, :b, :u, 'admin')"),
                    {"i": uuid.uuid4(), "b": b, "u": uid},
                )
                await s.flush()

    run(client, check)


def test_13_department_rls_fails_closed(client, container, w):
    a = uuid.UUID(w["a_id"])

    async def check():
        async with container.db.tenant_session(a, ["İK"]) as s:
            depts = set((await s.scalars(text("SELECT DISTINCT department FROM document_chunks"))).all())
            assert depts == {"İK"}
            assert await s.scalar(text("SELECT count(*) FROM documents WHERE department = 'Finans'")) == 0
            with pytest.raises((ProgrammingError, DBAPIError)):
                await s.execute(
                    text("UPDATE documents SET department = 'Finans' WHERE department = 'İK'")
                )
                await s.flush()
        async with container.db.tenant_session(a, []) as s:
            assert await s.scalar(text("SELECT count(*) FROM documents")) == 0
        # Uygulama katmanı olmadan: tenant ayarlı ama departman ayarı YOK → hiçbir belge
        async with container.db.engine.connect() as conn, conn.begin():
            await conn.execute(text("SET LOCAL ROLE rag_app"))
            await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(a)})
            assert await conn.scalar(text("SELECT count(*) FROM documents")) == 0
            assert await conn.scalar(text("SELECT count(*) FROM document_chunks")) == 0
        async with container.db.tenant_session(a) as s:  # tüm departmanlar
            assert await s.scalar(text("SELECT count(*) FROM documents WHERE department = 'Finans'")) >= 1

    run(client, check)


# ================================================================ 14–15
def test_14_both_credentials_rejected(client, w):
    r = client.get(f"{API}/documents", headers={**bearer(w["owner"]), "X-API-Key": "rag_x"})
    assert r.status_code == 400
    assert client.get(f"{API}/documents").status_code == 401


def test_15_failed_login_logs_contain_no_secrets(client, w, caplog):
    email = f"calisan-{SUFFIX}@a.com"
    with caplog.at_level(logging.INFO):
        login(client, email, w["a_slug"], "Log-Icin-Yanlis-Parola")
        token = login(client, email, w["a_slug"]).json()["token"]
    logged = caplog.text
    assert "Başarısız giriş" in logged
    for secret in (email, "Log-Icin-Yanlis-Parola", PW, token, token.split(".")[1]):
        assert secret not in logged


# ================================================================ 17–19 girişte izolasyon
def test_17_multi_company_user_is_isolated_per_session(client, w):
    email = f"cift-{SUFFIX}@x.com"
    a_tok = join(client, w["owner"], email, "user", ["İK"])["token"]
    b_tok = join(client, w["b_owner"], email, "user", ["İK"])["token"]
    assert doc_names(client, a_tok) == {"IK_Yonetmelik.md"}
    assert doc_names(client, b_tok) == {"B_IK.md"}
    assert client.get(f"{API}/documents/{w['docs']['B_IK.md']}", headers=bearer(a_tok)).status_code == 404
    assert client.get(f"{API}/auth/me", headers=bearer(a_tok)).json()["company"]["slug"] == w["a_slug"]
    assert login(client, email, w["b_slug"]).json()["me"]["company"]["slug"] == w["b_slug"]


def test_18_login_to_foreign_company_looks_like_wrong_password(client, w):
    email = f"calisan-{SUFFIX}@a.com"
    foreign = login(client, email, w["b_slug"])
    wrong = login(client, f"yok-{SUFFIX}@a.com", w["a_slug"])
    unknown_company = login(client, email, f"olmayan-{SUFFIX}")
    for r in (foreign, wrong, unknown_company):
        assert r.status_code == 401 and r.json() == {"detail": "E-posta, parola veya şirket hatalı."}


def test_19_responses_never_mention_other_companies(client, w):
    bodies = [
        client.get(f"{API}/auth/me", headers=bearer(w["user"])).text,
        client.get(f"{API}/company/members", headers=bearer(w["owner"])).text,
        client.get(f"{API}/company/invitations", headers=bearer(w["owner"])).text,
        login(client, f"calisan-{SUFFIX}@a.com", w["b_slug"]).text,
    ]
    for body in bodies:
        assert w["b_slug"] not in body and w["b_id"] not in body and "Beta Muhasebe" not in body


# ================================================================ 22 düz metin gizli yok
def test_22_no_plaintext_secrets_in_database(client, container, w):
    secrets_ = [PW, "Yeni-Guclu-Parola-99", *(w[k] for k in ("owner", "admin", "manager", "user", "b_owner"))]
    secrets_ += [s.split(".")[1] for s in secrets_ if s.startswith("rs_")]

    async def dump() -> str:
        async with container.db.engine.connect() as conn:  # superuser: RLS'siz tam tarama
            parts = []
            for table in ("users", "memberships", "invitations", "user_sessions", "tenants"):
                rows = await conn.execute(text(f"SELECT t::text FROM {table} t"))
                parts += [r[0] for r in rows]
            return "\n".join(parts)

    blob = run(client, dump)
    assert "scrypt$" in blob
    for s in secrets_:
        assert s not in blob


# ================================================================ 16 migration döngüsü (en sonda)
def test_16_migration_downgrade_and_upgrade_keep_documents(client, container, w):
    from alembic import command
    from alembic.config import Config

    def migrate(sync_conn, target: str, down: bool) -> None:  # noqa: ANN001
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "migrations"))
        cfg.attributes["connection"] = sync_conn
        (command.downgrade if down else command.upgrade)(cfg, target)

    async def counts() -> tuple[int, int, int]:
        async with container.db.engine.connect() as conn:
            return tuple([await conn.scalar(text(f"SELECT count(*) FROM {t}")) for t in ("tenants", "documents", "document_chunks")])

    async def cycle(target: str, down: bool) -> None:
        async with container.db.engine.begin() as conn:
            await conn.run_sync(migrate, target, down)

    before = run(client, counts)
    run(client, cycle, "0002", True)
    async def users_table() -> object:
        async with container.db.engine.connect() as conn:
            return await conn.scalar(text("SELECT to_regclass('users')"))
    assert run(client, users_table) is None
    assert run(client, counts) == before
    run(client, cycle, "head", False)
    assert run(client, counts) == before
    assert run(client, users_table) is not None
