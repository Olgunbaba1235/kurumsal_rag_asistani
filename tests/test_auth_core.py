"""Kimlik çekirdeği birim testleri (veritabanısız): parola özeti, belirteçler, yetki matrisi.

Matris, teknik plandaki "Rol ve yetki matrisi" tablosunun birebir kopyasıdır; kodda
bir yetki değişirse bu test kırılır ve plan ile kod arasındaki fark görünür olur.
"""

from __future__ import annotations

import time
import uuid

import pytest

from app.core import passwords as pw
from app.core.permissions import (
    Action,
    Actor,
    Role,
    Target,
    Via,
    allowed_departments,
    can,
    can_assign_role,
    can_invite,
    can_manage_member,
    can_revoke_sessions,
    can_transfer_ownership,
)
from app.core.tokens import INVITE_PREFIX, SESSION_PREFIX, hash_token, new_token, parse_token

OWNER = Actor(Via.SESSION, Role.ADMIN, is_owner=True)
ADMIN = Actor(Via.SESSION, Role.ADMIN)
MANAGER = Actor(Via.SESSION, Role.MANAGER, departments=("IK",))
USER = Actor(Via.SESSION, Role.USER, departments=("IK",))
API = Actor.api_key()
ACTORS = {"owner": OWNER, "admin": ADMIN, "manager": MANAGER, "user": USER, "api_key": API}


# ============================================================= parola
def test_hash_has_expected_format_and_unique_salt():
    a, b = pw.hash_password("dogru-parola-123"), pw.hash_password("dogru-parola-123")
    assert a != b and a.startswith("scrypt$32768$8$1$") and a.count("$") == 5
    assert "dogru-parola-123" not in a


def test_verify_accepts_right_and_rejects_wrong_password():
    h = pw.hash_password("dogru-parola-123")
    assert pw.verify_password("dogru-parola-123", h)
    assert not pw.verify_password("dogru-parola-124", h)
    assert not pw.verify_password("", h)


@pytest.mark.parametrize(
    "stored",
    [None, "", "duz-metin-parola", "md5$abc", "scrypt$1048576$8$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 44,
     "scrypt$32768$8$1$***$***", "scrypt$30000$8$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 44],
)
def test_malformed_or_costly_hashes_are_rejected(stored):
    assert pw.verify_password("herhangi-bir-parola", stored) is False


def test_missing_user_costs_about_the_same_as_wrong_password():
    h = pw.hash_password("dogru-parola-123")
    t0 = time.perf_counter()
    pw.verify_password("yanlis-parola-12", h)
    wrong = time.perf_counter() - t0
    t0 = time.perf_counter()
    pw.verify_password("yanlis-parola-12", None)  # kullanıcı yok
    missing = time.perf_counter() - t0
    assert missing > wrong * 0.5


def test_needs_rehash():
    assert not pw.needs_rehash(pw.hash_password("dogru-parola-123"))
    assert pw.needs_rehash("scrypt$16384$8$1$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 43 + "=")


@pytest.mark.parametrize(
    ("password", "ok"),
    [("kisa", False), ("a" * 11, False), ("a" * 12, True), ("a" * 129, False), (" " * 20, False),
     ("ayse.yilmaz@ornek.com", False), ("ayse.yilmaz", False), ("Uzun ve guvenli 2026!", True)],
)
def test_password_policy(password, ok):
    if ok:
        pw.check_password_policy(password, "ayse.yilmaz@ornek.com")
    else:
        with pytest.raises(pw.PasswordPolicyError):
            pw.check_password_policy(password, "ayse.yilmaz@ornek.com")


# ============================================================= belirteçler
def test_token_carries_tenant_and_only_hash_is_stored():
    tid = uuid.uuid4()
    token, digest = new_token(SESSION_PREFIX, tid)
    assert token.startswith(f"rs_{tid.hex}.") and len(digest) == 64 and token not in digest
    parsed = parse_token(token, SESSION_PREFIX)
    assert parsed.tenant_id == tid and parsed.token_hash == digest == hash_token(token)


def test_tokens_are_unique():
    tid = uuid.uuid4()
    assert len({new_token(SESSION_PREFIX, tid)[0] for _ in range(200)}) == 200


def test_forged_tenant_prefix_changes_the_hash():
    """B'nin şirket kimliği + A'nın gizli kısmı: özet A'nınkiyle aynı olamaz → B'de eşleşmez."""
    a, b = uuid.uuid4(), uuid.uuid4()
    token, digest = new_token(SESSION_PREFIX, a)
    forged = token.replace(a.hex, b.hex)
    parsed = parse_token(forged, SESSION_PREFIX)
    assert parsed.tenant_id == b and parsed.token_hash != digest


@pytest.mark.parametrize(
    "bad",
    [None, "", "rs_", "rs_abc.def", "Bearer rs_x", "rs_" + "g" * 32 + "." + "a" * 43,
     "rs_" + "a" * 32 + "." + "a" * 42, "rs_" + "a" * 32 + "." + "a" * 43 + "x", "x" * 500],
)
def test_malformed_tokens_are_rejected(bad):
    assert parse_token(bad, SESSION_PREFIX) is None


def test_session_and_invite_tokens_are_not_interchangeable():
    tid = uuid.uuid4()
    session, _ = new_token(SESSION_PREFIX, tid)
    invite, _ = new_token(INVITE_PREFIX, tid)
    assert parse_token(session, INVITE_PREFIX) is None and parse_token(invite, SESSION_PREFIX) is None
    with pytest.raises(ValueError):
        new_token("xx_", tid)


# ============================================================= yetki matrisi
Y, N = True, False
#                         owner admin manager user api_key
MATRIX = {
    Action.ASK:                (Y, Y, Y, Y, Y),
    Action.DOCS_READ:          (Y, Y, Y, Y, Y),
    Action.DOCS_UPLOAD:        (Y, Y, Y, N, Y),
    Action.DOCS_DELETE:        (Y, Y, Y, N, Y),
    Action.ANALYTICS:          (Y, Y, N, N, Y),
    Action.DEMO_DATA:          (Y, Y, N, N, Y),
    Action.MEMBERS_LIST:       (Y, Y, N, N, Y),
    Action.ROTATE_API_KEY:     (Y, N, N, N, N),
    Action.TRANSFER_OWNERSHIP: (Y, N, N, N, N),
}


@pytest.mark.parametrize("action", list(Action))
def test_action_matrix_matches_plan(action):
    assert action in MATRIX, "Plana eklenmemiş yeni işlem"
    got = tuple(can(a, action) for a in ACTORS.values())
    assert got == MATRIX[action]


TENANT_DEPTS = ["Genel", "IK", "Finans", "Hukuk"]


def test_department_scope():
    assert allowed_departments(OWNER, TENANT_DEPTS) == TENANT_DEPTS
    assert allowed_departments(API, TENANT_DEPTS) == TENANT_DEPTS
    assert allowed_departments(USER, TENANT_DEPTS) == ["IK"]
    # Şirket listesinde olmayan atama yok sayılır; boş atama = hiçbir departman (fail-closed)
    assert allowed_departments(Actor(Via.SESSION, Role.USER, departments=("IK", "Silinmis")), TENANT_DEPTS) == ["IK"]
    assert allowed_departments(Actor(Via.SESSION, Role.MANAGER), TENANT_DEPTS) == []


#                     owner admin manager user api_key
INVITE = {
    Role.USER:    (Y, Y, N, N, Y),
    Role.MANAGER: (Y, Y, N, N, Y),
    Role.ADMIN:   (Y, N, N, N, N),
}


@pytest.mark.parametrize("role", list(Role))
def test_invite_matrix(role):
    assert tuple(can_invite(a, role) for a in ACTORS.values()) == INVITE[role]


def test_owner_granting_invite_only_for_ownerless_tenant_via_api_key():
    assert can_invite(API, Role.ADMIN, grants_owner=True, tenant_has_owner=False)
    assert not can_invite(API, Role.ADMIN, grants_owner=True, tenant_has_owner=True)
    assert not can_invite(API, Role.MANAGER, grants_owner=True, tenant_has_owner=False)
    assert not can_invite(OWNER, Role.ADMIN, grants_owner=True, tenant_has_owner=False)


#                                  owner admin manager user api_key
MANAGE = {
    "manager/user":                (Y, Y, N, N, Y),
    "admin":                       (Y, N, N, N, N),
    "owner":                       (N, N, N, N, N),
}
TARGETS = {"manager/user": Target(Role.USER), "admin": Target(Role.ADMIN), "owner": Target(Role.ADMIN, is_owner=True)}


@pytest.mark.parametrize("target", list(MANAGE))
def test_manage_member_matrix(target):
    assert tuple(can_manage_member(a, TARGETS[target]) for a in ACTORS.values()) == MANAGE[target]


@pytest.mark.parametrize("actor", list(ACTORS))
def test_nobody_can_manage_or_promote_themselves(actor):
    a = ACTORS[actor]
    me = Target(a.role, is_owner=a.is_owner, is_self=True)
    for new_role in Role:
        assert not can_manage_member(a, me, new_role=new_role)


def test_admin_cannot_promote_anyone_to_admin():
    assert not can_manage_member(ADMIN, Target(Role.USER), new_role=Role.ADMIN)
    assert not can_manage_member(API, Target(Role.MANAGER), new_role=Role.ADMIN)
    assert can_manage_member(ADMIN, Target(Role.USER), new_role=Role.MANAGER)
    assert can_manage_member(OWNER, Target(Role.USER), new_role=Role.ADMIN)
    assert not can_assign_role(MANAGER, Role.USER)


def test_session_revocation_rules():
    for a in ACTORS.values():
        assert can_revoke_sessions(a, Target(a.role, is_owner=a.is_owner, is_self=True))
    assert can_revoke_sessions(ADMIN, Target(Role.USER))
    assert not can_revoke_sessions(ADMIN, Target(Role.ADMIN))
    assert not can_revoke_sessions(USER, Target(Role.USER))
    assert not can_revoke_sessions(OWNER, Target(Role.ADMIN, is_owner=True))


def test_ownership_transfer_rules():
    assert can_transfer_ownership(OWNER, Target(Role.ADMIN), target_active=True)
    assert not can_transfer_ownership(OWNER, Target(Role.ADMIN), target_active=False)
    assert not can_transfer_ownership(OWNER, Target(Role.MANAGER), target_active=True)
    assert not can_transfer_ownership(OWNER, Target(Role.ADMIN, is_self=True), target_active=True)
    assert not can_transfer_ownership(ADMIN, Target(Role.ADMIN), target_active=True)
    assert not can_transfer_ownership(API, Target(Role.ADMIN), target_active=True)


def test_owner_flag_without_admin_role_grants_nothing_extra():
    """Veritabanındaki CHECK kısıtı bunu engeller; kod da tek başına güvenmez."""
    fake = Actor(Via.SESSION, Role.USER, is_owner=True)
    assert not can(fake, Action.ROTATE_API_KEY) and not can_invite(fake, Role.ADMIN)
