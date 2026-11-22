"""Rol ve yetki kuralları — tek doğruluk kaynağı.

Veritabanına dokunmaz; servisler ve uç noktalar karar için yalnızca bu modülü kullanır.
Matris teknik plandaki "Rol ve yetki matrisi" tablosunun birebir karşılığıdır ve
``tests/test_auth_core.py`` ile tablo güdümlü test edilir.

Kimlikler: oturumla gelen kullanıcı (owner = admin + sahiplik, admin, manager, user)
veya şirket API anahtarı (admin gibi davranır, sahip değildir).
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass


class Role(enum.StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"


class Via(enum.StrEnum):
    SESSION = "session"
    API_KEY = "api_key"


class Action(enum.StrEnum):
    ASK = "ask"  # sohbet / ajan (departman kapsamlı)
    DOCS_READ = "docs_read"  # listeleme / görüntüleme (departman kapsamlı)
    DOCS_UPLOAD = "docs_upload"  # departman kapsamlı
    DOCS_DELETE = "docs_delete"  # departman kapsamlı
    ANALYTICS = "analytics"  # satış analitiği (departmansız veri)
    DEMO_DATA = "demo_data"
    MEMBERS_LIST = "members_list"
    ROTATE_API_KEY = "rotate_api_key"
    TRANSFER_OWNERSHIP = "transfer_ownership"


_RANK = {Role.USER: 0, Role.MANAGER: 1, Role.ADMIN: 2}


@dataclass(frozen=True, slots=True)
class Actor:
    """İsteği yapan kimlik. ``departments`` yalnızca manager/user için anlamlıdır."""

    via: Via
    role: Role
    is_owner: bool = False
    departments: tuple[str, ...] = ()

    @classmethod
    def api_key(cls) -> Actor:
        return cls(via=Via.API_KEY, role=Role.ADMIN)

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


@dataclass(frozen=True, slots=True)
class Target:
    """Yönetilen üyelik."""

    role: Role
    is_owner: bool = False
    is_self: bool = False


def _owner(actor: Actor) -> bool:
    return actor.via is Via.SESSION and actor.is_owner and actor.is_admin


def can(actor: Actor, action: Action) -> bool:
    """Departmandan bağımsız işlem izni. Departman kapsamı ayrıca ``allowed_departments`` ile uygulanır."""
    if action in (Action.ASK, Action.DOCS_READ):
        return True
    if action in (Action.DOCS_UPLOAD, Action.DOCS_DELETE):
        return _RANK[actor.role] >= _RANK[Role.MANAGER]
    if action in (Action.ANALYTICS, Action.DEMO_DATA, Action.MEMBERS_LIST):
        return actor.is_admin
    if action in (Action.ROTATE_API_KEY, Action.TRANSFER_OWNERSHIP):
        return _owner(actor)
    return False  # bilinmeyen işlem → reddet


def allowed_departments(actor: Actor, tenant_departments: Sequence[str]) -> list[str]:
    """Admin ve API anahtarı: şirketin tüm departmanları. Manager/user: atanmış olanlarla
    şirket listesinin kesişimi (şirket sırasıyla). Boş atama = hiçbir departman."""
    if actor.is_admin:
        return list(tenant_departments)
    assigned = set(actor.departments)
    return [d for d in tenant_departments if d in assigned]


def can_assign_role(actor: Actor, new_role: Role) -> bool:
    """Davet veya rol değişikliğinde verilebilecek rol. Admin rolünü yalnızca sahip verir."""
    if new_role is Role.ADMIN:
        return _owner(actor)
    return actor.is_admin


def can_manage_member(actor: Actor, target: Target, new_role: Role | None = None) -> bool:
    """Askıya alma, yeniden etkinleştirme, rol/departman değişikliği.

    * Kimse kendini yönetemez (kendi rolünü yükseltemez, kendini askıya alamaz).
    * Sahip yalnızca sahiplik devriyle değişir; kimse onu askıya alamaz veya düşüremez.
    * Admin'i yalnızca sahip yönetir; manager/user'ı admin ve API anahtarı da yönetir.
    """
    if target.is_self or target.is_owner:
        return False
    if target.role is Role.ADMIN:
        allowed = _owner(actor)
    else:
        allowed = actor.is_admin
    if allowed and new_role is not None:
        allowed = can_assign_role(actor, new_role)
    return allowed


def can_revoke_sessions(actor: Actor, target: Target) -> bool:
    """Herkes kendi oturumlarını kapatabilir; başkasınınkini yalnızca onu yönetebilen."""
    return target.is_self or can_manage_member(actor, target)


def can_invite(actor: Actor, role: Role, *, grants_owner: bool = False, tenant_has_owner: bool = True) -> bool:
    """Sahiplik veren davet yalnızca sahibi olmayan (eski) şirketlerde, API anahtarıyla
    ve admin rolüyle oluşturulabilir — mevcut şirketlere ilk sahibi atamanın tek yolu."""
    if grants_owner:
        return actor.via is Via.API_KEY and role is Role.ADMIN and not tenant_has_owner
    return can_assign_role(actor, role)


def can_transfer_ownership(actor: Actor, target: Target, target_active: bool) -> bool:
    return _owner(actor) and not target.is_self and target.role is Role.ADMIN and target_active
