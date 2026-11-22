"""Oturum ve davet belirteçleri.

Biçim: ``<önek><tenant_id hex>.<gizli>``; ör. ``rs_3f2a…9b.<43 karakter>``.

* Belirteç şirket kimliğini taşır: sunucu önce o şirketin RLS bağlamını açar, sonra
  belirtecin SHA-256 özetini **yalnızca o şirkette** arar. Başka bir şirketin kimliğiyle
  üretilmiş sahte belirteç orada eşleşme bulamaz.
* Veritabanında yalnızca özet saklanır; ham belirteç istemciye bir kez verilir.
* Şirket kimliği gizli değildir; güvenlik 32 baytlık rastgele gizli kısımdan gelir.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from dataclasses import dataclass

SESSION_PREFIX = "rs_"
INVITE_PREFIX = "ri_"
_SECRET_BYTES = 32
_TOKEN_RE = re.compile(r"^(?P<prefix>r[si]_)(?P<tenant>[0-9a-f]{32})\.(?P<secret>[A-Za-z0-9_-]{43})$")


@dataclass(frozen=True, slots=True)
class ParsedToken:
    tenant_id: uuid.UUID
    token_hash: str


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token(prefix: str, tenant_id: uuid.UUID) -> tuple[str, str]:
    """(ham belirteç, özet) döner. Ham belirteç saklanmaz."""
    if prefix not in (SESSION_PREFIX, INVITE_PREFIX):
        raise ValueError("Bilinmeyen belirteç öneki")
    token = f"{prefix}{tenant_id.hex}.{secrets.token_urlsafe(_SECRET_BYTES)}"
    return token, hash_token(token)


def parse_token(token: str | None, prefix: str) -> ParsedToken | None:
    """Biçimi geçersiz veya öneki farklı belirteç için None (istisna fırlatmaz)."""
    if not token or len(token) > 128:
        return None
    m = _TOKEN_RE.match(token)
    if not m or m.group("prefix") != prefix:
        return None
    return ParsedToken(tenant_id=uuid.UUID(hex=m.group("tenant")), token_hash=hash_token(token))
