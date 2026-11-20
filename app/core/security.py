"""API anahtarı üretimi ve doğrulama.

* Anahtar yalnızca oluşturulduğunda bir kez gösterilir; veritabanında
  SHA-256 özeti saklanır (yüksek entropili rastgele anahtar → tuzsuz hızlı
  hash yeterli ve sabit zamanlı indeks araması sağlar).
* Kiracı (tenant) kimliği **yalnızca** API anahtarından türetilir; istemcinin
  gönderdiği hiçbir tenant_id değerine güvenilmez.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

API_KEY_PREFIX = "rag_"


def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def key_preview(api_key: str) -> str:
    return api_key[: len(API_KEY_PREFIX) + 6]


def constant_time_equals(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return hmac.compare_digest(a.encode(), b.encode())
