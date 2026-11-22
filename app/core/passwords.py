"""Parola özeti (scrypt, Python standart kütüphanesi) ve parola kuralı.

Saklanan biçim: ``scrypt$<N>$<r>$<p>$<tuz b64>$<özet b64>``. Parametreler özetle birlikte
saklanır; ileride artırılırsa eski özetler doğrulanmaya devam eder (``needs_rehash``).

* Tuz her özet için 16 bayt rastgeledir.
* Karşılaştırma sabit zamanlıdır.
* Bozuk veya aşırı maliyetli parametreli özetler reddedilir (DoS koruması).
* ``dummy_verify`` kullanıcı bulunamadığında da aynı işi yapar; yanıt süresinden
  hesabın varlığı anlaşılmaz.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCHEME = "scrypt"
N, R, P = 2**15, 8, 1
SALT_BYTES = 16
KEY_BYTES = 32
MAX_N = 2**17  # bundan pahalı özet doğrulanmaz
MIN_LENGTH, MAX_LENGTH = 12, 128


class PasswordPolicyError(ValueError):
    """Parola kuralına uymayan parola (mesaj kullanıcıya gösterilebilir)."""


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=KEY_BYTES, maxmem=128 * r * n * 2 + 1024 * 1024
    )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    return f"{SCHEME}${N}${R}${P}${_b64(salt)}${_b64(_derive(password, salt, N, R, P))}"


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        n_i, r_i, p_i = int(n), int(r), int(p)
        if scheme != SCHEME or not (2 <= n_i <= MAX_N) or n_i & (n_i - 1) or not (1 <= r_i <= 16) or not (1 <= p_i <= 4):
            return None
        salt_b, key_b = base64.b64decode(salt, validate=True), base64.b64decode(key, validate=True)
        if len(salt_b) < 16 or len(key_b) != KEY_BYTES:
            return None
        return n_i, r_i, p_i, salt_b, key_b
    except (ValueError, TypeError):
        return None


def verify_password(password: str, stored: str | None) -> bool:
    if not password or len(password) > MAX_LENGTH or not stored:
        dummy_verify()
        return False
    parsed = _parse(stored)
    if parsed is None:
        dummy_verify()
        return False
    n, r, p, salt, key = parsed
    return hmac.compare_digest(_derive(password, salt, n, r, p), key)


_DUMMY_SALT = secrets.token_bytes(SALT_BYTES)


def dummy_verify() -> None:
    """Gerçek doğrulamayla aynı maliyette boş iş (kullanıcı yokken zamanlama eşitleme)."""
    _derive("zamanlama-esitleme", _DUMMY_SALT, N, R, P)


def needs_rehash(stored: str) -> bool:
    parsed = _parse(stored)
    return parsed is None or parsed[:3] != (N, R, P)


def check_password_policy(password: str, email: str | None = None) -> None:
    """Kural: 12–128 karakter, yalnızca boşluk değil, e-posta adresiyle aynı değil."""
    if not isinstance(password, str) or len(password) < MIN_LENGTH:
        raise PasswordPolicyError(f"Parola en az {MIN_LENGTH} karakter olmalıdır.")
    if len(password) > MAX_LENGTH:
        raise PasswordPolicyError(f"Parola en fazla {MAX_LENGTH} karakter olabilir.")
    if not password.strip():
        raise PasswordPolicyError("Parola yalnızca boşluktan oluşamaz.")
    if email and password.strip().lower() in {email.strip().lower(), email.strip().lower().split("@")[0]}:
        raise PasswordPolicyError("Parola e-posta adresinizle aynı olamaz.")
