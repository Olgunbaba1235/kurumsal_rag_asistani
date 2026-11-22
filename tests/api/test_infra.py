"""Veritabanı gerektirmeyen altyapı testleri: rate limit, güvenlik yardımcıları, yapılandırma."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import ApiSettings
from app.core.redis import RateLimiter
from app.core.security import constant_time_equals, generate_api_key, hash_api_key


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store: dict[str, int] = {}
        self.fail = fail

    async def incr(self, name):
        if self.fail:
            raise ConnectionError("redis yok")
        self.store[name] = self.store.get(name, 0) + 1
        return self.store[name]

    async def expire(self, name, time):
        return True

    async def ping(self):
        return True


def test_rate_limiter_blocks_after_limit_per_tenant():
    rl = RateLimiter(FakeRedis(), limit_per_minute=2)

    async def go():
        return [await rl.hit("a"), await rl.hit("a"), await rl.hit("a"), await rl.hit("b")]

    a1, a2, a3, b1 = asyncio.run(go())
    assert a1.allowed and a2.allowed and not a3.allowed and a3.retry_after > 0
    assert b1.allowed  # sayaçlar tenant başına


def test_rate_limiter_fails_open_when_redis_down():
    rl = RateLimiter(FakeRedis(fail=True), limit_per_minute=1)
    assert asyncio.run(rl.hit("a")).allowed


def test_api_keys_are_random_and_hashed():
    k1, k2 = generate_api_key(), generate_api_key()
    assert k1 != k2 and k1.startswith("rag_") and len(k1) > 40
    assert hash_api_key(k1) != k1 and len(hash_api_key(k1)) == 64
    assert constant_time_equals("abc", "abc") and not constant_time_equals("abc", None)


def test_settings_require_key_for_gemini_outside_tests():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        ApiSettings(ENVIRONMENT="production", EMBEDDING_PROVIDER="gemini", _env_file=None)
    ApiSettings(ENVIRONMENT="production", EMBEDDING_PROVIDER="hashing", LLM_PROVIDER="none", _env_file=None,
                DATABASE_URL="postgresql+asyncpg://rag_app:uretim-parolasi@db:5432/rag")


def test_session_role_must_be_identifier():
    with pytest.raises(ValueError):
        ApiSettings(ENVIRONMENT="test", DB_SESSION_ROLE='rag_app"; DROP TABLE x; --', _env_file=None)
