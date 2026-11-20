"""Redis istemcisi ve tenant başına rate limiting.

Sabit pencereli (fixed-window) sayaç: ``rl:{tenant}:{dakika}`` anahtarı INCR edilir.
Redis erişilemezse istek **engellenmez** (fail-open) ve uyarı loglanır — rate
limiting bir koruma katmanıdır, veri izolasyonu değildir.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class RedisLike(Protocol):
    async def incr(self, name: str) -> int: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def ping(self) -> bool: ...


def create_redis(url: str | None) -> RedisLike | None:
    if not url:
        return None
    from redis.asyncio import Redis

    return Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0, decode_responses=True)


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class RateLimiter:
    def __init__(self, redis: RedisLike | None, limit_per_minute: int):
        self.redis = redis
        self.limit = limit_per_minute

    @property
    def enabled(self) -> bool:
        return self.redis is not None and self.limit > 0

    async def hit(self, tenant_key: str) -> RateLimitResult:
        if not self.enabled:
            return RateLimitResult(True, self.limit, self.limit, 0)
        now = int(time.time())
        window = now // 60
        key = f"rl:{tenant_key}:{window}"
        try:
            count = await self.redis.incr(key)  # type: ignore[union-attr]
            if count == 1:
                await self.redis.expire(key, 61)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — Redis hatası isteği düşürmemeli
            logger.warning("Rate limiter devre dışı (Redis hatası: %s)", exc)
            return RateLimitResult(True, self.limit, self.limit, 0)
        remaining = max(0, self.limit - count)
        return RateLimitResult(count <= self.limit, self.limit, remaining, 60 - now % 60)
