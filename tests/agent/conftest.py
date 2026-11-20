"""`/api/v1/agent/query` için izole uygulama: veritabanı, Redis ve LLM yerine sahte servisler.

Gerçek `create_app` fabrikası, middleware'ler, bağımlılıklar (API anahtarı → tenant,
rate limit) ve yanıt şemaları kullanılır; yalnızca servis kabı sahtedir. PGlite gerekmez
(PGlite üzerindeki uçtan uca testler: `tests/api/test_agent_api.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import ApiSettings
from app.core.redis import RateLimiter
from app.main import create_app
from app.services.tenant_service import TenantContext
from tests.agent.fakes import KEY_A, KEY_B, TENANT_A, TENANT_B, MockOrchestrator


class StubDatabase:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.disposed = False

    async def embedding_column_type(self) -> str:
        return f"vector({self.dimension})"

    async def ping(self) -> None:
        return None

    async def dispose(self) -> None:
        self.disposed = True


class StubTenants:
    def __init__(self, keys: dict[str, TenantContext]):
        self.keys = keys

    async def authenticate(self, api_key: str) -> TenantContext | None:
        return self.keys.get(api_key)


class FakeRedis:
    """RateLimiter'ın kullandığı INCR/EXPIRE alt kümesi."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> bool:
        return True


@dataclass
class StubContainer:
    settings: ApiSettings
    db: StubDatabase
    tenants: StubTenants
    rate_limiter: RateLimiter
    orchestrator: Any
    redis: Any = None

    async def close(self) -> None:
        await self.db.dispose()


def make_settings(**overrides: Any) -> ApiSettings:
    base = {
        "ENVIRONMENT": "test",
        "DATABASE_URL": "postgresql+asyncpg://stub:stub@127.0.0.1:1/stub",
        "EMBEDDING_PROVIDER": "hashing",
        "LLM_PROVIDER": "none",
        "REDIS_URL": None,
        "RATE_LIMIT_PER_MINUTE": 0,
        "_env_file": None,
    }
    base.update(overrides)
    return ApiSettings(**base)


@dataclass
class AgentApp:
    client: TestClient
    container: StubContainer

    def post(self, body: dict[str, Any], key: str | None = KEY_A, **kw: Any):  # noqa: ANN201
        headers = {"X-API-Key": key} if key else {}
        return self.client.post("/api/v1/agent/query", json=body, headers=headers, **kw)


def start_agent_app(
    *,
    orchestrator: Any = None,
    rate_limit: int = 0,
    settings: ApiSettings | None = None,
) -> AgentApp:
    """Sahte servis kabıyla gerçek uygulamayı başlatır (çağıran `client.__exit__` ile kapatır)."""
    settings = settings or make_settings()
    redis = FakeRedis() if rate_limit else None
    container = StubContainer(
        settings=settings,
        db=StubDatabase(settings.embedding_dimension),
        tenants=StubTenants({KEY_A: TENANT_A, KEY_B: TENANT_B}),
        rate_limiter=RateLimiter(redis, rate_limit),
        orchestrator=orchestrator if orchestrator is not None else MockOrchestrator(),
        redis=redis,
    )
    app = create_app(settings, container_factory=lambda s: container)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return AgentApp(client, container)


@pytest.fixture
def build_agent_app() -> Iterator:
    """Fabrika: istenen orkestratör/ayarlarla izole bir uygulama başlatır."""
    opened: list[AgentApp] = []

    def _build(**kw: Any) -> AgentApp:
        opened.append(start_agent_app(**kw))
        return opened[-1]

    yield _build
    for a in opened:
        a.client.__exit__(None, None, None)


@pytest.fixture
def agent_app(build_agent_app) -> AgentApp:  # noqa: ANN001
    return build_agent_app()
