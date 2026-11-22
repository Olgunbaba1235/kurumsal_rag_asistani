"""P0-1 güvenlik sertleştirmesi: üretimde dokümantasyon kapalı, geliştirme parolası ve zayıf
yönetici anahtarı reddedilir, ajan uç noktası soru uzunluğu sınırını uygular."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agent.conftest import make_settings, start_agent_app

PROD_URL = "postgresql+asyncpg://rag_app:uretim-icin-uzun-parola@db:5432/rag"
STRONG_ADMIN = "x" * 40


def test_docs_and_schema_hidden_in_production():
    app = start_agent_app(settings=make_settings(ENVIRONMENT="production", DATABASE_URL=PROD_URL))
    try:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert app.client.get(path).status_code == 404, path
        assert "docs" not in app.client.get("/").json()
    finally:
        app.client.__exit__(None, None, None)


def test_docs_available_outside_production(agent_app):
    assert agent_app.client.get("/openapi.json").status_code == 200
    assert agent_app.client.get("/").json()["docs"] == "/docs"


@pytest.mark.parametrize(
    "url",
    ["postgresql+asyncpg://rag_app:rag_app_password@db:5432/rag", "postgresql+asyncpg://rag_admin:rag_admin_password@db/rag"],
)
def test_production_rejects_development_db_passwords(url):
    with pytest.raises(ValueError, match="geliştirme parolası"):
        make_settings(ENVIRONMENT="production", DATABASE_URL=url)
    with pytest.raises(ValueError, match="geliştirme parolası"):
        make_settings(ENVIRONMENT="production", DATABASE_URL=PROD_URL, DATABASE_MIGRATION_URL=url)


@pytest.mark.parametrize("key", ["kisa-anahtar", "change-me-to-a-long-random-string-xxxxxxxx"])
def test_production_rejects_weak_admin_key(key):
    with pytest.raises(ValueError, match="ADMIN_API_KEY"):
        make_settings(ENVIRONMENT="production", DATABASE_URL=PROD_URL, ADMIN_API_KEY=key)
    make_settings(ENVIRONMENT="production", DATABASE_URL=PROD_URL, ADMIN_API_KEY=STRONG_ADMIN)


def test_agent_enforces_max_question_chars(build_agent_app):
    app = build_agent_app(settings=make_settings(MAX_QUESTION_CHARS=50))
    r = app.post({"question": "a" * 51})
    assert r.status_code == 422 and "50" in r.json()["detail"]
    assert app.post({"question": "Kısa bir soru?"}).status_code == 200


def test_compose_has_no_default_passwords_and_binds_locally():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    assert "rag_admin_password" not in compose and "rag_app_password" not in compose
    assert compose.count(":?") == 4
    assert '"127.0.0.1:${POSTGRES_PORT:-5432}:5432"' in compose and '"127.0.0.1:${REDIS_PORT:-6379}:6379"' in compose
