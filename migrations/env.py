"""Alembic ortamı (async).

* CLI: `alembic upgrade head` → DATABASE_MIGRATION_URL (yoksa DATABASE_URL) ile bağlanır.
  Migration'lar tablo sahibi rolle çalışmalıdır (RLS'e tabi uygulama rolüyle değil).
* Programatik kullanım (testler): `config.attributes["connection"]` verilirse o
  bağlantı kullanılır.
"""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = os.getenv("DATABASE_MIGRATION_URL") or os.getenv("DATABASE_URL")
    if not url:
        from app.core.config import ApiSettings

        url = ApiSettings(ENVIRONMENT="test").migration_url
    return url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # noqa: ANN001
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
elif (conn := config.attributes.get("connection")) is not None:
    do_run_migrations(conn)
else:
    asyncio.run(run_migrations_online())
