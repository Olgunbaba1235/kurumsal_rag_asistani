"""Async SQLAlchemy motoru ve tenant kapsamlı oturumlar.

Çok kiracılı izolasyonun veritabanı tarafı burada uygulanır:

* `tenant_session(tenant_id)` ile açılan her oturumda, **her transaction
  başlangıcında** (`after_begin` olayı) şu ayarlar transaction kapsamlı yapılır:
    - ``set_config('app.tenant_id', <id>, true)`` → RLS politikaları bunu okur
    - ``SET LOCAL ROLE <rol>`` (yapılandırıldıysa) → RLS'e tabi role geçiş
    - ``SET LOCAL hnsw.iterative_scan = relaxed_order`` → filtreli vektör sorgusu
* `true` (is_local) parametresi sayesinde ayar transaction bitince sıfırlanır;
  havuza dönen bağlantıda başka bir tenant'a **sızmaz**.
* Ayar hiç yapılmamışsa RLS politikası NULL ile karşılaştırır ve hiçbir satır
  görünmez (fail-closed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from app.core.config import ApiSettings

logger = logging.getLogger(__name__)

TENANT_KEY = "tenant_id"
DEPTS_KEY = "departments"
ALL_DEPARTMENTS = "*"  # departman RLS'inde "şirketin tüm departmanları"
ROLE_KEY = "db_role"
VECTOR_KEY = "iterative_scan"


def create_engine(settings: ApiSettings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@event.listens_for(Session, "after_begin")
def _apply_session_scope(session: Session, transaction, connection) -> None:  # noqa: ANN001
    """Her transaction başında rol, tenant ve vektör arama ayarlarını uygular."""
    role = session.info.get(ROLE_KEY)
    if role:  # ApiSettings'te tanımlayıcı olarak doğrulandı
        connection.exec_driver_sql(f'SET LOCAL ROLE "{role}"')
    tenant_id = session.info.get(TENANT_KEY)
    if tenant_id is not None:
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
    departments = session.info.get(DEPTS_KEY)
    if tenant_id is not None and departments is not None:
        # documents / document_chunks üzerindeki department_scope politikası bunu okur;
        # ayar yoksa (ör. ham bağlantı) hiçbir belge görünmez.
        value = departments if departments == ALL_DEPARTMENTS else json.dumps(list(departments), ensure_ascii=False)
        connection.execute(text("SELECT set_config('app.departments', :d, true)"), {"d": value})
    if session.info.get(VECTOR_KEY):
        connection.exec_driver_sql("SET LOCAL hnsw.iterative_scan = relaxed_order")


class Database:
    """Motor + oturum fabrikası; uygulama yaşam döngüsü boyunca tek örnek."""

    def __init__(self, settings: ApiSettings, engine: AsyncEngine | None = None):
        self.settings = settings
        self.engine = engine or create_engine(settings)
        self._sessionmaker = create_sessionmaker(self.engine)

    def _info(self, tenant_id: uuid.UUID | None, departments: Sequence[str] | str = ALL_DEPARTMENTS) -> dict:
        info: dict = {ROLE_KEY: self.settings.db_session_role}
        if tenant_id is not None:
            info[TENANT_KEY] = tenant_id
            info[DEPTS_KEY] = departments if departments == ALL_DEPARTMENTS else tuple(departments)
            info[VECTOR_KEY] = self.settings.pgvector_iterative_scan
        return info

    @asynccontextmanager
    async def system_session(self) -> AsyncIterator[AsyncSession]:
        """Tenant kapsamı olmayan oturum: yalnızca `tenants` tablosu (API anahtarı
        doğrulama, tenant oluşturma) için. RLS'li tablolarda hiçbir satır görmez."""
        async with self._sessionmaker(info=self._info(None)) as session:
            yield session

    @asynccontextmanager
    async def tenant_session(
        self, tenant_id: uuid.UUID, departments: Sequence[str] | str = ALL_DEPARTMENTS
    ) -> AsyncIterator[AsyncSession]:
        """Tenant kapsamlı oturum. ``departments``: belge tablolarında görünür departmanlar
        (``ALL_DEPARTMENTS`` = şirketin tümü). İstek işleyen servisler her zaman isteği yapan
        kimliğin izinli departmanlarını verir (bkz. ``TenantContext.departments``)."""
        if not isinstance(tenant_id, uuid.UUID):
            raise TypeError("tenant_id bir UUID olmalıdır")
        if isinstance(departments, str) and departments != ALL_DEPARTMENTS:
            raise TypeError("departments bir liste ya da ALL_DEPARTMENTS olmalıdır")
        async with self._sessionmaker(info=self._info(tenant_id, departments)) as session:
            yield session

    async def ping(self) -> bool:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def embedding_column_type(self) -> str | None:
        async with self.engine.connect() as conn:
            return await conn.scalar(
                text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass('document_chunks') AND attname = 'embedding'"
                )
            )

    async def dispose(self) -> None:
        try:
            await asyncio.wait_for(self.engine.dispose(), timeout=self.settings.db_dispose_timeout)
        except TimeoutError:
            logger.warning("Veritabanı bağlantıları %ss içinde kapatılamadı.", self.settings.db_dispose_timeout)
