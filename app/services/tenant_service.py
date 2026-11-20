"""Kiracı yönetimi ve kimlik doğrulama."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.db import Database
from app.core.security import generate_api_key, hash_api_key, key_preview
from app.models import Tenant
from app.schemas.tenant import TenantCreate


class TenantError(Exception):
    pass


class TenantConflictError(TenantError):
    pass


class DepartmentNotAllowedError(TenantError):
    pass


@dataclass(frozen=True, slots=True)
class TenantContext:
    """İstek boyunca taşınan, değiştirilemez tenant kimliği.

    Yalnızca doğrulanmış API anahtarından üretilir; servis katmanındaki tüm
    veri erişimleri bu nesneyi zorunlu parametre olarak alır."""

    id: uuid.UUID
    slug: str
    name: str
    departments: tuple[str, ...]

    def resolve_departments(self, requested: Sequence[str] | None) -> list[str]:
        """İstenen departmanları tenant'ın tanımlı departmanlarıyla doğrular.
        Boşsa tenant'ın tüm departmanları döner."""
        if not requested:
            return list(self.departments)
        cleaned = list(dict.fromkeys(d.strip() for d in requested if d and d.strip()))
        unknown = [d for d in cleaned if d not in self.departments]
        if unknown or not cleaned:
            raise DepartmentNotAllowedError(
                f"Geçersiz departman: {', '.join(unknown) or '(boş)'}. Tanımlı departmanlar: {', '.join(self.departments)}"
            )
        return cleaned


def _context(t: Tenant) -> TenantContext:
    return TenantContext(id=t.id, slug=t.slug, name=t.name, departments=tuple(t.departments))


class TenantService:
    def __init__(self, db: Database):
        self.db = db

    async def create(self, data: TenantCreate) -> tuple[Tenant, str]:
        api_key = generate_api_key()
        tenant = Tenant(
            name=data.name,
            slug=data.slug,
            departments=data.departments,
            api_key_hash=hash_api_key(api_key),
            api_key_preview=key_preview(api_key),
            is_active=True,
        )
        async with self.db.system_session() as session:
            session.add(tenant)
            try:
                await session.commit()
            except IntegrityError as exc:
                raise TenantConflictError(f"'{data.slug}' kısa adlı tenant zaten mevcut.") from exc
            await session.refresh(tenant)
        return tenant, api_key

    async def authenticate(self, api_key: str) -> TenantContext | None:
        """API anahtarını doğrular; geçersiz veya pasif tenant için None döner."""
        if not api_key or len(api_key) > 256:
            return None
        async with self.db.system_session() as session:
            tenant = await session.scalar(
                select(Tenant).where(Tenant.api_key_hash == hash_api_key(api_key), Tenant.is_active.is_(True))
            )
        return _context(tenant) if tenant else None
