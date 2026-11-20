from __future__ import annotations

import uuid

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin
from src.config import DEFAULT_DEPARTMENTS


class Tenant(TimestampMixin, Base):
    """Kiracı (müşteri şirket). RLS uygulanmaz: kimlik doğrulama tenant'ı
    bulmak için bu tabloyu tenant bağlamı olmadan okur; satırlar yalnızca
    API anahtarı özetiyle eşleştirilir."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    api_key_preview: Mapped[str] = mapped_column(String(16), nullable=False)
    departments: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), nullable=False, default=lambda: list(DEFAULT_DEPARTMENTS)
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
