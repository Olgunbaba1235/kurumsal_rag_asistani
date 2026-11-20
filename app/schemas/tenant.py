from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel
from src.config import DEFAULT_DEPARTMENTS


class TenantCreate(BaseModel):
    name: str = Field(min_length=2, max_length=200, examples=["Örnek Holding A.Ş."])
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$", examples=["ornek-holding"])
    departments: list[str] = Field(default_factory=lambda: list(DEFAULT_DEPARTMENTS), min_length=1, max_length=50)

    @field_validator("departments")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(d.strip() for d in v if d.strip()))
        if not cleaned or any(len(d) > 64 for d in cleaned):
            raise ValueError("Departman adları boş olamaz ve en fazla 64 karakter olmalıdır.")
        return cleaned


class TenantOut(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    departments: list[str]
    api_key_preview: str
    is_active: bool
    created_at: dt.datetime


class TenantCreated(TenantOut):
    api_key: str = Field(description="Yalnızca bir kez gösterilir; güvenli bir yerde saklayın.")
