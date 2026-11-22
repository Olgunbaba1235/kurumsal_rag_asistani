from __future__ import annotations

import datetime as dt
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.core.permissions import Role
from app.schemas.tenant import TenantCreate

Password = Field(min_length=1, max_length=128, description="Kural: 12–128 karakter, e-postayla aynı olamaz.")
Email = Field(min_length=3, max_length=254, examples=["ayse.yilmaz@ornek.com"])


class RegisterRequest(BaseModel):
    company: TenantCreate
    full_name: str = Field(min_length=2, max_length=200)
    email: str = Email
    password: str = Password


class LoginRequest(BaseModel):
    email: str = Email
    password: str = Password
    company_slug: str = Field(min_length=2, max_length=64, examples=["ornek-holding"])


class AcceptInvitationRequest(BaseModel):
    token: str = Field(min_length=10, max_length=128)
    email: str = Email
    password: str = Password
    full_name: str | None = Field(default=None, max_length=200, description="Yeni hesap için zorunlu")


class PasswordChangeRequest(BaseModel):
    current_password: str = Password
    new_password: str = Password


class CompanyOut(BaseModel):
    id: uuid.UUID
    slug: str
    name: str


class MeOut(BaseModel):
    via: Literal["session", "api_key"]
    company: CompanyOut
    user_id: uuid.UUID | None
    membership_id: uuid.UUID | None
    role: Role
    is_owner: bool
    departments: list[str] = Field(description="Erişilebilen departmanlar")


class SessionOut(BaseModel):
    token: str = Field(description="Yalnızca bir kez gösterilir. `Authorization: Bearer <token>` ile kullanın.")
    expires_at: dt.datetime
    email: str
    full_name: str
    me: MeOut


class InvitationCreate(BaseModel):
    email: str = Email
    role: Role = Role.USER
    departments: list[str] = Field(default_factory=list, max_length=50)
    grants_owner: bool = Field(default=False, description="Yalnızca sahibi olmayan şirketlerde, API anahtarıyla")


class InvitationOut(BaseModel):
    id: uuid.UUID
    email: str
    role: Role
    departments: list[str]
    grants_owner: bool
    expires_at: dt.datetime
    accepted_at: dt.datetime | None
    revoked_at: dt.datetime | None
    created_at: dt.datetime


class InvitationCreated(InvitationOut):
    token: str = Field(description="Davet belirteci; yalnızca bir kez gösterilir, davet edilen kişiye iletin.")


class MemberOut(BaseModel):
    membership_id: uuid.UUID
    user_id: uuid.UUID
    email: str
    full_name: str
    role: Role
    is_owner: bool
    departments: list[str]
    status: Literal["active", "suspended"]
    created_at: dt.datetime


class MemberUpdate(BaseModel):
    role: Role | None = None
    departments: list[str] | None = Field(default=None, max_length=50)
    status: Literal["active", "suspended"] | None = None


class OwnershipTransfer(BaseModel):
    membership_id: uuid.UUID


class RevokedSessions(BaseModel):
    revoked: int


class ApiKeyRotated(BaseModel):
    api_key: str = Field(description="Yalnızca bir kez gösterilir. Eski anahtar hemen geçersiz olur.")
    api_key_preview: str
