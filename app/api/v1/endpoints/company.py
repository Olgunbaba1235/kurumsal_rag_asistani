"""Şirket yönetimi: üyeler, davetler, sahiplik, API anahtarı.

Yetki kararları servis katmanında ``app.core.permissions`` ile verilir. Başka şirketin
üyesi veya daveti RLS nedeniyle görünmez → 404.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from app.api.deps import Container, CurrentPrincipal
from app.schemas.auth import (
    ApiKeyRotated,
    InvitationCreate,
    InvitationCreated,
    InvitationOut,
    MemberOut,
    MemberUpdate,
    OwnershipTransfer,
    RevokedSessions,
)
from app.schemas.common import ErrorResponse
from app.core.security import key_preview

router = APIRouter()
_ERR = {403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


def _inv(i) -> dict:  # noqa: ANN001
    return {
        "id": i.id, "email": i.email, "role": i.role, "departments": list(i.departments),
        "grants_owner": i.grants_owner, "expires_at": i.expires_at, "accepted_at": i.accepted_at,
        "revoked_at": i.revoked_at, "created_at": i.created_at,
    }


@router.get("/members", response_model=list[MemberOut], summary="Şirket üyeleri", responses=_ERR)
async def list_members(principal: CurrentPrincipal, container: Container) -> list[MemberOut]:
    rows = await container.auth.list_members(principal)
    return [
        MemberOut(
            membership_id=m.id, user_id=u.id, email=u.email, full_name=u.full_name, role=m.role,
            is_owner=m.is_owner, departments=list(m.departments), status=m.status, created_at=m.created_at,
        )
        for m, u in rows
    ]


@router.patch(
    "/members/{membership_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Üyenin rolünü, departmanlarını veya durumunu (askıya alma) değiştir",
    responses=_ERR,
)
async def update_member(
    membership_id: uuid.UUID, body: MemberUpdate, principal: CurrentPrincipal, container: Container
) -> Response:
    await container.auth.update_member(
        principal, membership_id, role=body.role, departments=body.departments, status=body.status
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/members/{membership_id}/revoke-sessions",
    response_model=RevokedSessions,
    summary="Üyenin tüm oturumlarını sonlandır",
    responses=_ERR,
)
async def revoke_member_sessions(
    membership_id: uuid.UUID, principal: CurrentPrincipal, container: Container
) -> RevokedSessions:
    return RevokedSessions(revoked=await container.auth.revoke_member_sessions(principal, membership_id))


@router.post(
    "/invitations",
    response_model=InvitationCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Davet oluştur (72 saat geçerli, tek kullanımlık)",
    responses={**_ERR, 409: {"model": ErrorResponse}},
)
async def create_invitation(
    body: InvitationCreate, principal: CurrentPrincipal, container: Container
) -> InvitationCreated:
    issued = await container.auth.invite(
        principal, email=body.email, role=body.role, departments=body.departments, grants_owner=body.grants_owner
    )
    return InvitationCreated(**_inv(issued.invitation), token=issued.token)


@router.get("/invitations", response_model=list[InvitationOut], summary="Davetler", responses=_ERR)
async def list_invitations(principal: CurrentPrincipal, container: Container) -> list[InvitationOut]:
    return [InvitationOut(**_inv(i)) for i in await container.auth.list_invitations(principal)]


@router.delete(
    "/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Daveti iptal et", responses=_ERR
)
async def revoke_invitation(invitation_id: uuid.UUID, principal: CurrentPrincipal, container: Container) -> Response:
    await container.auth.revoke_invitation(principal, invitation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/transfer-ownership",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Sahipliği aktif bir admin'e devret",
    responses=_ERR,
)
async def transfer_ownership(body: OwnershipTransfer, principal: CurrentPrincipal, container: Container) -> Response:
    await container.auth.transfer_ownership(principal, body.membership_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/api-key/rotate",
    response_model=ApiKeyRotated,
    summary="Şirket API anahtarını yenile (yalnızca sahip)",
    responses=_ERR,
)
async def rotate_api_key(principal: CurrentPrincipal, container: Container) -> ApiKeyRotated:
    key = await container.auth.rotate_api_key(principal)
    return ApiKeyRotated(api_key=key, api_key_preview=key_preview(key))
