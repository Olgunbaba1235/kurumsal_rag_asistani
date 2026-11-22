"""Kayıt, giriş, oturum ve davet kabulü."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from app.api.deps import Container, CurrentPrincipal, auth_rate_limit
from app.schemas.auth import (
    AcceptInvitationRequest,
    CompanyOut,
    LoginRequest,
    MeOut,
    PasswordChangeRequest,
    RegisterRequest,
    RevokedSessions,
    SessionOut,
)
from app.schemas.common import ErrorResponse
from app.services.auth_service import IssuedSession, Principal

router = APIRouter()
UserAgent = Annotated[str | None, Header(include_in_schema=False)]
_PUBLIC = {401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}}


def me_out(p: Principal) -> MeOut:
    return MeOut(
        via=p.via.value,
        company=CompanyOut(id=p.tenant.id, slug=p.tenant.slug, name=p.tenant.name),
        user_id=p.user_id,
        membership_id=p.membership_id,
        role=p.actor.role,
        is_owner=p.actor.is_owner,
        departments=list(p.tenant.departments),
    )


def session_out(s: IssuedSession) -> SessionOut:
    return SessionOut(token=s.token, expires_at=s.expires_at, email=s.email, full_name=s.full_name, me=me_out(s.principal))


@router.post(
    "/register",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni şirket oluştur (oluşturan kişi sahip ve admin olur)",
    dependencies=[Depends(auth_rate_limit)],
    responses={**_PUBLIC, 403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def register(body: RegisterRequest, container: Container, user_agent: UserAgent = None) -> SessionOut:
    if not container.settings.allow_signup:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Yeni şirket kaydı şu anda kapalı.")
    issued = await container.auth.register(
        company_name=body.company.name,
        company_slug=body.company.slug,
        departments=body.company.departments,
        full_name=body.full_name,
        email=body.email,
        password=body.password,
        user_agent=user_agent,
    )
    return session_out(issued)


@router.post(
    "/login",
    response_model=SessionOut,
    summary="Giriş (e-posta + parola + şirket kısa adı)",
    dependencies=[Depends(auth_rate_limit)],
    responses=_PUBLIC,
)
async def login(body: LoginRequest, container: Container, user_agent: UserAgent = None) -> SessionOut:
    issued = await container.auth.login(
        email=body.email, password=body.password, company_slug=body.company_slug, user_agent=user_agent
    )
    return session_out(issued)


@router.post(
    "/accept-invitation",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Daveti kabul et ve şirkete katıl",
    dependencies=[Depends(auth_rate_limit)],
    responses={**_PUBLIC, 409: {"model": ErrorResponse}},
)
async def accept_invitation(
    body: AcceptInvitationRequest, container: Container, user_agent: UserAgent = None
) -> SessionOut:
    issued = await container.auth.accept_invitation(
        token=body.token, email=body.email, password=body.password, full_name=body.full_name, user_agent=user_agent
    )
    return session_out(issued)


@router.get("/me", response_model=MeOut, summary="Geçerli kimlik, rol ve erişilebilen departmanlar")
async def me(principal: CurrentPrincipal) -> MeOut:
    return me_out(principal)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Bu oturumu kapat")
async def logout(principal: CurrentPrincipal, container: Container) -> Response:
    await container.auth.logout(principal)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/logout-all", response_model=RevokedSessions, summary="Bu şirketteki tüm oturumlarımı kapat")
async def logout_all(principal: CurrentPrincipal, container: Container) -> RevokedSessions:
    return RevokedSessions(revoked=await container.auth.logout_all(principal))


@router.post(
    "/password",
    response_model=SessionOut,
    summary="Parola değiştir (tüm eski oturumlar kapanır, yeni oturum döner)",
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def change_password(
    body: PasswordChangeRequest, principal: CurrentPrincipal, container: Container, user_agent: UserAgent = None
) -> SessionOut:
    issued = await container.auth.change_password(
        principal, body.current_password, body.new_password, user_agent=user_agent
    )
    return session_out(issued)
