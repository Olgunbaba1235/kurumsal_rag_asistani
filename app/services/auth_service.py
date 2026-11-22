"""Kullanıcı kimliği: kayıt, giriş, oturum, davet ve şirket üyelik yönetimi.

Güvenlik kuralları (teknik plan "Oturum oluşturma, süre ve iptal" + "8 güvenlik kontrolü"):

* Şirket yalnızca doğrulanmış kimlikten çözülür. Oturum/davet belirteci şirket kimliğini taşır ve
  özeti **yalnızca o şirketin RLS bağlamında** aranır.
* Kullanıcının üye olduğu şirketleri listeleyen bir işlem yoktur; giriş şirket kısa adı ister.
* Giriş hatalarında tek, genel mesaj; kullanıcı yoksa da scrypt maliyeti ödenir.
* Oturum geçerliliği her istekte veritabanından okunur, önbelleğe alınmaz; iptal geri alınamaz.
* Davet kabulü tek koşullu UPDATE ile yapılır → tek kullanımlık, eşzamanlı kabulde yalnızca biri geçer.
* Yetki kararları yalnızca ``app.core.permissions`` ile verilir.
* Log'lara e-posta, parola veya belirteç yazılmaz; kimlik ``user:<uuid>`` olarak geçer.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.core.config import ApiSettings
from app.core.db import ALL_DEPARTMENTS, Database
from app.core.passwords import PasswordPolicyError, check_password_policy, hash_password, verify_password
from app.core.permissions import (
    Action,
    Actor,
    Role,
    Target,
    Via,
    allowed_departments,
    can,
    can_invite,
    can_manage_member,
    can_revoke_sessions,
    can_transfer_ownership,
)
from app.core.security import generate_api_key, hash_api_key, key_preview
from app.core.tokens import INVITE_PREFIX, SESSION_PREFIX, new_token, parse_token
from app.models import Invitation, Membership, Tenant, User, UserSession
from app.services.tenant_service import TenantContext

logger = logging.getLogger(__name__)

LOGIN_FAILED = "E-posta, parola veya şirket hatalı."
INVITE_INVALID = "Davet geçersiz, kullanılmış veya süresi dolmuş."
LAST_SEEN_RESOLUTION = dt.timedelta(seconds=60)


# ====================================================================== hatalar
class AuthServiceError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthenticationFailed(AuthServiceError):
    status_code = 401


class PermissionDenied(AuthServiceError):
    status_code = 403


class NotFound(AuthServiceError):
    status_code = 404


class Conflict(AuthServiceError):
    status_code = 409


class InvalidInput(AuthServiceError):
    status_code = 422


# ====================================================================== değer nesneleri
@dataclass(frozen=True, slots=True)
class Principal:
    """İsteği yapan doğrulanmış kimlik. ``tenant`` izinli departmanlara daraltılmıştır."""

    tenant: TenantContext
    actor: Actor
    user_id: uuid.UUID | None = None
    membership_id: uuid.UUID | None = None
    session_id: uuid.UUID | None = None

    @property
    def via(self) -> Via:
        return self.actor.via

    @property
    def label(self) -> str:
        return f"user:{self.user_id}" if self.user_id else "api_key"


@dataclass(frozen=True, slots=True)
class IssuedSession:
    token: str
    expires_at: dt.datetime
    principal: Principal
    email: str
    full_name: str


@dataclass(frozen=True, slots=True)
class IssuedInvitation:
    invitation: Invitation
    token: str


def normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    local, _, domain = e.partition("@")
    if not local or "." not in domain or len(e) > 254 or any(c.isspace() for c in e):
        raise InvalidInput("Geçerli bir e-posta adresi girin.")
    return e


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _tenant_ctx(t: Tenant) -> TenantContext:
    depts = tuple(t.departments)
    return TenantContext(id=t.id, slug=t.slug, name=t.name, departments=depts, all_departments=depts)


def _actor(m: Membership) -> Actor:
    return Actor(via=Via.SESSION, role=Role(m.role), is_owner=m.is_owner, departments=tuple(m.departments))


class AuthService:
    def __init__(self, db: Database, settings: ApiSettings, clock: Callable[[], dt.datetime] = _utcnow):
        self.db = db
        self.settings = settings
        self.clock = clock

    # ------------------------------------------------------------ yardımcılar
    @property
    def session_ttl(self) -> dt.timedelta:
        return dt.timedelta(hours=self.settings.session_ttl_hours)

    @property
    def session_idle(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.settings.session_idle_minutes)

    def _principal(self, tenant: Tenant, m: Membership, session_id: uuid.UUID | None) -> Principal:
        ctx = _tenant_ctx(tenant)
        actor = _actor(m)
        return Principal(
            tenant=ctx.scoped(allowed_departments(actor, ctx.defined_departments)),
            actor=actor,
            user_id=m.user_id,
            membership_id=m.id,
            session_id=session_id,
        )

    def _new_session(self, tenant_id: uuid.UUID, m: Membership, user_agent: str | None) -> tuple[UserSession, str]:
        now = self.clock()
        token, digest = new_token(SESSION_PREFIX, tenant_id)
        s = UserSession(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            membership_id=m.id,
            user_id=m.user_id,
            token_hash=digest,
            created_at=now,
            expires_at=now + self.session_ttl,
            last_seen_at=now,
            user_agent=(user_agent or "")[:200] or None,
        )
        return s, token

    async def _hash(self, password: str) -> str:
        return await asyncio.to_thread(hash_password, password)

    async def _verify(self, password: str, stored: str | None) -> bool:
        return await asyncio.to_thread(verify_password, password, stored)

    def _check_departments(self, ctx: TenantContext, departments: Sequence[str]) -> list[str]:
        cleaned = list(dict.fromkeys(d.strip() for d in departments if d and d.strip()))
        unknown = [d for d in cleaned if d not in ctx.defined_departments]
        if unknown:
            raise InvalidInput(f"Şirkette tanımlı olmayan departman: {', '.join(unknown)}")
        return cleaned

    async def _tenant_by_slug(self, slug: str) -> Tenant | None:
        async with self.db.system_session() as s:
            return await s.scalar(select(Tenant).where(Tenant.slug == slug, Tenant.is_active.is_(True)))

    async def _tenant_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        async with self.db.system_session() as s:
            return await s.scalar(select(Tenant).where(Tenant.id == tenant_id, Tenant.is_active.is_(True)))

    # ------------------------------------------------------------ kayıt
    async def register(
        self,
        *,
        company_name: str,
        company_slug: str,
        departments: Sequence[str],
        full_name: str,
        email: str,
        password: str,
        user_agent: str | None = None,
    ) -> IssuedSession:
        """Yeni şirket + sahibi. Mevcut bir kullanıcı ancak doğru parolasıyla yeni şirket açabilir."""
        email = normalize_email(email)
        full_name = (full_name or "").strip()
        if not 2 <= len(full_name) <= 200:
            raise InvalidInput("Ad soyad 2–200 karakter olmalıdır.")
        async with self.db.system_session() as s:
            existing = await s.scalar(select(User).where(User.email == email))
        if existing is not None:
            if not existing.is_active or not await self._verify(password, existing.password_hash):
                raise AuthenticationFailed("Bu e-posta zaten kayıtlı; mevcut parolanızla devam edin.")
        else:
            try:
                check_password_policy(password, email)
            except PasswordPolicyError as exc:
                raise InvalidInput(str(exc)) from exc
            password_hash = await self._hash(password)

        tenant_id = uuid.uuid4()
        api_key = generate_api_key()  # gösterilmez; sahip isterse yeniler
        now = self.clock()
        async with self.db.tenant_session(tenant_id, ALL_DEPARTMENTS) as s:
            tenant = Tenant(
                id=tenant_id,
                name=company_name,
                slug=company_slug,
                departments=list(departments),
                api_key_hash=hash_api_key(api_key),
                api_key_preview=key_preview(api_key),
                is_active=True,
            )
            s.add(tenant)
            if existing is None:
                user = User(id=uuid.uuid4(), email=email, password_hash=password_hash, full_name=full_name,
                            password_changed_at=now)
                s.add(user)
            else:
                user = existing
            try:
                await s.flush()
            except IntegrityError as exc:
                await s.rollback()
                raise Conflict(f"'{company_slug}' kısa adı kullanımda; başka bir kısa ad seçin.") from exc
            m = Membership(id=uuid.uuid4(), tenant_id=tenant_id, user_id=user.id, role=Role.ADMIN.value,
                           is_owner=True, departments=[], status="active")
            s.add(m)
            await s.flush()
            sess, token = self._new_session(tenant_id, m, user_agent)
            s.add(sess)
            await s.commit()
            principal = self._principal(tenant, m, sess.id)
        logger.info("Yeni şirket kaydı [tenant=%s, %s]", company_slug, principal.label)
        return IssuedSession(token, sess.expires_at, principal, user.email, user.full_name)

    # ------------------------------------------------------------ giriş
    async def login(
        self, *, email: str, password: str, company_slug: str, user_agent: str | None = None
    ) -> IssuedSession:
        try:
            email = normalize_email(email)
        except InvalidInput:
            await self._verify(password, None)
            raise AuthenticationFailed(LOGIN_FAILED) from None
        tenant = await self._tenant_by_slug(company_slug)
        async with self.db.system_session() as s:
            user = await s.scalar(select(User).where(User.email == email))
        if user is None:
            await self._verify(password, None)  # zamanlama eşitleme
            logger.info("Başarısız giriş [tenant=%s, neden=unknown_user]", company_slug)
            raise AuthenticationFailed(LOGIN_FAILED)

        now = self.clock()
        if user.locked_until is not None and user.locked_until > now:
            await self._verify(password, None)
            logger.info("Başarısız giriş [%s, neden=locked]", f"user:{user.id}")
            raise AuthenticationFailed(LOGIN_FAILED)

        if not await self._verify(password, user.password_hash):
            await self._register_failure(user.id)
            logger.info("Başarısız giriş [%s, neden=bad_password]", f"user:{user.id}")
            raise AuthenticationFailed(LOGIN_FAILED)

        if not user.is_active or tenant is None:
            logger.info("Başarısız giriş [%s, neden=inactive_or_unknown_tenant]", f"user:{user.id}")
            raise AuthenticationFailed(LOGIN_FAILED)

        async with self.db.tenant_session(tenant.id, ALL_DEPARTMENTS) as s:
            m = await s.scalar(
                select(Membership).where(
                    Membership.tenant_id == tenant.id, Membership.user_id == user.id, Membership.status == "active"
                )
            )
            if m is None:
                logger.info("Başarısız giriş [%s, neden=no_membership]", f"user:{user.id}")
                raise AuthenticationFailed(LOGIN_FAILED)
            await s.execute(
                update(User).where(User.id == user.id).values(failed_login_count=0, locked_until=None, last_login_at=now)
            )
            sess, token = self._new_session(tenant.id, m, user_agent)
            s.add(sess)
            await s.commit()
            principal = self._principal(tenant, m, sess.id)
        logger.info("Giriş [tenant=%s, %s]", tenant.slug, principal.label)
        return IssuedSession(token, sess.expires_at, principal, user.email, user.full_name)

    async def _register_failure(self, user_id: uuid.UUID) -> None:
        now = self.clock()
        async with self.db.system_session() as s, s.begin():
            user = await s.scalar(select(User).where(User.id == user_id).with_for_update())
            count = user.failed_login_count + 1
            if count >= self.settings.login_max_failures:
                user.failed_login_count = 0
                user.locked_until = now + dt.timedelta(minutes=self.settings.login_lockout_minutes)
            else:
                user.failed_login_count = count

    # ------------------------------------------------------------ oturum doğrulama
    async def authenticate_session(self, token: str | None) -> Principal | None:
        """Her istekte: özet o şirkette var, iptal yok, mutlak ve hareketsizlik süresi dolmamış,
        şifre oturumdan sonra değişmemiş, üyelik/kullanıcı/şirket aktif. Biri tutmazsa None."""
        parsed = parse_token(token, SESSION_PREFIX)
        if parsed is None:
            return None
        tenant = await self._tenant_by_id(parsed.tenant_id)
        if tenant is None:
            return None
        now = self.clock()
        async with self.db.tenant_session(tenant.id, ALL_DEPARTMENTS) as s:
            sess = await s.scalar(select(UserSession).where(UserSession.token_hash == parsed.token_hash))
            if (
                sess is None
                or sess.tenant_id != tenant.id
                or sess.revoked_at is not None
                or sess.expires_at <= now
                or sess.last_seen_at + self.session_idle <= now
            ):
                return None
            m = await s.scalar(select(Membership).where(Membership.id == sess.membership_id))
            user = await s.scalar(select(User).where(User.id == sess.user_id))
            if (
                m is None
                or m.status != "active"
                or m.user_id != sess.user_id
                or user is None
                or not user.is_active
                or sess.created_at < user.password_changed_at
            ):
                return None
            if now - sess.last_seen_at >= LAST_SEEN_RESOLUTION:
                await s.execute(update(UserSession).where(UserSession.id == sess.id).values(last_seen_at=now))
                await s.commit()
            return self._principal(tenant, m, sess.id)

    async def _revoke(self, tenant_id: uuid.UUID, reason: str, *conditions) -> int:  # noqa: ANN002
        async with self.db.tenant_session(tenant_id, ALL_DEPARTMENTS) as s, s.begin():
            r = await s.execute(
                update(UserSession)
                .where(UserSession.revoked_at.is_(None), *conditions)
                .values(revoked_at=self.clock(), revoke_reason=reason)
            )
        return r.rowcount or 0

    async def logout(self, p: Principal) -> None:
        if p.session_id is not None:
            await self._revoke(p.tenant.id, "logout", UserSession.id == p.session_id)

    async def logout_all(self, p: Principal) -> int:
        if p.user_id is None:
            return 0
        return await self._revoke(p.tenant.id, "logout_all", UserSession.user_id == p.user_id)

    async def change_password(
        self, p: Principal, current_password: str, new_password: str, user_agent: str | None = None
    ) -> IssuedSession:
        """Parola değişince kullanıcının TÜM şirketlerdeki eski oturumları geçersizleşir
        (``password_changed_at`` kontrolü); bu şirket için yeni oturum açılır."""
        if p.user_id is None:
            raise PermissionDenied("API anahtarıyla parola değiştirilemez.")
        async with self.db.system_session() as s:
            user = await s.scalar(select(User).where(User.id == p.user_id))
        if user is None or not await self._verify(current_password, user.password_hash):
            raise AuthenticationFailed("Mevcut parola hatalı.")
        try:
            check_password_policy(new_password, user.email)
        except PasswordPolicyError as exc:
            raise InvalidInput(str(exc)) from exc
        new_hash = await self._hash(new_password)
        now = self.clock()
        async with self.db.system_session() as s, s.begin():
            await s.execute(
                update(User).where(User.id == p.user_id).values(
                    password_hash=new_hash, password_changed_at=now, failed_login_count=0, locked_until=None
                )
            )
        await self._revoke(p.tenant.id, "password_change", UserSession.user_id == p.user_id)
        tenant = await self._tenant_by_id(p.tenant.id)
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s:
            m = await s.scalar(select(Membership).where(Membership.id == p.membership_id))
            sess, token = self._new_session(p.tenant.id, m, user_agent)
            s.add(sess)
            await s.commit()
        logger.info("Parola değişti [%s]", p.label)
        return IssuedSession(token, sess.expires_at, self._principal(tenant, m, sess.id), user.email, user.full_name)

    # ------------------------------------------------------------ davetler
    async def invite(
        self, p: Principal, *, email: str, role: Role, departments: Sequence[str], grants_owner: bool = False
    ) -> IssuedInvitation:
        email = normalize_email(email)
        depts = [] if role is Role.ADMIN else self._check_departments(p.tenant, departments)
        now = self.clock()
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s:
            has_owner = (
                await s.scalar(select(Membership.id).where(Membership.is_owner.is_(True)).limit(1))
            ) is not None
            if not can_invite(p.actor, role, grants_owner=grants_owner, tenant_has_owner=has_owner):
                raise PermissionDenied("Bu rolde davet oluşturma yetkiniz yok.")
            member = await s.scalar(
                select(Membership.id).join(User, User.id == Membership.user_id).where(User.email == email)
            )
            if member is not None:
                raise Conflict("Bu kişi zaten şirketin üyesi.")
            # Aynı kişiye açık eski davet varsa iptal edilir (tek açık davet kuralı)
            await s.execute(
                update(Invitation)
                .where(Invitation.email == email, Invitation.accepted_at.is_(None), Invitation.revoked_at.is_(None))
                .values(revoked_at=now)
            )
            token, digest = new_token(INVITE_PREFIX, p.tenant.id)
            inv = Invitation(
                id=uuid.uuid4(),
                tenant_id=p.tenant.id,
                email=email,
                role=role.value,
                departments=depts,
                grants_owner=grants_owner,
                token_hash=digest,
                invited_by=p.user_id,
                expires_at=now + dt.timedelta(hours=self.settings.invitation_ttl_hours),
                created_at=now,
            )
            s.add(inv)
            await s.commit()
        logger.info("Davet oluşturuldu [tenant=%s, %s, rol=%s]", p.tenant.slug, p.label, role.value)
        return IssuedInvitation(inv, token)

    async def list_invitations(self, p: Principal) -> list[Invitation]:
        self._require(p, Action.MEMBERS_LIST)
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s:
            return list((await s.scalars(select(Invitation).order_by(Invitation.created_at.desc()))).all())

    async def revoke_invitation(self, p: Principal, invitation_id: uuid.UUID) -> None:
        self._require(p, Action.MEMBERS_LIST)
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s, s.begin():
            r = await s.execute(
                update(Invitation)
                .where(Invitation.id == invitation_id, Invitation.accepted_at.is_(None), Invitation.revoked_at.is_(None))
                .values(revoked_at=self.clock())
            )
        if not r.rowcount:
            raise NotFound("Açık davet bulunamadı.")

    async def accept_invitation(
        self, *, token: str, email: str, password: str, full_name: str | None = None, user_agent: str | None = None
    ) -> IssuedSession:
        parsed = parse_token(token, INVITE_PREFIX)
        if parsed is None:
            raise AuthenticationFailed(INVITE_INVALID)
        email = normalize_email(email)
        tenant = await self._tenant_by_id(parsed.tenant_id)
        if tenant is None:
            raise AuthenticationFailed(INVITE_INVALID)
        now = self.clock()
        async with self.db.system_session() as s:
            existing = await s.scalar(select(User).where(User.email == email))
        if existing is not None:
            if not existing.is_active or not await self._verify(password, existing.password_hash):
                raise AuthenticationFailed("Bu e-posta zaten kayıtlı; mevcut parolanızı girin.")
            password_hash = None
        else:
            try:
                check_password_policy(password, email)
            except PasswordPolicyError as exc:
                raise InvalidInput(str(exc)) from exc
            name = (full_name or "").strip()
            if not 2 <= len(name) <= 200:
                raise InvalidInput("Ad soyad 2–200 karakter olmalıdır.")
            password_hash = await self._hash(password)

        async with self.db.tenant_session(tenant.id, ALL_DEPARTMENTS) as s:
            # Tek koşullu UPDATE: tek kullanımlık + süreli; eşzamanlı ikinci kabul 0 satır görür
            inv = (
                await s.execute(
                    update(Invitation)
                    .where(
                        Invitation.token_hash == parsed.token_hash,
                        Invitation.accepted_at.is_(None),
                        Invitation.revoked_at.is_(None),
                        Invitation.expires_at > now,
                        Invitation.email == email,
                    )
                    .values(accepted_at=now)
                    .returning(Invitation)
                )
            ).scalar_one_or_none()
            if inv is None:
                await s.rollback()
                raise AuthenticationFailed(INVITE_INVALID)
            if existing is None:
                user = User(id=uuid.uuid4(), email=email, password_hash=password_hash, full_name=name,
                            password_changed_at=now)
                s.add(user)
                await s.flush()
            else:
                user = existing
                if await s.scalar(select(Membership.id).where(Membership.user_id == user.id)) is not None:
                    await s.rollback()
                    raise Conflict("Bu şirkette zaten bir üyeliğiniz var.")
            has_owner = (
                await s.scalar(select(Membership.id).where(Membership.is_owner.is_(True)).limit(1))
            ) is not None
            m = Membership(
                id=uuid.uuid4(),
                tenant_id=tenant.id,
                user_id=user.id,
                role=inv.role,
                is_owner=bool(inv.grants_owner and not has_owner),
                departments=list(inv.departments),
                status="active",
                invited_by=inv.invited_by,
            )
            s.add(m)
            await s.flush()
            sess, session_token = self._new_session(tenant.id, m, user_agent)
            s.add(sess)
            await s.commit()
            principal = self._principal(tenant, m, sess.id)
        logger.info("Davet kabul edildi [tenant=%s, %s]", tenant.slug, principal.label)
        return IssuedSession(session_token, sess.expires_at, principal, user.email, user.full_name)

    # ------------------------------------------------------------ üye yönetimi
    @staticmethod
    def _require(p: Principal, action: Action) -> None:
        if not can(p.actor, action):
            raise PermissionDenied("Bu işlem için yetkiniz yok.")

    async def list_members(self, p: Principal) -> list[tuple[Membership, User]]:
        self._require(p, Action.MEMBERS_LIST)
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s:
            rows = await s.execute(
                select(Membership, User).join(User, User.id == Membership.user_id).order_by(User.email)
            )
            return [(m, u) for m, u in rows.all()]

    async def _target(self, s, p: Principal, membership_id: uuid.UUID) -> tuple[Membership, Target]:  # noqa: ANN001
        m = await s.scalar(select(Membership).where(Membership.id == membership_id).with_for_update())
        if m is None:  # başka şirketin üyeliği de RLS nedeniyle burada görünmez → 404
            raise NotFound("Üye bulunamadı.")
        return m, Target(role=Role(m.role), is_owner=m.is_owner, is_self=m.user_id == p.user_id)

    async def update_member(
        self,
        p: Principal,
        membership_id: uuid.UUID,
        *,
        role: Role | None = None,
        departments: Sequence[str] | None = None,
        status: str | None = None,
    ) -> Membership:
        if status not in (None, "active", "suspended"):
            raise InvalidInput("Durum 'active' veya 'suspended' olmalıdır.")
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s, s.begin():
            m, target = await self._target(s, p, membership_id)
            if not can_manage_member(p.actor, target, new_role=role):
                raise PermissionDenied("Bu üyeyi bu şekilde yönetme yetkiniz yok.")
            if role is not None:
                m.role = role.value
            if departments is not None:
                m.departments = self._check_departments(p.tenant, departments)
            if status is not None and status != m.status:
                m.status = status
                if status == "suspended":
                    await s.execute(
                        update(UserSession)
                        .where(UserSession.membership_id == m.id, UserSession.revoked_at.is_(None))
                        .values(revoked_at=self.clock(), revoke_reason="suspended")
                    )
        logger.info("Üye güncellendi [tenant=%s, %s → membership:%s]", p.tenant.slug, p.label, membership_id)
        return m

    async def revoke_member_sessions(self, p: Principal, membership_id: uuid.UUID) -> int:
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s:
            m, target = await self._target(s, p, membership_id)
            await s.rollback()
        if not can_revoke_sessions(p.actor, target):
            raise PermissionDenied("Bu üyenin oturumlarını sonlandırma yetkiniz yok.")
        return await self._revoke(p.tenant.id, "admin_revoke", UserSession.membership_id == membership_id)

    async def transfer_ownership(self, p: Principal, membership_id: uuid.UUID) -> None:
        async with self.db.tenant_session(p.tenant.id, ALL_DEPARTMENTS) as s, s.begin():
            m, target = await self._target(s, p, membership_id)
            if not can_transfer_ownership(p.actor, target, target_active=m.status == "active"):
                raise PermissionDenied("Sahiplik yalnızca sahip tarafından, aktif bir admin'e devredilebilir.")
            await s.execute(update(Membership).where(Membership.id == p.membership_id).values(is_owner=False))
            await s.execute(update(Membership).where(Membership.id == m.id).values(is_owner=True))
        logger.info("Sahiplik devredildi [tenant=%s, %s → membership:%s]", p.tenant.slug, p.label, membership_id)

    async def rotate_api_key(self, p: Principal) -> str:
        self._require(p, Action.ROTATE_API_KEY)
        api_key = generate_api_key()
        async with self.db.system_session() as s, s.begin():
            await s.execute(
                update(Tenant)
                .where(Tenant.id == p.tenant.id)
                .values(api_key_hash=hash_api_key(api_key), api_key_preview=key_preview(api_key))
            )
        logger.info("API anahtarı yenilendi [tenant=%s, %s]", p.tenant.slug, p.label)
        return api_key
