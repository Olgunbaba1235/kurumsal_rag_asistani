"""Kullanıcılar, şirket üyelikleri, davetler, oturumlar + departman düzeyinde RLS

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-22

Yalnızca ekler: mevcut tablolarda sütun değişmez, hiçbir satır güncellenmez veya silinmez.

* ``users`` RLS'siz (girişte şirket bağlamı oluşmadan e-postayla bulunur; şirket içeriği tutmaz).
* ``memberships``, ``invitations``, ``user_sessions``: FORCE RLS + ``tenant_isolation``.
* ``documents`` ve ``document_chunks``: ek ``department_scope`` politikası (RESTRICTIVE).
  Uygulama her transaction'da ``app.departments`` ayarına izinli departmanları JSON dizisi
  olarak yazar; ``'*'`` = şirketin tüm departmanları (API anahtarı, admin, sistem işleri).
  Ayar yoksa hiçbir belge görünmez (fail-closed).

Ön kontrol: departmanı şirketin departman listesinde olmayan belge varsa migration durur
(yeni politika bu belgeleri admin dışındaki herkesten gizlerdi; veri değişmeden rapor edilir).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = os.getenv("DB_APP_ROLE", "rag_app")
if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", APP_ROLE):
    raise ValueError(f"Geçersiz DB_APP_ROLE: {APP_ROLE!r}")

TENANT_TABLES = ("memberships", "invitations", "user_sessions")
DEPARTMENT_TABLES = ("documents", "document_chunks")
TENANT_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"
# '*' → tüm departmanlar; JSON dizi → yalnızca listedekiler; ayar yok/boş → hiçbiri
DEPARTMENT_PREDICATE = (
    "(current_setting('app.departments', true) = '*' OR department = ANY (ARRAY("
    "SELECT jsonb_array_elements_text(NULLIF(NULLIF(current_setting('app.departments', true), ''), '*')::jsonb))))"
)
ROLES = "('admin', 'manager', 'user')"


def _ts(name: str, nullable: bool = False, default: bool = False) -> sa.Column:
    return sa.Column(
        name, sa.DateTime(timezone=True), nullable=nullable, server_default=sa.func.now() if default else None
    )


def _precheck_orphan_departments() -> None:
    bind = op.get_bind()
    orphans = bind.execute(
        sa.text(
            "SELECT t.slug, d.department, count(*) FROM documents d JOIN tenants t ON t.id = d.tenant_id "
            "WHERE NOT (d.department = ANY (t.departments)) GROUP BY t.slug, d.department ORDER BY 1, 2"
        )
    ).all()
    if orphans:
        detail = ", ".join(f"{slug}/{dept}: {n}" for slug, dept, n in orphans)
        raise RuntimeError(
            "Migration 0003 durduruldu: departmanı şirketin departman listesinde olmayan belgeler var "
            f"({detail}). Hiçbir veri değiştirilmedi. Departmanları tenants.departments'a ekleyin ve tekrar deneyin."
        )


def upgrade() -> None:
    _precheck_orphan_departments()

    # ---------------------------------------------------------------- users
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        _ts("email_verified_at", nullable=True),
        sa.Column("failed_login_count", sa.SmallInteger(), nullable=False, server_default="0"),
        _ts("locked_until", nullable=True),
        _ts("last_login_at", nullable=True),
        _ts("password_changed_at", default=True),
        _ts("created_at", default=True),
        _ts("updated_at", default=True),
        sa.CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),
        sa.CheckConstraint("failed_login_count >= 0", name="ck_users_failed_login_count"),
    )
    op.create_index("uq_users_email", "users", ["email"], unique=True)

    # ---------------------------------------------------------------- memberships
    op.create_table(
        "memberships",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("is_owner", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("departments", sa.ARRAY(sa.String(64)), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("invited_by", sa.Uuid(), nullable=True),
        _ts("created_at", default=True),
        _ts("updated_at", default=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE", name="fk_memberships_tenant_id_tenants"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE", name="fk_memberships_user_id_users"),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="SET NULL", name="fk_memberships_invited_by_users"),
        sa.UniqueConstraint("tenant_id", "user_id", name="uq_memberships_tenant_user"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_memberships_id_tenant"),
        sa.CheckConstraint(f"role IN {ROLES}", name="ck_memberships_role"),
        sa.CheckConstraint("NOT is_owner OR role = 'admin'", name="ck_memberships_owner_is_admin"),
        sa.CheckConstraint("status IN ('active', 'suspended')", name="ck_memberships_status"),
        sa.CheckConstraint("NOT (is_owner AND status <> 'active')", name="ck_memberships_owner_active"),
    )
    op.create_index(
        "uq_memberships_owner", "memberships", ["tenant_id"], unique=True, postgresql_where=sa.text("is_owner")
    )
    op.create_index("ix_memberships_user", "memberships", ["user_id"])

    # ---------------------------------------------------------------- invitations
    op.create_table(
        "invitations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("departments", sa.ARRAY(sa.String(64)), nullable=False, server_default="{}"),
        sa.Column("grants_owner", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("token_hash", sa.CHAR(64), nullable=False),
        sa.Column("invited_by", sa.Uuid(), nullable=True),
        _ts("expires_at"),
        _ts("accepted_at", nullable=True),
        _ts("revoked_at", nullable=True),
        _ts("created_at", default=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE", name="fk_invitations_tenant_id_tenants"),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="SET NULL", name="fk_invitations_invited_by_users"),
        sa.UniqueConstraint("token_hash", name="uq_invitations_token_hash"),
        sa.CheckConstraint(f"role IN {ROLES}", name="ck_invitations_role"),
        sa.CheckConstraint("email = lower(email)", name="ck_invitations_email_lowercase"),
        sa.CheckConstraint("NOT grants_owner OR role = 'admin'", name="ck_invitations_owner_is_admin"),
    )
    op.create_index(
        "uq_invitations_open_email",
        "invitations",
        ["tenant_id", "email"],
        unique=True,
        postgresql_where=sa.text("accepted_at IS NULL AND revoked_at IS NULL"),
    )

    # ---------------------------------------------------------------- user_sessions
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("membership_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.CHAR(64), nullable=False),
        _ts("created_at", default=True),
        _ts("expires_at"),
        _ts("last_seen_at", default=True),
        _ts("revoked_at", nullable=True),
        sa.Column("revoke_reason", sa.String(32), nullable=True),
        sa.Column("user_agent", sa.String(200), nullable=True),
        # Oturumun şirketi, üyeliğin şirketiyle aynı olmak zorunda
        sa.ForeignKeyConstraint(
            ["membership_id", "tenant_id"],
            ["memberships.id", "memberships.tenant_id"],
            ondelete="CASCADE",
            name="fk_user_sessions_membership_tenant",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE", name="fk_user_sessions_user_id_users"),
        sa.UniqueConstraint("token_hash", name="uq_user_sessions_token_hash"),
    )
    op.create_index("ix_user_sessions_user", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_tenant_expires", "user_sessions", ["tenant_id", "expires_at"])

    # ---------------------------------------------------------------- Row-Level Security
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING ({TENANT_PREDICATE}) WITH CHECK ({TENANT_PREDICATE})"
        )
    for table in DEPARTMENT_TABLES:
        op.execute(
            f"CREATE POLICY department_scope ON {table} AS RESTRICTIVE "
            f"USING {DEPARTMENT_PREDICATE} WITH CHECK {DEPARTMENT_PREDICATE}"
        )

    # ---------------------------------------------------------------- Yetkiler
    # users / memberships silinmez, pasifleştirilir; oturum ve davet kayıtları temizlenebilir.
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON users, memberships TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON invitations, user_sessions TO {APP_ROLE}")
    # Yeni şirket kaydı tek transaction'da tenants satırı da oluşturur (0001'de zaten INSERT var).


def downgrade() -> None:
    for table in DEPARTMENT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS department_scope ON {table}")
    for table in ("user_sessions", "invitations", "memberships", "users"):
        op.execute(f"REVOKE ALL ON {table} FROM {APP_ROLE}")
    op.drop_table("user_sessions")
    op.drop_table("invitations")
    op.drop_table("memberships")
    op.drop_table("users")
