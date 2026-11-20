"""Çok kiracılı başlangıç şeması: tenants, documents, document_chunks (pgvector) + RLS

Revision ID: 0001
Revises:
Create Date: 2026-11-20
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))
APP_ROLE = os.getenv("DB_APP_ROLE", "rag_app")
if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", APP_ROLE):
    raise ValueError(f"Geçersiz DB_APP_ROLE: {APP_ROLE!r}")

TENANT_SCOPED_TABLES = ("documents", "document_chunks")
# Ayar yoksa current_setting(..., true) NULL döner → hiçbir satır eşleşmez (fail-closed)
TENANT_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("api_key_hash", sa.String(64), nullable=False),
        sa.Column("api_key_preview", sa.String(16), nullable=False),
        sa.Column("departments", postgresql.ARRAY(sa.String(64)), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
        sa.UniqueConstraint("api_key_hash", name="uq_tenants_api_key_hash"),
        sa.CheckConstraint("cardinality(departments) > 0", name="ck_tenants_has_departments"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("department", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(16), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE", name="fk_documents_tenant_id_tenants"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_documents_id_tenant"),
        sa.CheckConstraint("status IN ('processing', 'ready', 'failed')", name="ck_documents_status"),
    )
    op.create_index("ix_documents_tenant_department", "documents", ["tenant_id", "department"])
    op.create_index(
        "uq_documents_ready_filename",
        "documents",
        ["tenant_id", "department", "filename"],
        unique=True,
        postgresql_where=sa.text("status = 'ready'"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("department", sa.String(64), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("section", sa.String(200)),
        sa.Column("sheet_name", sa.String(100)),
        sa.Column("content_type", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        # Bileşik FK: parçanın tenant'ı, ait olduğu belgenin tenant'ından farklı olamaz
        sa.ForeignKeyConstraint(
            ["document_id", "tenant_id"],
            ["documents.id", "documents.tenant_id"],
            ondelete="CASCADE",
            name="fk_document_chunks_document_tenant",
        ),
    )
    op.create_index("ix_document_chunks_tenant_department", "document_chunks", ["tenant_id", "department"])
    op.create_index("ix_document_chunks_document", "document_chunks", ["document_id"])
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # ---------------------------------------------------------------- Row-Level Security
    for table in TENANT_SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # FORCE: tablo sahibi de politikalara tabi olur (superuser hariç)
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({TENANT_PREDICATE}) WITH CHECK ({TENANT_PREDICATE})"
        )

    # ---------------------------------------------------------------- Uygulama rolü
    # Docker init betiği rolü LOGIN + parolayla önceden oluşturur; yoksa NOLOGIN oluşturulur.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS;
            END IF;
        END
        $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON tenants TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON documents, document_chunks TO {APP_ROLE}")


def downgrade() -> None:
    for table in ("document_chunks", "documents", "tenants"):
        op.execute(f"REVOKE ALL ON {table} FROM {APP_ROLE}")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("tenants")
