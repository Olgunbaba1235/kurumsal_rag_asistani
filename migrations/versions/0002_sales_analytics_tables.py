"""Satış analitiği tabloları (customers, products, orders, order_items) + RLS + salt-okunur rol

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-20
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = os.getenv("DB_APP_ROLE", "rag_app")
READONLY_ROLE = os.getenv("ANALYTICS_READONLY_ROLE", "rag_readonly")
for _role in (APP_ROLE, READONLY_ROLE):
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", _role):
        raise ValueError(f"Geçersiz rol adı: {_role!r}")

TABLES = ("customers", "products", "orders", "order_items")
TENANT_PREDICATE = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def _tenant_fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE", name=f"fk_{table}_tenant_id_tenants")


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("segment", sa.String(40), nullable=False),
        sa.Column("city", sa.String(60), nullable=False),
        sa.Column("region", sa.String(60), nullable=False),
        sa.Column("credit_limit", sa.Numeric(14, 2), nullable=False),
        sa.Column("payment_terms_days", sa.Integer(), nullable=False),
        sa.Column("since", sa.Date(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.PrimaryKeyConstraint("id", name="pk_customers"),
        _tenant_fk("customers"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_customers_id_tenant"),
        sa.UniqueConstraint("tenant_id", "code", name="uq_customers_tenant_code"),
    )
    op.create_index("ix_customers_tenant_segment", "customers", ["tenant_id", "segment"])

    op.create_table(
        "products",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("sku", sa.String(30), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("category", sa.String(80), nullable=False),
        sa.Column("unit", sa.String(10), nullable=False),
        sa.Column("unit_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("unit_cost", sa.Numeric(12, 2), nullable=False),
        sa.Column("stock_qty", sa.Integer(), nullable=False),
        sa.Column("reorder_level", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.PrimaryKeyConstraint("id", name="pk_products"),
        _tenant_fk("products"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_products_id_tenant"),
        sa.UniqueConstraint("tenant_id", "sku", name="uq_products_tenant_sku"),
        sa.CheckConstraint("unit_price >= 0 AND unit_cost >= 0", name="ck_products_non_negative_prices"),
    )
    op.create_index("ix_products_tenant_category", "products", ["tenant_id", "category"])

    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("order_no", sa.String(30), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column("order_date", sa.Date(), nullable=False),
        sa.Column("shipped_date", sa.Date()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("discount_rate", sa.Numeric(5, 2), nullable=False, server_default="0"),
        sa.Column("total_amount", sa.Numeric(14, 2), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_orders"),
        _tenant_fk("orders"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_orders_id_tenant"),
        sa.UniqueConstraint("tenant_id", "order_no", name="uq_orders_tenant_order_no"),
        # Bileşik FK: siparişin müşterisi aynı tenant'a ait olmak zorunda
        sa.ForeignKeyConstraint(
            ["customer_id", "tenant_id"], ["customers.id", "customers.tenant_id"], name="fk_orders_customer_tenant"
        ),
        sa.CheckConstraint(
            "status IN ('teslim_edildi', 'sevk_edildi', 'hazirlaniyor', 'iptal')", name="ck_orders_valid_status"
        ),
    )
    op.create_index("ix_orders_tenant_date", "orders", ["tenant_id", "order_date"])
    op.create_index("ix_orders_tenant_customer", "orders", ["tenant_id", "customer_id"])

    op.create_table(
        "order_items",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("line_total", sa.Numeric(14, 2), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_order_items"),
        sa.ForeignKeyConstraint(
            ["order_id", "tenant_id"], ["orders.id", "orders.tenant_id"], ondelete="CASCADE",
            name="fk_order_items_order_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["product_id", "tenant_id"], ["products.id", "products.tenant_id"], name="fk_order_items_product_tenant"
        ),
        sa.CheckConstraint("quantity > 0", name="ck_order_items_positive_quantity"),
    )
    op.create_index("ix_order_items_tenant_order", "order_items", ["tenant_id", "order_id"])
    op.create_index("ix_order_items_tenant_product", "order_items", ["tenant_id", "product_id"])

    # ---------------------------------------------------------------- Row-Level Security
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING ({TENANT_PREDICATE}) WITH CHECK ({TENANT_PREDICATE})"
        )

    # ---------------------------------------------------------------- Roller ve yetkiler
    # rag_readonly: Text-to-SQL sorguları YALNIZCA bu rolle çalışır → yalnızca SELECT,
    # yalnızca bu dört tabloda. Belge/tenant tablolarına hiç erişimi yoktur.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{READONLY_ROLE}') THEN
                CREATE ROLE {READONLY_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS;
            END IF;
        END
        $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {READONLY_ROLE}")
    op.execute(f"GRANT SELECT ON {', '.join(TABLES)} TO {READONLY_ROLE}")
    # Uygulama rolü, analitik transaction'larında SET LOCAL ROLE ile salt-okunur role geçebilmeli
    op.execute(f"GRANT {READONLY_ROLE} TO {APP_ROLE}")
    # Uygulama rolü veri yüklemesi (ETL/demo) için yazabilir
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(TABLES)} TO {APP_ROLE}")

    # Tenant bağlamı (app.tenant_id) set_config ile ayarlanır. Herhangi bir rol bu
    # fonksiyonu çağırabilseydi, doğrulayıcıyı atlatan bir sorgu tek SELECT içinde
    # tenant bağlamını değiştirip RLS'i yanıltabilirdi. Yetki yalnızca uygulama rolüne
    # verilir; rag_readonly (Text-to-SQL) set_config çağıramaz.
    op.execute("REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) TO {APP_ROLE}")


def downgrade() -> None:
    op.execute("GRANT EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) TO PUBLIC")
    op.execute(f"REVOKE {READONLY_ROLE} FROM {APP_ROLE}")
    for table in reversed(TABLES):
        op.execute(f"REVOKE ALL ON {table} FROM {READONLY_ROLE}, {APP_ROLE}")
        op.drop_table(table)
