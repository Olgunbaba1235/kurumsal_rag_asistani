"""B2B toptan satış ve üretim firması için analitik tablolar.

Tüm tablolar çok kiracılıdır: ``tenant_id`` zorunlu, PostgreSQL RLS ile
korunur ve tablolar arası ilişkiler ``(…_id, tenant_id)`` bileşik yabancı
anahtarlarıyla kurulur — bir siparişin müşterisi ya da bir sipariş kaleminin
ürünü başka bir tenant'a ait olamaz.

Tutarlar TRY ve KDV hariçtir.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

ORDER_STATUSES = ("teslim_edildi", "sevk_edildi", "hazirlaniyor", "iptal")
SALES_CHANNELS = ("saha_satis", "b2b_portal", "telefon", "ihale")
CUSTOMER_SEGMENTS = ("Distribütör", "Bayi", "Kurumsal", "Perakende Zinciri")


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_customers_id_tenant"),
        UniqueConstraint("tenant_id", "code", name="uq_customers_tenant_code"),
        Index("ix_customers_tenant_segment", "tenant_id", "segment"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    code: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    segment: Mapped[str] = mapped_column(String(40), nullable=False)
    city: Mapped[str] = mapped_column(String(60), nullable=False)
    region: Mapped[str] = mapped_column(String(60), nullable=False)
    credit_limit: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    payment_terms_days: Mapped[int] = mapped_column(Integer, nullable=False)
    since: Mapped[dt.date] = mapped_column(Date, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_products_id_tenant"),
        UniqueConstraint("tenant_id", "sku", name="uq_products_tenant_sku"),
        Index("ix_products_tenant_category", "tenant_id", "category"),
        CheckConstraint("unit_price >= 0 AND unit_cost >= 0", name="non_negative_prices"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    sku: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    unit: Mapped[str] = mapped_column(String(10), nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    unit_cost: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    stock_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    reorder_level: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_orders_id_tenant"),
        UniqueConstraint("tenant_id", "order_no", name="uq_orders_tenant_order_no"),
        ForeignKeyConstraint(
            ["customer_id", "tenant_id"], ["customers.id", "customers.tenant_id"], name="fk_orders_customer_tenant"
        ),
        Index("ix_orders_tenant_date", "tenant_id", "order_date"),
        Index("ix_orders_tenant_customer", "tenant_id", "customer_id"),
        CheckConstraint(
            "status IN ('teslim_edildi', 'sevk_edildi', 'hazirlaniyor', 'iptal')", name="valid_status"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    order_no: Mapped[str] = mapped_column(String(30), nullable=False)
    customer_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    order_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    shipped_date: Mapped[dt.date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    discount_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=Decimal("0"))
    total_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        ForeignKeyConstraint(
            ["order_id", "tenant_id"], ["orders.id", "orders.tenant_id"], ondelete="CASCADE",
            name="fk_order_items_order_tenant",
        ),
        ForeignKeyConstraint(
            ["product_id", "tenant_id"], ["products.id", "products.tenant_id"], name="fk_order_items_product_tenant"
        ),
        Index("ix_order_items_tenant_order", "tenant_id", "order_id"),
        Index("ix_order_items_tenant_product", "tenant_id", "product_id"),
        CheckConstraint("quantity > 0", name="positive_quantity"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    order_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    product_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    line_total: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)


ANALYTICS_TABLES = ("customers", "products", "orders", "order_items")
