"""Text-to-SQL güvenlik doğrulayıcısı ve Python analiz/grafik birim testleri (veritabanı gerekmez)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from app.services.analytics_service import heuristic_route
from app.services.sql_guard import SQLGuard, UnsafeSQLError
from app.services.text_to_sql_service import ChartHint, QueryResult, _kind, analyze, build_chart

GUARD = SQLGuard(["customers", "products", "orders", "order_items"], max_rows=100)


# ------------------------------------------------------------------ izin verilenler
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c.name, SUM(o.total_amount) AS ciro FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status <> 'iptal' GROUP BY c.name ORDER BY ciro DESC LIMIT 5",
        "WITH aylik AS (SELECT date_trunc('month', order_date) AS ay, SUM(total_amount) AS t FROM orders GROUP BY 1) "
        "SELECT * FROM aylik ORDER BY ay",
        "SELECT category, COUNT(*) FROM products GROUP BY category UNION ALL SELECT 'TOPLAM', COUNT(*) FROM products",
        "SELECT p.name, RANK() OVER (ORDER BY SUM(oi.line_total) DESC) FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id GROUP BY p.name",
        "SELECT to_char(order_date, 'YYYY-MM') AS ay, COUNT(*) FROM orders WHERE order_date >= '2026-01-01' GROUP BY 1",
        "SELECT name FROM products WHERE stock_qty < reorder_level",
    ],
)
def test_safe_queries_pass(sql):
    v = GUARD.validate(sql)
    assert v.tables and "LIMIT" in v.sql.upper()


# ------------------------------------------------------------------ reddedilenler
@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("SELECT 1; DROP TABLE customers", "tek bir SELECT"),
        ("DROP TABLE customers", "Yalnızca SELECT"),
        ("TRUNCATE orders", "SELECT"),
        ("DELETE FROM orders", "Yalnızca SELECT"),
        ("UPDATE products SET unit_price = 0", "Yalnızca SELECT"),
        ("INSERT INTO customers (name) VALUES ('x')", "Yalnızca SELECT"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "DELETE"),
        ("SELECT * INTO yedek FROM customers", "INTO"),
        ("SELECT * FROM customers FOR UPDATE", "LOCK"),
        ("COPY customers TO '/tmp/x'", "SELECT"),
        ("SELECT pg_sleep(10) FROM customers", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd') FROM customers", "pg_read_file"),
        ("SELECT set_config('app.tenant_id', 'x', true) FROM customers", "set_config"),
        ("SELECT current_setting('app.tenant_id') FROM customers", "current_setting"),
        ("SELECT dblink('host=x', 'select 1') FROM customers", "dblink"),
        ("SELECT * FROM pg_catalog.pg_roles", "Şema"),
        ("SELECT * FROM information_schema.tables", "Şema"),
        ("SELECT * FROM pg_roles", "pg_roles"),
        ("SELECT * FROM documents", "documents"),
        ("SELECT * FROM document_chunks", "document_chunks"),
        ("SELECT * FROM tenants", "tenants"),
        ("SELECT tenant_id FROM customers", "tenant_id"),
        ("SELECT * FROM orders WHERE tenant_id <> '00000000-0000-0000-0000-000000000000'", "tenant_id"),
        ("SELECT * FROM customers LIMIT (SELECT 10)", "LIMIT"),
        ("SELECT some_custom_udf(id) FROM customers", "İzin verilmeyen fonksiyon"),
        ("SELECT", "tablolardan"),
        ("SELEKT * FROM customers", "ayrıştırılamadı"),
        ("", "Boş"),
    ],
)
def test_unsafe_queries_rejected(sql, reason):
    with pytest.raises(UnsafeSQLError, match=reason):
        GUARD.validate(sql)


def test_limit_is_enforced_and_capped():
    assert GUARD.validate("SELECT name FROM customers").sql.endswith("LIMIT 100")
    assert GUARD.validate("SELECT name FROM customers LIMIT 5000").sql.endswith("LIMIT 100")
    assert GUARD.validate("SELECT name FROM customers LIMIT 7").sql.endswith("LIMIT 7")


def test_comments_are_stripped_from_executed_sql():
    v = GUARD.validate("SELECT name FROM customers -- ; DROP TABLE customers\n/* gizli */")
    assert "DROP" not in v.sql and "--" not in v.sql and "/*" not in v.sql


def test_cte_names_are_allowed_but_not_other_tables():
    GUARD.validate("WITH t AS (SELECT id FROM orders) SELECT COUNT(*) FROM t")
    with pytest.raises(UnsafeSQLError):
        GUARD.validate("WITH t AS (SELECT id FROM tenants) SELECT COUNT(*) FROM t")


# ------------------------------------------------------------------ analiz ve grafik
def _result(columns, raw_rows, truncated=False):
    kinds = {c: _kind([r[i] for r in raw_rows]) for i, c in enumerate(columns)}
    rows = [[float(v) if isinstance(v, Decimal) else (v.isoformat() if isinstance(v, dt.date) else v) for v in r]
            for r in raw_rows]
    return QueryResult(list(columns), rows, kinds, len(rows), truncated)


def test_analyze_categorical_insights():
    r = _result(["musteri", "ciro"], [["A", Decimal("500")], ["B", Decimal("300")], ["C", Decimal("200")]])
    stats, insights = analyze(r)
    assert stats["ciro"] == {"count": 3, "sum": 1000.0, "mean": 333.33, "min": 200.0, "max": 500.0}
    assert any("En yüksek ciro: A (500 — toplamın %50.0'i)" in i for i in insights)
    assert any("Toplam ciro: 1.000" in i for i in insights)


def test_analyze_temporal_trend_and_identifier_exclusion():
    r = _result(
        ["ay", "customer_id", "ciro"],
        [[dt.date(2026, 1, 1), 1, Decimal("100")], [dt.date(2026, 2, 1), 2, Decimal("150")], [dt.date(2026, 3, 1), 3, Decimal("120")]],
    )
    stats, insights = analyze(r)
    assert "customer_id" not in stats
    assert any("%+20.0" in i for i in insights) and any("En yüksek ciro: 2026-02-01" in i for i in insights)


def test_build_chart_auto_and_hints():
    temporal = _result(["ay", "ciro"], [[dt.date(2026, m, 1), Decimal(m * 10)] for m in range(1, 7)])
    spec = build_chart(temporal, None, "Aylık ciro")
    assert spec["mark"]["type"] == "line" and spec["encoding"]["x"]["type"] == "temporal"

    cat = _result(["kategori", "adet"], [[f"K{i}", i] for i in range(12)])
    assert build_chart(cat, ChartHint(type="pie", x="kategori", y="adet"), "t")["mark"]["type"] == "bar"  # >8 dilim
    small = _result(["kanal", "pay"], [["a", 1], ["b", 2], ["c", 3]])
    assert build_chart(small, ChartHint(type="pie", x="kanal", y="pay"), "t")["mark"]["type"] == "arc"
    assert build_chart(_result(["toplam"], [[5]]), None, "t") is None  # tek sütun: grafik yok


# ------------------------------------------------------------------ yönlendirme
@pytest.mark.parametrize(
    ("q", "route"),
    [
        ("2026'da en çok ciro yapan 5 müşteri kim?", "data"),
        ("Kritik stok seviyesinin altındaki ürünleri listele", "data"),
        ("Yıllık izin politikası nedir?", "documents"),
        ("Harcama onay prosedürüne göre 250.000 TL üzerindeki siparişler hangileri?", "both"),
    ],
)
def test_heuristic_route(q, route):
    assert heuristic_route(q)[0] == route
