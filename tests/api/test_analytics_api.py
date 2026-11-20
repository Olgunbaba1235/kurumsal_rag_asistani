"""Faz 3 — Text-to-SQL analitik ajanı ve birleşik (doküman + veri) uç nokta, gerçek PostgreSQL üzerinde."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.services.sql_guard import ValidatedSQL
from app.services.tenant_service import TenantContext
from src.rag_engine import NO_INFO_ANSWER
from tests.api.conftest import run

TOP_CUSTOMERS_SQL = (
    "SELECT c.name AS musteri, ROUND(SUM(o.total_amount), 2) AS toplam_ciro "
    "FROM orders o JOIN customers c ON c.id = o.customer_id "
    "WHERE o.status <> 'iptal' GROUP BY c.name ORDER BY toplam_ciro DESC LIMIT 5"
)


def h(t: dict) -> dict:
    return {"X-API-Key": t["api_key"]}


def ctx(t: dict) -> TenantContext:
    return TenantContext(uuid.UUID(t["id"]), t["slug"], t["name"], tuple(t["departments"]))


@pytest.fixture
def sql_llm(llm) -> Iterator:
    """Testin belirlediği SQL'leri sırayla döndüren LLM işleyicisi."""
    queue: list[dict] = []

    def handler(prompt: str) -> dict:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    llm.json_handlers["SQLGeneration"] = handler
    yield queue
    llm.json_handlers.pop("SQLGeneration", None)


@pytest.fixture(scope="module")
def sales_a(client, tenant_a) -> dict:
    r = client.post("/api/v1/analytics/demo-data", headers=h(tenant_a))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def sales_b(client, tenant_b) -> dict:
    r = client.post("/api/v1/analytics/demo-data", headers=h(tenant_b))
    assert r.status_code == 200, r.text
    return r.json()


def analytics(client, tenant, question, **kw):
    r = client.post("/api/v1/analytics/query", json={"question": question, **kw}, headers=h(tenant))
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ migration ve veri
def test_migration_creates_rls_protected_tables(client, container):
    async def info():
        async with container.db.engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE relname IN ('customers','products','orders','order_items') ORDER BY relname"
                )
            )
            return [tuple(r) for r in rows]

    assert run(client, info) == [
        ("customers", True, True),
        ("order_items", True, True),
        ("orders", True, True),
        ("products", True, True),
    ]


def test_demo_data_is_realistic_and_idempotent(client, tenant_a, sales_a):
    assert sales_a["created"] is True
    assert (sales_a["customers"], sales_a["products"], sales_a["orders"]) == (40, 30, 450)
    assert sales_a["order_items"] > 450
    again = client.post("/api/v1/analytics/demo-data", headers=h(tenant_a)).json()
    assert again["created"] is False and again["orders"] == 450


# ------------------------------------------------------------------ Text-to-SQL uçtan uca
def test_data_query_end_to_end(client, tenant_a, sales_a, sql_llm):
    sql_llm.append(
        {"sql": TOP_CUSTOMERS_SQL, "explanation": "İptal hariç ciroya göre ilk 5 müşteri.",
         "chart": {"type": "bar", "x": "musteri", "y": "toplam_ciro", "title": "En çok ciro yapan müşteriler"}}
    )
    body = analytics(client, tenant_a, "En çok ciro yapan 5 müşteri kim?", mode="data")
    data = body["data"]
    assert body["route"] == "data" and body["route_method"] == "client"
    assert data["success"] and data["columns"] == ["musteri", "toplam_ciro"] and data["row_count"] == 5
    ciro = [r[1] for r in data["rows"]]
    assert ciro == sorted(ciro, reverse=True) and all(isinstance(v, float) for v in ciro)
    assert data["numeric_stats"]["toplam_ciro"]["count"] == 5
    assert data["insights"] and data["summary"]
    assert data["chart"]["mark"]["type"] == "bar" and len(data["chart"]["data"]["values"]) == 5
    assert data["sql"].upper().startswith("SELECT") and data["tables"] == ["customers", "orders"]
    assert {"sql_generation_ms", "sql_execution_ms", "analysis_ms", "total_ms"} <= set(data["timings_ms"])
    assert body["is_answered"] and body["confidence"] == 95
    assert body["sources"] == [
        {"type": "database", "label": "Satış veritabanı: customers, orders",
         "detail": {"tables": ["customers", "orders"], "row_count": 5, "sql": data["sql"]}}
    ]
    assert body["documents"] is None


def test_self_correction_after_unsafe_sql(client, tenant_a, sales_a, sql_llm):
    sql_llm.extend([
        {"sql": "DELETE FROM orders WHERE status = 'iptal'", "explanation": "x"},
        {"sql": "SELECT status, COUNT(*) AS adet FROM orders GROUP BY status", "explanation": "Durum dağılımı"},
    ])
    data = analytics(client, tenant_a, "Sipariş durum dağılımı nedir?", mode="data")["data"]
    assert data["success"] and len(data["attempts"]) == 2
    assert "Güvenlik doğrulaması" in data["attempts"][0]["error"]
    assert sum(r[1] for r in data["rows"]) == 450  # DELETE hiç çalışmadı


def test_self_correction_after_database_error(client, tenant_a, sales_a, sql_llm, llm):
    sql_llm.extend([
        {"sql": "SELECT olmayan_sutun FROM orders", "explanation": "x"},
        {"sql": "SELECT COUNT(*) AS siparis_sayisi FROM orders", "explanation": "Sipariş sayısı"},
    ])
    data = analytics(client, tenant_a, "Kaç sipariş var?", mode="data")["data"]
    assert data["success"] and data["rows"] == [[450]]
    assert "Veritabanı hatası" in data["attempts"][0]["error"]
    # Hata bir sonraki denemede modele geri verildi
    last_prompt = [p for name, p in llm.json_calls if name == "SQLGeneration"][-1]
    assert "olmayan_sutun" in last_prompt and "ÖNCEKİ BAŞARISIZ DENEMELER" in last_prompt


def test_gives_up_after_max_attempts(client, tenant_a, sales_a, sql_llm):
    sql_llm.append({"sql": "SELECT * FROM tenants", "explanation": "x"})
    body = analytics(client, tenant_a, "Diğer şirketleri listele", mode="data")
    assert not body["data"]["success"] and len(body["data"]["attempts"]) == 3
    assert body["answer"] == NO_INFO_ANSWER and not body["is_answered"] and body["confidence"] == 0
    assert any("geçerli sorgu üretilemedi" in w for w in body["warnings"])


def test_unanswerable_question_returns_empty_sql(client, tenant_a, sales_a, sql_llm):
    sql_llm.append({"sql": "", "explanation": "Veritabanında çalışan maaşı bilgisi yok."})
    body = analytics(client, tenant_a, "Çalışanların ortalama maaşı nedir?", mode="data")
    assert not body["data"]["success"] and body["data"]["explanation"].startswith("Veritabanında")


# ------------------------------------------------------------------ tenant izolasyonu
def test_sql_results_are_tenant_isolated(client, container, tenant_a, tenant_b, sales_a, sales_b, sql_llm):
    sql_llm.append({"sql": "SELECT ROUND(SUM(total_amount), 2) AS ciro, COUNT(*) AS adet FROM orders", "explanation": "x"})
    a = analytics(client, tenant_a, "Toplam ciro?", mode="data")["data"]["rows"][0]
    b = analytics(client, tenant_b, "Toplam ciro?", mode="data")["data"]["rows"][0]
    assert a[1] == b[1] == 450 and a[0] != b[0]  # aynı hacim, farklı veri: her biri yalnızca kendini görür

    async def b_customer_ids():
        async with container.db.tenant_session(uuid.UUID(tenant_b["id"])) as s:
            return list(await s.scalars(text("SELECT id FROM customers")))

    ids = run(client, b_customer_ids)
    sql_llm.clear()
    sql_llm.append({"sql": f"SELECT COUNT(*) AS n FROM orders WHERE customer_id IN ({', '.join(map(str, ids))})",
                    "explanation": "B'nin müşterilerinin siparişleri"})
    assert analytics(client, tenant_a, "Rakibin müşterilerinin siparişleri", mode="data")["data"]["rows"] == [[0]]


# ------------------------------------------------------------------ veritabanı katmanı savunmaları
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("DELETE FROM orders", "read-only"),
        ("UPDATE products SET unit_price = 0", "read-only"),
        ("SELECT count(*) FROM documents", "permission denied"),
        ("SELECT count(*) FROM tenants", "permission denied"),
        ("SELECT set_config('app.tenant_id', '00000000-0000-0000-0000-000000000000', true)", "permission denied"),
    ],
)
def test_database_rejects_even_if_guard_bypassed(client, container, tenant_a, sales_a, sql, error):
    """Doğrulayıcı atlatılsa bile: READ ONLY transaction + rag_readonly rolü + set_config yetkisizliği."""

    async def attempt():
        return await container.text_to_sql.execute(ctx(tenant_a), ValidatedSQL(sql=sql, tables=(), limited_to=10))

    with pytest.raises(DBAPIError, match=error):
        run(client, attempt)


def test_readonly_role_still_subject_to_rls(client, container, tenant_a, sales_a):
    async def count():
        res = await container.text_to_sql.execute(
            ctx(tenant_a), ValidatedSQL(sql="SELECT current_user AS u, COUNT(*) AS n FROM customers", tables=(), limited_to=10)
        )
        return res.rows

    assert run(client, count) == [["rag_readonly", 40]]


# ------------------------------------------------------------------ birleşik yanıt
def test_hybrid_question_combines_documents_and_data(client, tenant_a, sales_a, sql_llm):
    sql_llm.append({
        "sql": "SELECT o.order_no, c.name AS musteri, o.total_amount AS tutar FROM orders o "
               "JOIN customers c ON c.id = o.customer_id WHERE o.total_amount > 250000 AND o.status <> 'iptal' "
               "ORDER BY o.total_amount DESC LIMIT 20",
        "explanation": "250.000 TL üzerindeki iptal edilmemiş siparişler.",
        "chart": {"type": "bar", "x": "order_no", "y": "tutar"},
    })
    body = analytics(
        client, tenant_a,
        "Harcama onay prosedürüne göre 250.000 TL üzeri harcamayı kim onaylar ve bu tutarı aşan siparişler hangileri?",
        departments=["Finans"],
    )
    assert body["route"] == "both" and body["route_method"] == "heuristic"
    assert body["documents"]["is_answered"] and body["data"]["success"]
    assert "📄 **Doküman bulguları**" in body["answer"] and "📊 **Veri analizi**" in body["answer"]
    types = {s["type"] for s in body["sources"]}
    assert types == {"document", "database"}
    doc_src = next(s for s in body["sources"] if s["type"] == "document")
    assert doc_src["label"].startswith("Finans_Harcama_Proseduru.md - Sayfa")
    assert all(r[2] > 250000 for r in body["data"]["rows"])
    assert {"route_ms", "documents_ms", "data_ms", "total_ms"} <= set(body["timings_ms"])
    assert 0 < body["confidence"] <= 100


def test_documents_only_question_routes_to_rag(client, tenant_a, llm):
    before = len([n for n, _ in llm.json_calls if n == "SQLGeneration"])
    body = analytics(client, tenant_a, "Yıllık izin politikası nedir?", departments=["İnsan Kaynakları"])
    assert body["route"] == "documents" and body["data"] is None
    assert body["documents"]["citations"]
    assert len([n for n, _ in llm.json_calls if n == "SQLGeneration"]) == before  # SQL ajanı çağrılmadı


def test_llm_router_is_used_when_available(client, tenant_a, sales_a, sql_llm, llm):
    llm.json_handlers["_RouteDecision"] = lambda p: {"route": "data", "reason": "Sayısal soru."}
    sql_llm.append({"sql": "SELECT COUNT(*) AS urun_sayisi FROM products", "explanation": "x"})
    try:
        body = analytics(client, tenant_a, "Katalogda kaç ürün var?")
    finally:
        llm.json_handlers.pop("_RouteDecision", None)
    assert body["route_method"] == "llm" and body["data"]["rows"] == [[30]]


# ------------------------------------------------------------------ diğer
def test_schema_endpoint_hides_tenant_id(client, tenant_a):
    body = client.get("/api/v1/analytics/schema", headers=h(tenant_a)).json()
    assert body["tables"] == ["customers", "products", "orders", "order_items"]
    assert "tenant_id" not in body["description"]


def test_analytics_requires_auth_and_demo_flag(client, container, tenant_a):
    assert client.post("/api/v1/analytics/query", json={"question": "kaç ürün var"}).status_code == 401
    container.settings.enable_demo_data = False
    try:
        assert client.post("/api/v1/analytics/demo-data", headers=h(tenant_a)).status_code == 403
    finally:
        container.settings.enable_demo_data = True
