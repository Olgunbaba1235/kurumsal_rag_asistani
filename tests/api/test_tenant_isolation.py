"""Çok kiracılı veri izolasyonu: bir şirketin verisi diğerine hiçbir yoldan sızmamalı.

Üç katman ayrı ayrı doğrulanır:
1. API (kimlik → tenant, 404 ile varlık gizleme)
2. Servis/arama (vektör + BM25 her zaman tenant + departman kapsamlı)
3. Veritabanı RLS — uygulama filtresi OLMADAN çalıştırılan ham SQL ile
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from tests.api.conftest import SECRET_B, run


def h(t: dict) -> dict:
    return {"X-API-Key": t["api_key"]}


# ------------------------------------------------------------------ API katmanı
def test_document_lists_are_tenant_scoped(client, tenant_a, tenant_b):
    a = {d["filename"] for d in client.get("/api/v1/documents", headers=h(tenant_a)).json()["items"]}
    b = {d["filename"] for d in client.get("/api/v1/documents", headers=h(tenant_b)).json()["items"]}
    assert a == {"IK_Izin_ve_Calisma_Yonetmeligi.md", "Finans_Harcama_Proseduru.md"}
    assert b == {"Rakip_IK.md"}


def test_cannot_read_or_delete_other_tenants_document(client, tenant_a, tenant_b):
    a_doc_id = tenant_a["docs"]["IK_Izin_ve_Calisma_Yonetmeligi.md"]["id"]
    # Varlığı bile sızdırılmaz: 403 değil 404
    assert client.get(f"/api/v1/documents/{a_doc_id}", headers=h(tenant_b)).status_code == 404
    assert client.delete(f"/api/v1/documents/{a_doc_id}", headers=h(tenant_b)).status_code == 404
    # A'nın belgesi yerinde
    assert client.get(f"/api/v1/documents/{a_doc_id}", headers=h(tenant_a)).json()["status"] == "ready"


def test_chat_never_returns_other_tenants_content(client, tenant_a, tenant_b):
    # B'nin gizli kodu A'nın aramasında görünmemeli
    r = client.post("/api/v1/chat/query", json={"question": f"{SECRET_B} kodu nedir?"}, headers=h(tenant_a))
    body = r.json()
    assert r.status_code == 200
    assert all(e["source"] != "Rakip_IK.md" for e in body["evidence"])
    # Gizli kod soruda geçtiği için yeniden yazılmış sorguda görünür; içerik alanlarında görünmemeli
    leaked = [body["answer"], *map(str, body["citations"]), *(e["text"] for e in body["evidence"])]
    assert not any(SECRET_B in part for part in leaked)
    assert not body["is_answered"]

    # B, A'ya özgü bir soruda yalnızca kendi belgesini görür
    r = client.post(
        "/api/v1/chat/query", json={"question": "Harcama onay yetkileri ve CFO onay limiti nedir?"}, headers=h(tenant_b)
    )
    body = r.json()
    assert {e["source"] for e in body["evidence"]} <= {"Rakip_IK.md"}
    assert "Finans_Harcama_Proseduru" not in str(body)


def test_same_question_gets_each_tenants_own_answer(client, tenant_a, tenant_b):
    q = {"question": "Yıllık izin kaç iş günü?", "departments": ["İnsan Kaynakları"]}
    a = client.post("/api/v1/chat/query", json=q, headers=h(tenant_a)).json()
    b = client.post("/api/v1/chat/query", json=q, headers=h(tenant_b)).json()
    assert a["citations"] and all(c["source"] == "IK_Izin_ve_Calisma_Yonetmeligi.md" for c in a["citations"])
    assert b["citations"] and all(c["source"] == "Rakip_IK.md" for c in b["citations"])


# ------------------------------------------------------------------ Servis katmanı
def test_retriever_scoped_by_tenant_and_department(client, container, tenant_a, tenant_b):
    from app.services.tenant_service import TenantContext

    ctx_b = TenantContext(uuid.UUID(tenant_b["id"]), "rakip-as", "Rakip", tuple(tenant_b["departments"]))

    async def search():
        async with container.db.tenant_session(ctx_b.id) as s:
            hits, _, _ = await container.retriever.retrieve(
                s, ctx_b, "harcama onay yetkileri yıllık izin", ["İnsan Kaynakları", "Finans"], top_k=20
            )
            return hits

    hits = run(client, search)
    assert hits and {c.source for c in hits} == {"Rakip_IK.md"}


def test_bm25_cache_is_partitioned_by_tenant(client, container, tenant_a, tenant_b):
    keys = list(container.retriever._cache)
    tenants_in_cache = {k[0] for k in keys}
    assert len(tenants_in_cache) >= 2  # her tenant'ın ayrı indeksi var
    for idx_key, index in container.retriever._cache.items():
        sources = {c.source for c in index.chunks}
        if idx_key[0] == uuid.UUID(tenant_b["id"]):
            assert sources <= {"Rakip_IK.md"}


# ------------------------------------------------------------------ Veritabanı (RLS) katmanı
async def _raw_count(container, tenant_id: uuid.UUID | None, table: str) -> int:
    cm = container.db.tenant_session(tenant_id) if tenant_id else container.db.system_session()
    async with cm as s:
        return int(await s.scalar(text(f"SELECT count(*) FROM {table}")))  # WHERE YOK


def test_rls_filters_raw_sql_without_application_where(client, container, tenant_a, tenant_b):
    a_id, b_id = uuid.UUID(tenant_a["id"]), uuid.UUID(tenant_b["id"])
    b_chunks = run(client, _raw_count, container, b_id, "document_chunks")
    a_chunks = run(client, _raw_count, container, a_id, "document_chunks")
    assert b_chunks == tenant_b["docs"]["Rakip_IK.md"]["chunk_count"]
    assert a_chunks >= 17  # İK (9) + Finans (8)
    assert run(client, _raw_count, container, b_id, "documents") == 1


def test_rls_fails_closed_without_tenant_context(client, container, tenant_a):
    assert run(client, _raw_count, container, None, "documents") == 0
    assert run(client, _raw_count, container, None, "document_chunks") == 0


def test_rls_blocks_cross_tenant_writes(client, container, tenant_a, tenant_b):
    a_doc_id = tenant_a["docs"]["Finans_Harcama_Proseduru.md"]["id"]

    async def attempt_update() -> int:
        # B'nin oturumunda A'nın belgesini doğrudan güncellemeyi dene
        async with container.db.tenant_session(uuid.UUID(tenant_b["id"])) as s, s.begin():
            res = await s.execute(text("UPDATE documents SET filename = 'hacked' WHERE id = :id"), {"id": a_doc_id})
            return res.rowcount

    async def attempt_insert() -> None:
        # B'nin oturumunda A adına satır eklemeyi dene → WITH CHECK ihlali
        async with container.db.tenant_session(uuid.UUID(tenant_b["id"])) as s, s.begin():
            await s.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, department, filename, file_type, size_bytes, "
                    "content_hash, status) VALUES (:id, :tid, 'Genel', 'x.md', 'md', 1, 'h', 'processing')"
                ),
                {"id": uuid.uuid4(), "tid": tenant_a["id"]},
            )

    assert run(client, attempt_update) == 0
    with pytest.raises((DBAPIError, ProgrammingError), match="row-level security"):
        run(client, attempt_insert)
    assert client.get(f"/api/v1/documents/{a_doc_id}", headers=h(tenant_a)).json()["filename"] != "hacked"


def test_app_role_is_not_superuser_and_cannot_bypass_rls(client, container, tenant_a):
    async def role_info():
        async with container.db.tenant_session(uuid.UUID(tenant_a["id"])) as s:
            return (
                await s.execute(
                    text("SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
                )
            ).one()

    user, is_super, bypass = run(client, role_info)
    assert user == "rag_app" and not is_super and not bypass


def test_tenant_setting_does_not_leak_to_next_transaction(client, container, tenant_a):
    """SET LOCAL transaction kapsamlıdır: havuza dönen bağlantıda tenant kalmamalı."""

    async def check():
        async with container.db.tenant_session(uuid.UUID(tenant_a["id"])) as s:
            await s.scalar(text("SELECT 1"))
        async with container.db.system_session() as s:
            return await s.scalar(text("SELECT NULLIF(current_setting('app.tenant_id', true), '')"))

    assert run(client, check) is None
