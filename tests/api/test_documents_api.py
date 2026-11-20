from __future__ import annotations

import io
import time
import uuid

from openpyxl import Workbook
from sqlalchemy import text

from tests.api.conftest import create_tenant, run, upload


def h(t: dict) -> dict:
    return {"X-API-Key": t["api_key"]}


def test_auth_required(client):
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "rag_yanlis"}).status_code == 401
    assert client.post("/api/v1/admin/tenants", json={"name": "X Co", "slug": "x-co"}).status_code == 401


def test_tenant_slug_conflict(client):
    create_tenant(client, "cakisma-co", "Çakışma Co")
    r = client.post(
        "/api/v1/admin/tenants",
        json={"name": "Tekrar", "slug": "cakisma-co"},
        headers={"X-Admin-Key": "test-admin-key-0123456789"},
    )
    assert r.status_code == 409


def test_sample_docs_indexed_with_pages(client, tenant_a):
    ik = tenant_a["docs"]["IK_Izin_ve_Calisma_Yonetmeligi.md"]
    assert ik["status"] == "ready" and ik["page_count"] == 4 and ik["chunk_count"] == 9
    assert ik["department"] == "İnsan Kaynakları"


def test_duplicate_upload_is_detected(client, tenant_a):
    from tests.api.conftest import SAMPLE_DIR

    name = "Finans_Harcama_Proseduru.md"
    r = upload(client, tenant_a["api_key"], name, (SAMPLE_DIR / name).read_bytes(), "Finans")
    assert r.status_code == 200 and r.json()["outcome"] == "duplicate"


def test_upload_validation_errors(client):
    t = create_tenant(client, "dogrulama-co", "Doğrulama Co")
    k = t["api_key"]
    assert upload(client, k, "sunum.pptx", b"x", "Genel").status_code == 415
    assert upload(client, k, "bos.txt", b"", "Genel").status_code == 422
    r = upload(client, k, "a.txt", b"yeterince uzun bir metin icerigi", "Pazarlama")
    assert r.status_code == 422 and "Geçersiz departman" in r.json()["detail"]


def test_background_upload_then_status_ready(client):
    t = create_tenant(client, "arkaplan-co", "Arka Plan Co")
    r = upload(client, t["api_key"], "notlar.txt", "Arka planda işlenen yeterince uzun bir not.".encode(), "Genel", wait=False)
    assert r.status_code == 202 and r.json()["outcome"] == "accepted"
    doc_id = r.json()["document"]["id"]
    for _ in range(20):  # TestClient arka plan görevini yanıt sonrası çalıştırır
        status = client.get(f"/api/v1/documents/{doc_id}", headers=h(t)).json()["status"]
        if status != "processing":
            break
        time.sleep(0.1)
    assert status == "ready"


def test_failed_processing_is_recorded(client):
    t = create_tenant(client, "bozuk-co", "Bozuk Co")
    r = upload(client, t["api_key"], "bozuk.pdf", b"%PDF-1.4 bozuk icerik", "Genel")
    assert r.status_code == 422
    assert r.json()["outcome"] == "failed" and r.json()["document"]["status"] == "failed"
    assert r.json()["document"]["error"]


def test_xlsx_upload_keeps_sheet_pages(client, container):
    t = create_tenant(client, "excel-co", "Excel Co")
    wb = Workbook()
    ws = wb.active
    ws.title = "Limitler"
    ws.append(["Kalem", "Limit"])
    ws.append(["Otel", 4500])
    ws2 = wb.create_sheet("Harcırah")
    ws2.append(["Kalem", "Tutar"])
    ws2.append(["Yemek", 1200])
    buf = io.BytesIO()
    wb.save(buf)
    r = upload(client, t["api_key"], "limitler.xlsx", buf.getvalue(), "Finans")
    assert r.status_code == 201, r.text

    async def pages():
        async with container.db.tenant_session(uuid.UUID(t["id"])) as s:
            rows = await s.execute(text("SELECT page, sheet_name, content FROM document_chunks ORDER BY page"))
            return [tuple(r) for r in rows]

    rows = run(client, pages)
    assert [(p, sh) for p, sh, _ in rows] == [(1, "Limitler"), (2, "Harcırah")]
    assert "| Otel | 4500 |" in rows[0][2]


def test_new_version_replaces_old_atomically(client, container):
    t = create_tenant(client, "surum-co", "Sürüm Co")
    k = t["api_key"]
    upload(client, k, "politika.md", "İlk sürüm: uzaktan çalışma haftada iki gündür.".encode(), "Genel")
    r = upload(client, k, "politika.md", "İkinci sürüm: uzaktan çalışma haftada üç gündür.".encode(), "Genel")
    assert r.status_code == 201

    docs = client.get("/api/v1/documents", headers=h(t)).json()["items"]
    assert len(docs) == 1 and docs[0]["id"] == r.json()["document"]["id"]

    async def contents():
        async with container.db.tenant_session(uuid.UUID(t["id"])) as s:
            return list(await s.scalars(text("SELECT content FROM document_chunks")))

    texts = run(client, contents)
    assert len(texts) == 1 and "İkinci sürüm" in texts[0]


def test_delete_removes_chunks(client, container):
    t = create_tenant(client, "silme-co", "Silme Co")
    doc = upload(client, t["api_key"], "gecici.md", "Silinecek yeterince uzun bir belge içeriği.".encode(), "Genel").json()
    assert client.delete(f"/api/v1/documents/{doc['document']['id']}", headers=h(t)).status_code == 204

    async def count():
        async with container.db.tenant_session(uuid.UUID(t["id"])) as s:
            return await s.scalar(text("SELECT count(*) FROM document_chunks"))

    assert run(client, count) == 0
    assert client.get(f"/api/v1/documents/{doc['document']['id']}", headers=h(t)).status_code == 404


def test_custom_tenant_departments(client):
    t = create_tenant(client, "hukuk-co", "Hukuk Co", departments=["Genel", "Hukuk"])
    assert upload(client, t["api_key"], "sozlesme.md", "Sözleşme fesih süresi otuz gündür.".encode(), "Hukuk").status_code == 201
    assert upload(client, t["api_key"], "x.md", "Finans belgesi yeterince uzun.".encode(), "Finans").status_code == 422


def test_list_filter_and_pagination(client, tenant_a):
    r = client.get("/api/v1/documents", params={"department": "Finans"}, headers=h(tenant_a)).json()
    assert [d["filename"] for d in r["items"]] == ["Finans_Harcama_Proseduru.md"] and r["total"] == 1
    r = client.get("/api/v1/documents", params={"limit": 1}, headers=h(tenant_a)).json()
    assert len(r["items"]) == 1 and r["total"] == 2
