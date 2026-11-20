from tests.api.conftest import create_tenant, upload

SAMPLE = "Yıllık izin süresi 14 iş günüdür. Fazla mesai sınırı yıllık 270 saattir.".encode()


def test_smoke_end_to_end(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200, r.text
    assert r.json()["components"]["database"]["status"] == "ok"

    t = create_tenant(client, "smoke-co", "Smoke Co")
    key = t["api_key"]
    r = upload(client, key, "smoke.md", SAMPLE, "Genel")
    assert r.status_code == 201, r.text
    assert r.json()["outcome"] == "indexed"

    r = client.post(
        "/api/v1/chat/query",
        json={"question": "Fazla mesai sınırı kaç saat?", "departments": ["Genel"]},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_answered"], body
    assert body["citations"][0]["label"] == "smoke.md - Sayfa 1"
