# Kurumsal RAG Asistanı — Proje Notları

Çok kiracılı (multi-tenant) B2B SaaS RAG platformu: FastAPI + PostgreSQL/pgvector + Redis + Google Gemini.
Kullanıcıyla Türkçe çalışılır.

## Mimari (kısa)

- `app/` — FastAPI REST API (`api/v1`, `core`, `models`, `schemas`, `services`). Uygulama fabrikası:
  `uvicorn app.main:create_app --factory` (modül düzeyinde `app` nesnesi YOK, bilinçli).
- `src/` — arka uçtan bağımsız RAG çekirdeği (belge işleme, hibrit arama, katı bağlam, atıf/güven). API ve
  Streamlit ortak kullanır. `src/` ChromaDB'yi yalnızca `src/vector_store.py` içinde import eder.
- `migrations/` — Alembic: `0001` belgeler + RLS, `0002` satış analitiği tabloları + `rag_readonly` rolü.
- Çok kiracılık: tenant yalnızca `X-API-Key`'den çözülür; `app/core/db.py` her transaction başında
  `app.tenant_id` + rolü `after_begin` olayıyla ayarlar; tüm tenant tablolarında `FORCE ROW LEVEL SECURITY`.
- Text-to-SQL: `app/services/sql_guard.py` (sqlglot, deny-by-default) → `rag_readonly` rolü + READ ONLY tx.
  `set_config` yetkisi PUBLIC'ten alınıp yalnızca `rag_app`'e verilir (migration 0002).

## Komutlar

```powershell
.\.venv\Scripts\activate
npm install --prefix tests/pglite        # bir kez: testler için gömülü PostgreSQL (PGlite) + pgvector
pytest                                    # 167 test; API testleri PGlite üzerinde gerçek Postgres ile
python scripts/quick_check.py --offline   # Chroma tabanlı çekirdeğin hızlı terminal doğrulaması
```

- PGlite tek backend'dir: testlerde `DB_POOL_SIZE=1` ve tek oturum açık `TestClient` kullanılır; bağlantı
  kapatma takılır → `DB_DISPOSE_TIMEOUT` ile sınırlıdır (teardown'daki "Exception closing connection" logu beklenir).
- Servisler kendi kısa ömürlü oturumlarını açar; endpoint'ler DB oturumu tutmaz (LLM üretimi sırasında bağlantı tutulmaz).

## Durum (2026-11-20 itibarıyla)

Tamamlanan fazlar:
1. Faz 1 — belge işleme (PDF/DOCX/XLSX/TXT/MD, tablo korumalı chunking), ChromaDB deposu.
2. Faz 2 — hibrit arama (vektör + BM25Plus + RRF + reranking), sorgu yeniden yazma, katı bağlam, atıf/güven.
3. Çok kiracılı FastAPI + pgvector backend, Docker Compose, Alembic, RLS, Redis rate limit.
4. Faz 3 — Text-to-SQL analitik ajanı + `/api/v1/analytics/query` (doküman + veri birleşik yanıt).

Doğrulanmamış / bilinen açıklar:
- **Docker Compose hiç çalıştırılmadı** (bu makinede Docker/WSL yok). `docker/postgres/initdb/01-app-role.sh`
  hiç test edilmedi. Testler PGlite'ın PostgreSQL 18'i üzerinde; compose `pgvector/pgvector:pg16` kullanır.
- **Gemini hiç çağrılmadı** (API anahtarı yok). Model adları (`gemini-2.5-flash`, `-flash-lite`,
  `gemini-embedding-001`) doğrulanmadı; `MIN_RELEVANCE=0.30` eşiği yalnızca çevrimdışı reranker ile kalibre edildi.
- Streamlit `app.py` hâlâ eski `src/rag_assistant/` paketini kullanıyor (yeni API'ye bağlı değil).
- Arka plan ingestion süreç içi (BackgroundTasks); BM25 önbelleği worker başına.

## Sıradaki adım

1. **Docker ile gerçek uçtan uca test:** kullanıcı `wsl --install` + Docker Desktop kurduktan sonra
   `docker compose up -d --build` → tenant oluştur (`/api/v1/admin/tenants`) → `/health` → `/documents/upload`
   (data/sample_docs) → `/chat/query` → `/analytics/demo-data` + `/analytics/query`. Hata çıkarsa düzelt.
2. Gerçek `GOOGLE_API_KEY` ile `python scripts/quick_check.py` ve analitik sorgularla eşik/kalite kalibrasyonu.
3. Sonraki faz adayları: Streamlit/ön yüzü API'ye bağlama · SSE akışlı sohbet · kuyruk tabanlı ingestion (Arq) ·
   kullanıcı düzeyinde departman yetkileri (SSO/OIDC).
