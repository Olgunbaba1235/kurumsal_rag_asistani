# 📚 Kurumsal RAG Asistanı

Şirket dokümanları (PDF, DOCX, XLSX, TXT/MD) üzerinde **kaynak referanslı**, halüsinasyona karşı korumalı soru-cevap sunan, **çok kiracılı (multi-tenant) B2B SaaS** RAG platformu.

**Teknolojiler:** FastAPI · PostgreSQL 16 + pgvector (HNSW) · SQLAlchemy 2 (async) · Alembic · Redis · Google Gemini (`google-genai`) · rank-bm25 · Pydantic v2 · Docker Compose · Pytest

---

## ✨ Özellikler

| Alan | Uygulama |
|---|---|
| **Belge işleme** | PDF (pdfplumber; tablolar Markdown'a), DOCX (başlık + tablo + sayfa sonları), XLSX (her sekme bir sayfa), TXT/MD. Tabloyu satır ortasından bölmeyen, başlık satırını tekrarlayan akıllı chunking |
| **Hibrit arama** | pgvector kosinüs (HNSW) + BM25Plus (Türkçe normalizasyon, F5 kök) → ağırlıklı **RRF** füzyonu → **reranking** (Gemini, hata halinde IDF ağırlıklı sözcüksel yedek) |
| **Sorgu genişletme** | Sohbet geçmişine duyarlı yeniden yazma (takip soruları → bağımsız sorgu) + 2 arama varyantı |
| **Katı bağlam** | Alaka kapısı (eşik altında LLM çağrılmaz) · katı sistem talimatı · uydurma atıfların ayıklanması · bilgi yoksa: *"Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır."* |
| **Yanıt sözleşmesi** | Satır içi `[Dosya Adı - Sayfa X]` atıfları, kanıt metinleri, güven skoru (%), adım adım süreler (ms) |
| **Çok kiracılık** | API anahtarından türetilen tenant + uygulama filtresi + **PostgreSQL Row-Level Security** + bileşik FK (bkz. aşağı) |
| **Operasyon** | Arka planda belge işleme, atomik sürüm değiştirme, tenant başına rate limit (Redis), health check, request-id, Alembic migration'ları |

## 🏗️ Mimari

```
                    ┌────────────────────────────── FastAPI (app/) ──────────────────────────────┐
  X-API-Key ──────▶ │ api/v1  ─ health · documents · chat · admin   (deps: auth → TenantContext)  │
                    │ services ─ DocumentService · PgHybridRetriever · RAGService · TenantService  │
                    │ core     ─ config · db (tenant oturumları, RLS bağlamı) · redis · security    │
                    └───────────────┬───────────────────────────────┬───────────────────────────┘
                                    │ ortak RAG çekirdeği (src/)    │
            document_loader · text_utils · hybrid_retriever (BM25, RRF, reranker) · rag_engine
            (rewrite_query, katı talimat, atıf doğrulama, güven skoru) · embeddings · llm
                                    │                               │
                     PostgreSQL 16 + pgvector (RLS)            Redis 7 (rate limit)
```

```
app/
├── main.py                 # Uygulama fabrikası, lifespan, middleware
├── api/deps.py             # Kimlik doğrulama, rate limit, admin
├── api/v1/endpoints/       # health · documents · chat · admin
├── core/                   # config · db · security · redis · logging
├── models/                 # Tenant · Document · DocumentChunk (Vector)
├── schemas/                # Pydantic v2 istek/yanıt modelleri
└── services/               # İş mantığı (oturumları servisler yönetir)
src/                        # Arka uçtan bağımsız RAG çekirdeği (API ve Streamlit ortak)
migrations/                 # Alembic (şema, HNSW, RLS, uygulama rolü)
docker/postgres/initdb/     # rag_app rolünü oluşturan init betiği
tests/                      # API testleri gerçek PostgreSQL (PGlite) üzerinde
```

### 🔒 Çok kiracılı izolasyon — derinlemesine savunma

1. **Kimlik:** Tenant yalnızca `X-API-Key` başlığından (SHA-256 özetiyle) çözülür; istemci tenant_id gönderemez.
2. **Uygulama:** Tüm sorgular zorunlu `TenantContext` alır; her sorguda açık `tenant_id` + `department` filtresi vardır. Başka tenant'ın kaynağı **404** döner (varlığı sızdırılmaz).
3. **Veritabanı (RLS):** `documents` ve `document_chunks` üzerinde `ENABLE` + `FORCE ROW LEVEL SECURITY`. Her transaction başında `set_config('app.tenant_id', …, true)` otomatik uygulanır (SQLAlchemy `after_begin`) ve transaction bitince sıfırlanır. Ayar yoksa hiçbir satır görünmez (**fail-closed**).
4. **Rol ayrımı:** Uygulama `rag_app` rolüyle bağlanır (`NOSUPERUSER NOBYPASSRLS`); migration'lar ayrı sahip rolle çalışır.
5. **Şema:** Parçalar `(document_id, tenant_id)` bileşik FK ile belgeye bağlıdır — bir parçanın tenant'ı belgesininkinden farklı olamaz.
6. **Önbellek:** BM25 indeks anahtarı tenant_id içerir; indeksler tenant'lar arasında paylaşılmaz.

Testler bu katmanları ayrı ayrı doğrular — RLS testleri **uygulama filtresi olmadan ham SQL** çalıştırır.

## 🚀 Çalıştırma (Docker — tek komut)

```bash
cp .env.example .env          # GOOGLE_API_KEY ve ADMIN_API_KEY değerlerini girin
docker compose up -d --build  # postgres(pgvector) + redis + api; migration otomatik
```

Swagger UI: **http://localhost:8000/docs**

1. **Tenant oluşturun** (API anahtarı yalnızca bir kez gösterilir):
   ```bash
   curl -X POST localhost:8000/api/v1/admin/tenants -H "X-Admin-Key: $ADMIN_API_KEY" \
        -H "Content-Type: application/json" -d '{"name":"Örnek Holding A.Ş.","slug":"ornek-holding"}'
   # veya: docker compose exec api python scripts/create_tenant.py "Örnek Holding" ornek-holding
   ```
2. Swagger'da **Authorize** → `TenantApiKey` alanına anahtarı girin.
3. `POST /api/v1/documents/upload` — `data/sample_docs/` altındaki belgeleri `İnsan Kaynakları` / `Finans` departmanına yükleyin (`wait=true` ile senkron).
4. `POST /api/v1/chat/query` — ör. `{"question": "250.000 TL üzeri harcamayı kim onaylar?", "departments": ["Finans"]}`

> API anahtarı olmadan denemek için `.env` içinde `EMBEDDING_PROVIDER=hashing` ve `LLM_PROVIDER=none` ayarlayın (arama ve kanıtlar çalışır, yanıt üretilmez).

### Yerel geliştirme (Docker'sız API)

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head                                  # DATABASE_MIGRATION_URL ile
uvicorn app.main:create_app --factory --reload
```

## 📡 Uç noktalar

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/api/v1/health` | DB, Redis, sağlayıcı durumu (kimlik doğrulama yok) |
| POST | `/api/v1/documents/upload` | Çok parçalı yükleme (`file`, `department`, `?wait=`) → 202 / 201 / 200 duplicate |
| GET | `/api/v1/documents` | Listeleme (`department`, `status`, `limit`, `offset`) |
| GET / DELETE | `/api/v1/documents/{id}` | Durum / silme |
| POST | `/api/v1/chat/query` | Soru → yanıt + atıflar + kanıtlar + güven + süreler |
| POST | `/api/v1/analytics/query` | **Birleşik analitik:** doküman (RAG) + satış verisi (Text-to-SQL) tek yanıtta |
| GET | `/api/v1/analytics/schema` | Sorgulanabilir tablolar ve sütunlar |
| POST | `/api/v1/analytics/demo-data` | Kendi tenant'ınız için gerçekçi B2B satış verisi (`ENABLE_DEMO_DATA`) |
| POST | `/api/v1/admin/tenants` | Tenant oluşturma (`X-Admin-Key`) |

## 📊 Text-to-SQL analitik ajanı

`customers`, `products`, `orders`, `order_items` tabloları (B2B toptan satış / üretim) üzerinde doğal dil soruları:

```
soru → yönlendirme (doküman / veri / ikisi) → SQL üretimi (Gemini, JSON) → SQLGuard doğrulaması
     → salt-okunur çalıştırma → hata varsa geri besleme ile yeniden deneme (≤3)
     → Python analizi (istatistik, içgörü) + Vega-Lite grafik → sonuca dayalı özet
```

**Güvenlik katmanları** (her biri tek başına test edilir):

| Katman | Koruma |
|---|---|
| `SQLGuard` (sqlglot AST) | Tek ifade · yalnızca SELECT/küme işlemi · ağaçta DML/DDL/`INTO`/`FOR UPDATE` yok (CTE içindeki `DELETE` dahil) · yalnızca 4 tablo + CTE'ler · `pg_catalog`/`information_schema` yok · `tenant_id` yok · fonksiyon beyaz listesi (`pg_*`, `set_config`, `dblink`, `lo_*` yasak) · zorunlu `LIMIT` · ağaçtan **yeniden üretilen** SQL çalıştırılır (yorumlar düşer) |
| Transaction | `SET TRANSACTION READ ONLY` · `statement_timeout` |
| Rol | `rag_readonly`: yalnızca 4 tabloda `SELECT`; belge/tenant tablolarına erişim yok; `set_config` çalıştırma yetkisi yok (tenant bağlamı değiştirilemez) |
| RLS | Tüm analitik tablolarda `FORCE ROW LEVEL SECURITY`; bileşik FK'larla tenant'lar arası ilişki imkânsız |

Birleşik yanıtta her bölüm kendi doğrulanmış kaynağından gelir (RAG: atıflı yanıt; veri: sorgu sonucuna dayalı özet) — birleştirme LLM'e yeniden yazdırılmaz.

## 🧪 Testler

```bash
npm install --prefix tests/pglite   # bir kez: gömülü PostgreSQL + pgvector (PGlite/WASM)
pytest
```

* `tests/api/` — REST API, tenant izolasyonu (API + servis + RLS), belge yaşam döngüsü, katı bağlam. Docker gerektirmez; Node.js yoksa otomatik atlanır.
* `tests/test_*.py` — belge işleme, chunking, hibrit arama, RAG motoru (ağ gerektirmez).

## 🖥️ Streamlit vitrin arayüzü (tek komut)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # bir kez (plotly dahil)
pip install -e .                  # bir kez: src/app/ui/rag_assistant paketleri her dizinden import edilebilir
streamlit run app.py              # http://localhost:8501
```

* Kurulum gerektirmez: örnek İK ve Finans dokümanları açılışta bellek içi hibrit indekse (vektör + BM25 + RRF)
  yüklenir. Koyu tema, KPI kartları, Plotly grafikleri, sayfa düzeyinde kaynak kartları ve **Canlı Demo Modu** içerir.
* `.env` dosyasında `GOOGLE_API_KEY` varsa yanıtları Gemini üretir; yoksa **çevrimdışı özetleyici** yalnızca
  dokümandaki cümleleri kaynak atfıyla aktarır (yeni bilgi üretmez).
* Her tarayıcı oturumu ve her tenant kendi indeksini kullanır; yüklenen belgeler kalıcı değildir.
* Departman filtresi varsayılan olarak boştur (tüm departmanlar). Filtre yüzünden reddedilen soruda neden açıkça
  yazılır, filtre dışındaki belgelerde ön kontrol yapılır ve tek tıkla filtresiz yeniden sorulabilir.
* Bileşik sorular ("X nasıl hesaplanır ve Y nedir?") alt sorulara ayrılır: dokümanda karşılığı olan kısım kaynaklı
  yanıtlanır, olmayan kısım için "dokümanlarda yer almamaktadır" notu düşülür. "dk"/"dakika", "sa"/"saat" gibi
  kısaltmalar aramada eşdeğerdir.
* Sorgu sayacı ve KPI'lar yalnızca gerçekten yanıtlanan sorulardan hesaplanır; rerun veya yarıda kesilen çizim
  sayacı şişirmez. "Sohbeti temizle" ve demo yeniden başlatma oturum metriklerini de sıfırlar.
* Vitrin modunda Streamlit'in Deploy butonu ve geliştirici menüsü gizlidir (`.streamlit/config.toml` →
  `[client] toolbarMode = "minimal"`; geliştirirken `"auto"` yapılabilir).
* REST API'ye bağlı çok kiracılı arayüz: `streamlit run ui/streamlit_app.py` (API: `uvicorn app.main:create_app --factory`).

## 🛣️ Yol haritası

- SSE ile akışlı yanıt uç noktası
- Kuyruk tabanlı ingestion (Arq/Celery) ve çoklu worker ölçekleme
- Kullanıcı düzeyinde departman yetkileri (SSO/OIDC)
- Büyük korpuslar için BM25 yerine PostgreSQL tam metin araması / ParadeDB
- RAGAS ile otomatik kalite değerlendirmesi
