# Kurumsal RAG Asistanı — Proje Notları

Çok kiracılı (multi-tenant) B2B SaaS RAG platformu: FastAPI + PostgreSQL/pgvector + Redis + Google Gemini.
Kullanıcıyla Türkçe çalışılır.

## Mimari (kısa)

- `app/` — FastAPI REST API (`api/v1`, `core`, `models`, `schemas`, `services`). Uygulama fabrikası:
  `uvicorn app.main:create_app --factory` (modül düzeyinde `app` nesnesi YOK, bilinçli).
- `src/` — arka uçtan bağımsız RAG çekirdeği (belge işleme, hibrit arama, katı bağlam, atıf/güven). API ve
  Streamlit ortak kullanır. `src/` ChromaDB'yi yalnızca `src/vector_store.py` içinde import eder.
- `migrations/` — Alembic: `0001` belgeler + RLS, `0002` satış analitiği tabloları + `rag_readonly` rolü.
- Multi-Agent: `app/services/orchestrator_service.py` (`MultiAgentOrchestrator`) niyeti (DOCUMENT_ONLY/SQL_ONLY/HYBRID)
  belirler, RAG + Text-to-SQL'i `asyncio.gather` ile paralel çalıştırır, sonucu B2B raporuna sentezler
  (LLM yoksa/başarısızsa şablon rapor). Uç nokta: `POST /api/v1/agent/query`; şemalar `app/schemas/agent.py`.
- Çok kiracılık: tenant yalnızca `X-API-Key`'den çözülür; `app/core/db.py` her transaction başında
  `app.tenant_id` + rolü `after_begin` olayıyla ayarlar; tüm tenant tablolarında `FORCE ROW LEVEL SECURITY`.
- Text-to-SQL: `app/services/sql_guard.py` (sqlglot, deny-by-default) → `rag_readonly` rolü + READ ONLY tx.
  `set_config` yetkisi PUBLIC'ten alınıp yalnızca `rag_app`'e verilir (migration 0002).

## Komutlar

```powershell
.\.venv\Scripts\activate
npm install --prefix tests/pglite        # bir kez: testler için gömülü PostgreSQL (PGlite) + pgvector
pytest                                    # API testleri PGlite üzerinde gerçek Postgres ile; tests/agent ve UI testleri sahte servislerle
pip install -r requirements-dev.txt       # tam ortam (requirements.txt = yalnızca Streamlit Cloud vitrini)
pip install -e .                          # bir kez: src/app/ui/rag_assistant paketlerini her dizinden import edilebilir yapar
streamlit run app.py                      # vitrin arayüzü: kurulumsuz, bellek içi hibrit indeks + Canlı Demo Modu
streamlit run ui/streamlit_app.py         # B2B arayüz (API'ye bağlı; tenant profilleri .streamlit/secrets.toml → [tenants.<id>])
python scripts/quick_check.py --offline   # Chroma tabanlı çekirdeğin hızlı terminal doğrulaması
python scripts/dev_pglite.py [--port 8010] [--no-seed]  # Docker'sız dev sunucu: PGlite + tek süreçli uvicorn --factory,
                                          # açılışta migration + örnek tenant/doküman/demo veri (API anahtarını yazar)
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
5. Faz 4 — Multi-Agent orkestratör + `/api/v1/agent/query` + `ui/streamlit_app.py` (3 sekme: asistan, doküman
   yönetimi, veri analitiği; Tenant ID seçici, sohbet ve analiz sonuçları tenant başına ayrı).
   Testler: `tests/api/test_agent_api.py` (PGlite), `tests/agent/` (sahte RAG/SQL/LLM ile izole orkestratör +
   uç nokta), `tests/test_ui_app.py` (streamlit AppTest; `/agent/query` yanıtı gerçek uç noktadan).

Doğrulanmamış / bilinen açıklar:
- **Docker Compose hiç çalıştırılmadı** (bu makinede Docker/WSL yok). `docker/postgres/initdb/01-app-role.sh`
  hiç test edilmedi. Testler PGlite'ın PostgreSQL 18'i üzerinde; compose `pgvector/pgvector:pg16` kullanır.
- **Gemini hiç çağrılmadı** (API anahtarı yok). Model adları (`gemini-2.5-flash`, `-flash-lite`,
  `gemini-embedding-001`) doğrulanmadı; `MIN_RELEVANCE=0.30` eşiği yalnızca çevrimdışı reranker ile kalibre edildi.
- `app.py` = vitrin arayüzü: `src/local_engine.py` (bellek içi depo + `HybridRetriever` + `RAGEngine`; anahtar yoksa
  `OfflineSummarizer` yalnızca bağlam cümlelerini atıfla aktarır), `ui/showcase_theme.py` (CSS, HTML kartlar — tüm
  dinamik metin `html.escape`), `ui/showcase_charts.py` (Plotly; yoksa Vega-Lite yedeği). `rag_assistant` artık
  yalnızca eski testlerde kullanılıyor. Testler: `tests/test_local_engine.py`, `tests/test_showcase_ui.py` (AppTest).
- Araştırmacı + Denetçi: `src/auditor.py`. Araştırmacı = `RAGEngine` taslağı; Denetçi yanıtı yalnızca getirilen parçalarla
  karşılaştırır (`approved`/`revised`/`rejected`/`skipped`). Anahtar varsa `GeminiAuditor` (fast=True, Flash) + kural
  tabanlı sayısal güvenlik ağı; yoksa `RuleBasedAuditor` (örtüşme ≥%60, kaynakta olmayan sayı/tarih/oran ve alıntı
  kontrolü). `LocalWorkspace.ask_audited`; arayüzde "Yüksek Doğruluk (Denetçi Modu)" anahtarı (varsayılan açık),
  rozetler ve "Denetçi Kontrol Raporu". Testler: `tests/test_auditor.py`. Gemini denetçi yalnızca sahte LLM ile test edildi.
- Vitrin akışı (`app.py`): soru gönderimleri callback ile `session_state.inbox`'a yazılır, `process_submission` yanıtı
  **çizimden önce** sohbete ve `ws.history`'ye kaydeder (tam bir kez; rerun/kesilen çizim sayacı şişirmez). KPI satırı
  `KpiSlot` ile sorgudan sonra çizilir. `LocalWorkspace.run()` → `Turn(response, audit, insight)`; `ask`/`ask_audited`
  bunun ince sarmalayıcıları. `AnswerInsight`: en iyi alaka, eşik, alt sorular, filtre notu (+ filtre dışı ön kontrol).
  Departman filtresi `dept_filter` anahtarlı, varsayılan boş; tenant değişince/demo başlayınca sıfırlanır.
- Bileşik soru: `src/compound.py` (temkinli ayırma: bağlaçla ayırmak için iki tarafta da soru sözcüğü gerekir; tüm
  soru reddedilirse `relaxed`). Önce yalnızca arama ile alt soru kapısı ölçülür; karşılığı olan alt sorular ayrı
  yanıtlanıp ayrı denetlenir, olmayanlar için not eklenir. Kısaltma eşitleme `src/text_utils.ABBREVIATIONS`
  (tokenize içinde; API tarafındaki BM25/HashingEmbedder'ı da etkiler — yalnızca dk/dak/sn/sa/km/kg/lt).
- Denetçi kapsam notlarını ("…dokümanlarda yer almamaktadır", sayı içermiyorsa) iddia saymaz; düzeltilen yanıtta
  paragraf/madde yapısı korunur. Testler: `tests/test_ux_fixes.py`.
- Plotly grafikleri tarayıcıda Streamlit'in plotly.js'i ile görsel olarak kontrol edildi, ancak Python tarafında
  gerçek plotly paketi yerine test amaçlı bir taklit kullanıldı (geliştirme ortamında plotly yoktu; kullanıcının
  .venv'inde de kurulu değil → testler Vega-Lite yolunu çalıştırır, Plotly yolu `tests/test_ux_fixes.py`'de taklitle).
- Yükleme alanının Türkçeleştirilmesi Streamlit 1.67 DOM'una (data-testid) dayalı CSS'tir; Streamlit sürümü
  değişirse `ui/showcase_theme.uploader_css` gözden geçirilmeli.
- Tam pytest paketi en son kullanıcının Windows .venv'inde 285 testle geçti (Denetçi katmanından sonra).
  `tests/test_ux_fixes.py` (31 test) eklendikten sonra tam paket orada yeniden çalıştırılmadı; geliştirme
  ortamında API (PGlite/asyncpg) testleri ve pyarrow isteyen `tests/test_ui_app.py` dışındaki tüm paket geçti.
- `/agent/query` `MAX_QUESTION_CHARS` ayarını kontrol etmiyor (yalnızca şema sınırı 2000); orkestratör istisnası
  yakalanmıyor → middleware genel 500 döner (ayrıntı sızmaz, testli).
- Arka plan ingestion süreç içi (BackgroundTasks); BM25 önbelleği worker başına.
- Streamlit Cloud: ana dosya `app.py`, yalnızca `requirements.txt` kurulur (sunucu paketi eklemeyin; testi
  `tests/test_deployment.py`). `landing/index.html` statik tanıtım sayfası (CONFIG: demo URL + iletişim e-postası),
  `pazarlama/` lansman metinleri.

## Sıradaki adım

1. **Docker ile gerçek uçtan uca test:** kullanıcı `wsl --install` + Docker Desktop kurduktan sonra
   `docker compose up -d --build` → tenant oluştur (`/api/v1/admin/tenants`) → `/health` → `/documents/upload`
   (data/sample_docs) → `/chat/query` → `/analytics/demo-data` + `/analytics/query`. Hata çıkarsa düzelt.
2. Gerçek `GOOGLE_API_KEY` ile `python scripts/quick_check.py` ve analitik sorgularla eşik/kalite kalibrasyonu.
3. Sonraki faz adayları: SSE akışlı sohbet · kuyruk tabanlı ingestion (Arq) ·
   kullanıcı düzeyinde departman yetkileri (SSO/OIDC).
