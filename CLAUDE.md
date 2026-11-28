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

## Durum (2026-11-28 itibarıyla)

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
- **Gemini hiç çağrılmadı** (API anahtarı yok). Model adları (`gemini-3.5-flash`, `-flash-lite`,
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
- Gemini hata politikası (`src/llm.py`): `call_with_retry` 429/5xx/ağ hatalarını üstel beklemeyle 3 kez dener,
  400/401/403/404 beklemeden döner; `stream` yalnızca ilk parça gelmeden tekrar dener. `LLMError.user_message`
  Türkçe; `RAGResponse.error` doluysa yanıt "bilgi yok" reddi DEĞİL servis hatasıdır (Denetçi atlanır, arayüz
  "Tekrar dene" gösterir). Testler: `tests/test_robustness.py` (sahte `google.genai` ile).
  İstek zaman aşımı `LLM_TIMEOUT_SECONDS=30` (`HttpOptions(timeout=ms)`); bekleme üstel + en çok 8 sn. Ana model
  denemelerden sonra 503 / "overloaded" verirse veya zaman aşımına uğrarsa (aynı model ikinci kez beklenmez)
  istek `fast_llm_model` ile tekrarlanır (`LLM_FALLBACK_TO_FAST`); 429/4xx ve akış başladıktan sonraki hatalar
  model değiştirmez. `GeminiLLM.last_model` son yanıtı veren modeldir.
- "Bunu mu sormak istediniz?" (`src/suggestions.py`, `rag_engine` yalnızca yeniden dışa aktarır):
  - `CorpusHint` = kullanıcının ERİŞEBİLDİĞİ parçalardan belge adları, departmanlar, başlıklar (metadata `section`/
    `sheet_name` + parça içi `#`–`####`; numara ve "Kurum — " öneki ayıklanır), dağarcık, önekler, başlık→departman.
    `hint_from_index(index)` kapsamlı BM25 indeksinden kurar ve indeks nesnesinde önbellekler (vitrin:
    `HybridRetriever.index_for(depts)`, kilitli; API: `PgHybridRetriever.bm25_index` RLS'li `tenant_session` içinde).
  - `analyze_query` (LLM'siz): net soru → `method="skipped"`, model çağrılmaz. Türkçe ek toleransı: belge sözcüğünün
    öneki/çekimi bilinen sayılır; kök↔çekim ("gün"/"günü") ve ≤4 harfte tek harf değişimi ("izin"→"için") düzeltme
    değildir; cümle içi büyük harfli sözcük özel isimdir; stopword'e düzeltilmez. Kısa (≤2 içerik sözcüğü) ve soru
    kalıbı yok → belirsiz. Departman anılırsa ("ik prosedürü", "finans") niyetler o departmanın başlıklarıyla sınırlı;
    genel bölümler (Amaç, Kapsam, Yürürlük…) niyet önerilmez.
  - `QuerySuggester.suggest(question, hint, scope=)` asla yükseltmez; model (fast, `_QuerySuggestionOut`:
    has_suggestion/suggested_query/alternatives, hepsi zorunlu) her hatada sezgisel yedeğe düşer ve o sonuç
    önbelleğe alınmaz. Doğrulama: aynı soru/yalnızca harf-noktalama-Türkçe karakter farkı, boş, >200 karakter,
    yinelenen ve kapsam dışı DOSYA ADI içeren öneriler atılır; en çok 3 seçenek. LRU anahtarı
    `(scope, hint.fingerprint, normalize soru)`; scope API'de `(tenant_id, *sıralı departmanlar)`, vitrinde departmanlar.
  - Vitrin (`app.py`): `_suggest_pool()` (cache_resource ThreadPoolExecutor) ile ana aramayla EŞZAMANLI; token
    akışında hazırsa yanıtın üstündeki yuvaya çizilir, `finish()` sonrası en çok `suggest_wait`
    (`SUGGESTION_WAIT_SECONDS`, 1,5 sn) beklenir. `_submit(..., accepted=True)` (öneri/niyet butonu, örnek soru, demo)
    → öneri hiç çalışmaz (döngü yok). Tek öneri: `st.info` + "🔁 Önerilen soruyla sor"; çoklu: niyet butonları
    `suggest_{key}_{i}`; eski mesajlarda butonsuz liste. Model metni `_md` ile markdown kaçışlı.
    `chat_input(max_chars=2000)`. Eski imzalı (sıcak yenilemeden kalmış) `ws.suggest` hatası yutulur.
  - API: `ChatQuery.suggest` (varsayılan true; öneriden tekrar soruluyorsa false), `ChatResponse.suggestion`
    (`SuggestionOut` | null). `/chat/query` öneriyi `asyncio.create_task` ile `RAGService.ask`'la paralel başlatır,
    sonra `wait_for(SUGGESTION_WAIT_SECONDS)`; departmansız kullanıcıda öneri yok.
  - Testler: `tests/test_query_suggestion.py` (birim + AppTest), `tests/api/test_suggestion_api.py` (RBAC: istemde ve
    yanıtta yetkisiz departman/başka şirket belgesi yok, kapsamlar arası önbellek yok, yavaş öneri yanıtı
    geciktirmez). Gerçek Gemini ile öneri kalitesi hâlâ doğrulanmadı (yalnızca SDK sözleşmesi).
- Gemini SDK sözleşmesi: `tests/test_gemini_contract.py` gerçek `google-genai` (CI'da 3.x) ile `httpx.Client.send`'i
  taklit eder — tüm yapısal şemalar, SSE akışı, 503/overloaded/zaman aşımı → fast_model, 429, 400/403/404, embedding,
  istek zaman aşımının HTTP katmanına ulaşması. Yapısal çıktı modelleri `src.llm.StructuredOutput`'tan türemeli
  (JSON şemasında `default` ilan edilmez; Gemini API reddeder). `generate_json` SDK'nın ValueError/TypeError'ını
  `LLMError(retryable=False)` yapar; yeniden yazma/rerank/Denetçi `ValueError`'ı da yakalar. `src.llm.http_options`
  zaman aşımını verir ve SDK'nın kendi tekrar denemesini (varsa `HttpRetryOptions`) tek denemeye sabitler;
  `GeminiEmbedder(timeout=)` aynı ayarı kullanır.
- Sıcak yenileme: `src/` içinde fonksiyon içinden `from src…` yapmayın (Streamlit modülleri boşaltırken
  `KeyError: 'src'`); `app.py` yarım/yabancı `src` modüllerini temizleyip yeniden yükler, eski sınıftan kalan
  çalışma alanını yeniden kurar. Atıf eşleştirme önce uzantılı tam adla yapılır (tr.pdf ≠ tr.docx).
- Akışlı yanıt: `LocalWorkspace.start()` → `PendingTurn` (`tokens` ekrana akar, `finish()` Denetçi + tek seferlik
  kayıt); `run()` = `start().finish()`. Bileşik/kurtarma/red yolları tek parça verir. Arayüz akışı `st.empty()`
  yuvasına yazar, Denetçi sonrası nihai metni aynı yuvaya koyar; rozet + güven + kaynaklar altta.
- Güven: `calibrated_confidence` (kanıt 0.35 + atıf kapsamı 0.25 + Denetçi doğrulaması 0.40, en çok %97) →
  `AnswerInsight.confidence/confidence_level`; KPI ortalaması bunu kullanır. `RAGResponse.confidence` ham kalır.
- Atıflar ekranda `ui/citations.number_citations` ile `[1]`, `[2]`; saklanan metin etiketli kalır (doğrulama için).
- Vitrin `Hukuk` departmanını ekler (yalnızca `LocalWorkspace`; API varsayılanları değişmedi). Örnek sözleşme PDF'i
  `scripts/make_sample_contract.py` ile üretilir; örnek belgeler `app._cached_samples` ile sunucu başına önbellekli.
  Demo senaryosu 6 soru (4 politika + sözleşme + Mars). `.github/workflows/keep-alive.yml` demoyu uyanık tutar.
- Streamlit Cloud: ana dosya `app.py`, yalnızca `requirements.txt` kurulur (sunucu paketi eklemeyin; testi
  `tests/test_deployment.py`). `landing/index.html` statik tanıtım sayfası (CONFIG: demo URL + iletişim e-postası),
  `pazarlama/` lansman metinleri.

## Faz 2 – kimlik ve yetki (dal `faz2/kimlik-rbac`; CI tam yeşil olduktan sonra 2026-11-28'de main'e fast-forward)

- Teknik plan ve 8 güvenlik kontrolü: Claude Docs "Kurumsal RAG Pro – Faz 2 Teknik Denetim ve Uygulama Planı".
- Migration 0003: `users` (RLS'siz), `memberships`/`invitations`/`user_sessions` (FORCE RLS) + belge
  tablolarında `department_scope` (RESTRICTIVE). `app.departments`: JSON dizi veya `'*'`; ayar yoksa belge görünmez.
  Başta ön kontrol: departmanı şirket listesinde olmayan belge varsa durur. **Üretimde henüz uygulanmadı.**
- `app/core/{passwords,tokens,permissions}.py`: scrypt; `rs_/ri_<tenant hex>.<gizli>` belirteçleri (yalnızca SHA-256
  saklanır); rol matrisi (tek kaynak, `tests/test_auth_core.py` plandaki tabloyla birebir).
- `app/services/auth_service.py` + `/api/v1/auth/*`, `/api/v1/company/*`. `get_principal`: Bearer veya `X-API-Key`
  (ikisi birden 400). `TenantContext.departments` = izinli departmanlar; servisler `tenant_session(tenant.id,
  tenant.departments)` çağırır. Tanımsız departman 422, yetkisiz 403 (`DepartmentForbiddenError`).
- Satış analitiği yalnızca owner/admin; `/agent/query` diğer rollerde DOCUMENT_ONLY'ye zorlanır.
- CI: `.github/workflows/tests.yml` (PGlite dahil tam paket, `REQUIRE_PGLITE=1` → API testleri atlanamaz).
  Tetik: `faz2/**` push + `main`'e PR (main'e doğrudan push test çalıştırmaz → önce dalda yeşil, sonra birleştir).
  CI günlükleri herkese açık API'den okunabilir: `/repos/<sahip>/<repo>/actions/jobs/<id>/logs`.
- Geliştirme ortamı (bulut oturumu) notu: yerelde streamlit/httpx/sqlalchemy/google-genai yok ve paket kurulamıyor;
  `PYTHONPATH=/tmp/stubs:. pytest --noconftest` yalnızca saf çekirdek testlerini çalıştırır (rank_bm25 taklidi
  `doc_freqs` içermediği için aramalı testler yerelde düşer). Doğrulama kaynağı CI'dır.
- B2B arayüz: "🔐 Kullanıcı girişi" modu (Bearer); API anahtarı profilleri aynen çalışır.

## Sıradaki adım

1. **Docker ile gerçek uçtan uca test:** kullanıcı `wsl --install` + Docker Desktop kurduktan sonra
   `docker compose up -d --build` → tenant oluştur (`/api/v1/admin/tenants`) → `/health` → `/documents/upload`
   (data/sample_docs) → `/chat/query` → `/analytics/demo-data` + `/analytics/query`. Hata çıkarsa düzelt.
2. Gerçek `GOOGLE_API_KEY` ile `python scripts/quick_check.py` ve analitik sorgularla eşik/kalite kalibrasyonu;
   "Bunu mu sormak istediniz?" model önerilerinin kalitesi (yanlış pozitif oranı) ve `SUGGESTION_WAIT_SECONDS`.
   B2B arayüz (`ui/streamlit_app.py`, `/agent/query`) henüz öneri göstermiyor; `/chat/query` döndürüyor.
3. Sonraki faz adayları: SSE akışlı sohbet · kuyruk tabanlı ingestion (Arq) ·
   kullanıcı düzeyinde departman yetkileri (SSO/OIDC).
