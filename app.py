"""Kurumsal RAG Asistanı — B2B SaaS vitrin arayüzü (tek komutla çalışır).

Çalıştırma (proje kökünde, sanal ortam etkinken):
    streamlit run app.py

Kurulum gerektirmez: örnek kurumsal dokümanlar açılışta bellek içi hibrit indekse yüklenir.
`.env` içinde GOOGLE_API_KEY varsa yanıtları Gemini üretir; yoksa yalnızca dokümandaki cümleleri
kaynak atfıyla aktaran çevrimdışı özetleyici çalışır. Çok kiracılı REST API'ye bağlı arayüz için:
`ui/streamlit_app.py` (bkz. README).

Soru akışı "tam bir kez" ilkesine göre kuruludur: butonlar ve sohbet kutusu soruyu yalnızca bir gelen
kutusuna (inbox) yazar (callback). Soru yanıtlandığı anda — ekrana çizilmeden önce — sohbete ve oturum
metriklerine kaydedilir; böylece rerun, yarıda kesilen çizim veya sayfa yenileme sayaçları şişiremez.
KPI satırı sorgu işlendikten sonra doldurulur, yani her zaman güncel değeri gösterir.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import uuid
from pathlib import Path

# Proje kökü import yolunun başında: `src.*` ve `ui.*` hangi dizinden/başlatıcıdan çalıştırılırsa çalıştırılsın
# bulunur ve site-packages'taki aynı adlı bir paket (bazı kütüphaneler üst düzey `src` kurar) gölgeleyemez.
PROJECT_ROOT = Path(__file__).resolve().parent
if sys.path[:1] != [str(PROJECT_ROOT)]:
    while str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
_CORE_PACKAGES = ("src", "ui")


def _core_is_ours() -> bool:
    """sys.modules'taki `src`/`ui` bu projeye mi ait ve tutarlı mı (alt modülü olup ebeveyni eksik değil)?"""
    for pkg in _CORE_PACKAGES:
        mod = sys.modules.get(pkg)
        children = any(name.startswith(pkg + ".") for name in list(sys.modules))
        if mod is None:
            if children:
                return False  # sıcak yenileme ebeveyni sildi, çocuklar kaldı → yarım durum
            continue
        origin = getattr(mod, "__file__", None) or next(iter(getattr(mod, "__path__", [])), "")
        if not origin or not Path(origin).resolve().is_relative_to(PROJECT_ROOT):
            return False  # başka bir `src` paketi
    return True


def _purge_core_modules() -> None:
    for name in list(sys.modules):
        if name in _CORE_PACKAGES or name.startswith(tuple(p + "." for p in _CORE_PACKAGES)):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


if not _core_is_ours():
    _purge_core_modules()

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Kurumsal RAG Asistanı · Kurumsal Bilgi Platformu",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",  # masaüstünde açık, mobilde kapalı başlar
)

for _attempt in range(3):
    # Streamlit sıcak yenilemesi (veya başka bir oturumun yenilemesi) `src.*` modüllerini tam bu sırada
    # sys.modules'tan çıkarırsa import `KeyError: 'src'` ile yarıda kalabilir: temizleyip tekrar dene.
    try:
        from src.auditor import AuditResult
        from src.config import get_settings
        from src.document_loader import DocumentProcessingError
        from src.local_engine import OFFLINE_MODE, AnswerInsight, LocalWorkspace
        from src.rag_engine import QuerySuggestion, RAGResponse
        from ui import showcase_charts as charts
        from ui import showcase_theme as theme
        from ui.citations import number_citations, streaming_view
        break
    except (KeyError, ImportError):
        if _attempt == 2:
            raise
        _purge_core_modules()
        time.sleep(0.05 * (_attempt + 1))

UPLOAD_TYPES = ["pdf", "docx", "xlsx", "txt", "md"]
SAMPLE_TENANT = "Örnek Holding A.Ş."
TENANTS = {
    SAMPLE_TENANT: "Örnek İK yönetmeliği ve finans prosedürü yüklü demo şirketi",
    "Ege Vadi Gıda A.Ş.": "Boş çalışma alanı — kendi belgelerinizi yükleyin",
}
SAMPLE_QUESTIONS = [
    "Fazla mesai ücreti nasıl hesaplanır?",
    "Uzaktan çalışma kuralları nelerdir?",
    "Satın almada kaç teklif almak gerekir?",
    "Yurt içi seyahatte günlük yemek harcırahı ne kadar?",
]
# Butonlarda kısa etiket; tıklanınca tam soru gönderilir (etiket kesilmez, tam soru ipucunda görünür)
SAMPLE_LABELS = [
    "⏱️ Fazla mesai nasıl hesaplanır?",
    "🏠 Uzaktan çalışma kuralları",
    "🧾 Kaç teklif almak gerekir?",
    "🍽️ Günlük yemek harcırahı",
]
CONTRACT_QUESTION = "Bakım sözleşmesinde P1 çözüm süresi aşılırsa hizmet kredisi ne kadar?"  # PDF + tablo
DEMO_SCRIPT = [*SAMPLE_QUESTIONS, CONTRACT_QUESTION,
               "Şirketin Mars'taki ofisinin adresi nedir?"]  # son soru: halüsinasyon koruması
FILTER_KEY = "dept_filter"
CHAT_KEY = "chat_box"
ALL_DEPARTMENTS = "Tüm departmanlar"


# ====================================================================== durum
def _state():  # noqa: ANN202
    ss = st.session_state
    ss.setdefault("workspaces", {})   # tenant → LocalWorkspace (oturuma özel)
    ss.setdefault("chats", {})        # tenant → mesajlar
    ss.setdefault("inbox", [])        # gönderilmiş, henüz yanıtlanmamış sorular (her biri bir kez işlenir)
    ss.setdefault("demo_queue", [])
    ss.setdefault("tenant", SAMPLE_TENANT)
    ss.setdefault(FILTER_KEY, [])     # departman filtresi: varsayılan boş = tüm departmanlar
    ss.setdefault("stream_delay", 0.012)  # yazma efekti (sn/kelime); testlerde 0
    ss.setdefault("demo_delay", 0.6)
    ss.setdefault("auditor_on", True)  # Yüksek Doğruluk (Denetçi Modu) varsayılan açık
    return ss


@st.cache_resource(show_spinner=False)
def _cached_samples(signature: tuple) -> list:
    """Örnek belgeler (PDF ayrıştırma + parçalama) sunucu başına bir kez işlenir; oturumlar paylaşır.

    `signature` dosya adları/değişiklik zamanları ve parçalama ayarlarıdır: dosya değişirse yeniden işlenir.
    """
    return LocalWorkspace(get_settings()).sample_documents()


def _sample_signature() -> tuple:
    s = get_settings()
    files = sorted((p.name, p.stat().st_mtime_ns) for p in Path(s.sample_docs_dir).glob("*.*"))
    return (str(s.sample_docs_dir), s.chunk_size, s.chunk_overlap, tuple(files))


def workspace(tenant: str) -> LocalWorkspace:
    ss = st.session_state
    current = ss.workspaces.get(tenant)
    if current is not None and type(current) is not LocalWorkspace:
        # Kod sıcak yenilendi: oturumdaki nesne eski sınıfa ait, yeni kodla uyumsuz olabilir → yeniden kur
        ss.workspaces.pop(tenant)
        st.toast("♻️ Uygulama güncellendi; bilgi tabanı yeniden hazırlandı.")
    if tenant not in ss.workspaces:
        with st.spinner("Bilgi tabanı hazırlanıyor…"):
            ws = LocalWorkspace(get_settings())
            if tenant == SAMPLE_TENANT:
                ws.load_sample_documents(_cached_samples(_sample_signature()))
            ss.workspaces[tenant] = ws
    return ss.workspaces[tenant]


def messages(tenant: str) -> list[dict]:
    return st.session_state.chats.setdefault(tenant, [])


# ====================================================================== callback'ler (gövdeden önce çalışır)
def _submit(text: str, departments: list[str] | None = None) -> None:
    """Soruyu gelen kutusuna yazar. `departments=None` → işlendiği andaki aktif filtre kullanılır."""
    text = (text or "").strip()
    if text:
        ss = st.session_state
        ss.inbox.append({"id": uuid.uuid4().hex, "tenant": ss.tenant, "text": text,
                         "departments": departments})


def _on_chat_submit() -> None:
    _submit(st.session_state.get(CHAT_KEY) or "")


def _clear_filter() -> None:
    st.session_state[FILTER_KEY] = []


def _retry_without_filter(question: str) -> None:
    _clear_filter()
    _submit(question, departments=[])


def _reset_tenant_session(tenant: str) -> None:
    ss = st.session_state
    ss.chats[tenant] = []
    ss.inbox = [s for s in ss.inbox if s["tenant"] != tenant]
    if tenant in ss.workspaces:
        ss.workspaces[tenant].reset_history()  # "X sorgu bu oturumda" sohbetle birlikte sıfırlanır


def _start_demo() -> None:
    ss = st.session_state
    ss.tenant = SAMPLE_TENANT
    _clear_filter()
    _reset_tenant_session(SAMPLE_TENANT)
    ss.demo_queue = list(DEMO_SCRIPT)


def _stop_demo() -> None:
    st.session_state.demo_queue = []


def _remove_document(ws: LocalWorkspace, doc_id: str) -> None:
    ws.store.remove_document(doc_id)


# ====================================================================== kenar çubuğu
def sidebar() -> tuple[str, list[str]]:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 🧠 Kurumsal RAG Asistanı")
        st.caption("Kaynak atıflı kurumsal bilgi platformu")

        tenant = st.selectbox(
            "Aktif tenant (şirket)", list(TENANTS), key="tenant", on_change=_clear_filter,
            help="Her tenant'ın bilgi tabanı ve sohbet geçmişi birbirinden tamamen ayrıdır.",
        )
        st.caption(TENANTS[tenant])
        ws = workspace(tenant)
        options = list(ws.settings.departments)
        current = list(ss.get(FILTER_KEY) or [])
        if any(d not in options for d in current):  # eski/geçersiz seçim takılı kalmasın
            ss[FILTER_KEY] = [d for d in current if d in options]
        departments = st.multiselect(
            "Departman filtresi", options, key=FILTER_KEY, placeholder=ALL_DEPARTMENTS,
            help="Boş bırakılırsa tüm departmanların belgelerinde arama yapılır. Filtre seçilirse yalnızca "
                 "o departmanların belgeleri taranır; filtre nedeniyle reddedilen soruda nedeni gösterilir.",
        )
        st.toggle(
            "🛡️ Yüksek Doğruluk (Denetçi Modu)", key="auditor_on",
            help="Araştırmacı ajanın yanıtı, Denetçi ajan tarafından yalnızca getirilen kaynak parçalarıyla "
                 "karşılaştırılır; kaynakta geçmeyen sayı/tarih/oran veya bilgi düzeltilir ya da reddedilir.",
        )
        st.caption(f"Denetçi: {ws.auditor.name}" if ss.auditor_on else "Denetçi kapalı: yanıtlar denetlenmeden gösterilir.")

        st.divider()
        st.markdown("**🎬 Canlı Demo Modu**")
        st.caption("Hazır soruları sırayla sorar; kaynak kartlarını ve halüsinasyon korumasını gösterir.")
        c1, c2 = st.columns(2)
        c1.button("▶ Başlat", type="primary", width="stretch", disabled=bool(ss.demo_queue),
                  on_click=_start_demo)
        c2.button("⏹ Durdur", width="stretch", disabled=not ss.demo_queue, on_click=_stop_demo)
        st.button("🧹 Sohbeti temizle", width="stretch", on_click=_reset_tenant_session, args=(tenant,),
                  help="Sohbeti ve bu tenant'ın oturum metriklerini (sorgu sayısı, ortalama güven) sıfırlar.")

        st.divider()
        st.markdown("**📤 Belge yükle**")
        max_mb = int(st.get_option("server.maxUploadSize") or 25)
        st.markdown(theme.uploader_css(max_mb, UPLOAD_TYPES), unsafe_allow_html=True)
        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader("Belgeler", type=UPLOAD_TYPES, accept_multiple_files=True,
                                     help=f"PDF, DOCX, XLSX, TXT veya MD · dosya başına en fazla {max_mb} MB")
            dept = st.selectbox("Departman", ws.settings.departments)
            submitted = st.form_submit_button("İndeksle", width="stretch")
        if submitted and files:
            for f in files:
                try:
                    doc, added = ws.add_bytes(f.name, f.getvalue(), dept)
                    st.toast(f"✅ {doc.filename}: {len(doc.chunks)} parça" if added else f"♻️ {f.name} zaten indeksli")
                except (DocumentProcessingError, ValueError) as exc:
                    st.error(f"{f.name}: {exc}")
                except Exception:  # noqa: BLE001 — bozuk/beklenmedik dosya uygulamayı düşürmesin
                    logging.getLogger(__name__).exception("Belge işlenemedi: %s", f.name)
                    st.error(f"{f.name}: dosya işlenemedi (bozuk veya desteklenmeyen içerik). "
                             "Dosyayı yeniden kaydedip tekrar deneyin.")
        elif submitted:
            st.warning("Önce yüklenecek bir dosya seçin.")

        st.divider()
        if ws.mode == OFFLINE_MODE:
            st.info("Yanıt modu: **Çevrimdışı özetleyici** (yalnızca belge cümleleri). "
                    "`.env` dosyasına (yerel) veya Streamlit Cloud **Secrets**'a `GOOGLE_API_KEY` ekleyerek "
                    "Gemini'yi açabilirsiniz.", icon="🔌")
        else:
            st.success("Yanıt modu: **Gemini** (katı bağlam + atıf doğrulama)", icon="✨")
    return tenant, list(departments)


# ====================================================================== bileşenler
def kpi_row(tenant: str, ws: LocalWorkspace) -> None:
    docs = ws.documents
    hist = ws.history  # yalnızca gerçekten yanıtlanan kullanıcı/demo sorguları (tam bir kez kaydedilir)
    avg_conf = round(sum(h.confidence for h in hist) / len(hist)) if hist else None
    avg_ms = round(sum(h.total_ms for h in hist) / len(hist)) if hist else None
    healthy = ws.store.chunk_count > 0
    cards = [
        theme.kpi_card("📚", "Doküman", str(len(docs)), f"{ws.store.chunk_count} bilgi parçası indeksli"),
        theme.kpi_card("🏢", "Aktif tenant", tenant, "İzole çalışma alanı · ayrı indeks", small=True,
                       dot=theme.SUCCESS),
        theme.kpi_card("🩺", "Sistem sağlığı", "Sağlıklı" if healthy else "Hazır · indeks boş",
                       f"Arama: vektör + BM25 + RRF · {ws.mode}", small=True,
                       dot=theme.SUCCESS if healthy else theme.WARNING),
        theme.kpi_card("🎯", "Ortalama güven", f"%{avg_conf}" if avg_conf is not None else "—",
                       f"{len(hist)} sorgu bu oturumda"),
        theme.kpi_card("⚡", "Ortalama yanıt", f"{avg_ms} ms" if avg_ms is not None else "—",
                       "Arama + yanıt üretimi"),
    ]
    for col, card in zip(st.columns(len(cards)), cards, strict=True):
        col.markdown(card, unsafe_allow_html=True)


class KpiSlot:
    """KPI satırının yeri sayfanın üstünde ayrılır; içeriği sorgular işlendikten sonra (bir kez) çizilir."""

    def __init__(self, tenant: str, ws: LocalWorkspace):
        self._box = st.container()
        self._tenant, self._ws = tenant, ws
        self._done = False

    def render(self) -> None:
        if not self._done:
            self._done = True
            with self._box:
                kpi_row(self._tenant, self._ws)


def render_chart(fig) -> None:  # noqa: ANN001
    if charts.PLOTLY:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.vega_lite_chart(spec=fig, width="stretch")


# Rapor bu uyarıları metrik/alt soru satırı olarak gösterir; ayrıca not olarak tekrarlanmaz
_FOLDED_WARNINGS = ("Yeterince alakalı bağlam bulunamadı", "Bileşik soru")


def render_retrieval_facts(resp: RAGResponse, insight: AnswerInsight | None) -> None:
    """Rapor gövdesi: en iyi alaka/eşik/kapsam, filtre uyarısı, alt sorular ve kalan doğrulama notları."""
    if insight is not None:
        st.markdown(theme.retrieval_facts(insight.best_relevance, insight.threshold, insight.departments),
                    unsafe_allow_html=True)
        if insight.filter_note:
            st.markdown(theme.filter_fact(insight.departments, insight.filter_probe), unsafe_allow_html=True)
        if insight.parts:
            st.markdown("**Alt sorular**")
            st.markdown("".join(theme.part_row(p.text, p.answered, p.relevance) for p in insight.parts),
                        unsafe_allow_html=True)
    for w in resp.warnings:
        if insight is None or not w.startswith(_FOLDED_WARNINGS):
            st.caption(f"ℹ️ {w}")


def render_audit_report(audit: AuditResult, resp: RAGResponse, insight: AnswerInsight | None) -> None:
    """Tek, sade "Denetçi Kontrol Raporu": karar özeti + arama metrikleri + iddia kontrolleri (rozetsiz)."""
    with st.expander("🛡️ Denetçi Kontrol Raporu"):
        st.markdown(f"**Özet:** {audit.summary}")
        st.caption(f"Denetçi: {audit.method} · Denetim süresi: {audit.duration_ms:.1f} ms · "
                   f"{len(audit.checks)} iddia kontrol edildi")
        render_retrieval_facts(resp, insight)
        if audit.checks:
            st.markdown("**İddia kontrolleri**")
            for c in audit.checks:
                st.markdown(theme.audit_row(c.text, c.reason, c.supported), unsafe_allow_html=True)


_MD_SPECIAL = str.maketrans({c: "\\" + c for c in "\\`*_{}[]()<>#+-.!|$~"})


def render_suggestion(suggestion: QuerySuggestion | None, *, key: str, latest: bool = True) -> None:
    """"Bunu mu sormak istediniz?" kutusu (cevabın hemen üstünde) + tek tıkla önerilen soruyla yeniden sor."""
    if suggestion is None or not suggestion.has_suggestion:
        return
    # Sondaki "?" kalıba eklenir (çift soru işareti olmasın); model çıktısı markdown olarak yorumlanmasın
    shown = suggestion.suggested_query.rstrip(" ?").translate(_MD_SPECIAL)
    st.info(f"Bunu mu sormak istediniz: **{shown}**?", icon="💡")
    if latest:
        st.button("🔁 Önerilen soruyla sor", key=f"suggest_{key}", on_click=_submit,
                  args=(suggestion.suggested_query,), help=suggestion.suggested_query)


def render_answer(msg: dict, *, key: str, delay: float = 0.0, latest: bool = True, text_slot=None) -> None:  # noqa: ANN001
    """Yanıt metni (Denetçi kararıyla) + rozet + meta satırı + filtre nedeni + kaynak kartları + rapor."""
    if text_slot is None:  # canlı akışta öneri kutusu zaten yanıt yuvasının üstüne çizildi
        render_suggestion(msg.get("suggestion"), key=key, latest=latest)
    resp: RAGResponse = msg["response"]
    audit: AuditResult | None = msg.get("audit")
    insight: AnswerInsight | None = msg.get("insight")
    if getattr(resp, "error", ""):
        # Servis hatası (Gemini 429/503…): "bilgi yok" reddi değildir → ayrı uyarı + tekrar dene
        if text_slot is not None:
            text_slot.empty()
        st.markdown(theme.service_error_card(resp.error), unsafe_allow_html=True)
        if latest:
            st.button("🔄 Tekrar dene", key=f"again_{key}", on_click=_submit,
                      args=(msg.get("question") or resp.question,))
        return
    audited = audit is not None and audit.status != "skipped"
    text = audit.answer if audited else resp.answer
    answered = resp.is_answered and not (audited and audit.status == "rejected")
    # Uzun etiketler ([IK_….md - Sayfa 4]) ekranda [1], [2] … olur; kaynak kartları aynı numaralarla listelenir
    shown, numbered = number_citations(text, resp.citations)
    if text_slot is not None:
        text_slot.markdown(shown)  # akışın gösterildiği yere nihai (Denetçi sonrası) metin yazılır
    elif delay > 0:
        def words():
            for w in shown.split(" "):
                yield w + " "
                time.sleep(delay)
        st.write_stream(words)
    else:
        st.markdown(shown)

    partial = bool(insight and insight.missing_parts) and answered
    status = ("🧩 Kısmi kaynaklı yanıt" if partial else "✅ Kaynaklı yanıt") if answered \
        else "🛡️ Bilgi yok — uydurma yanıt engellendi"
    by_label = {c.label: c for c in resp.citations}
    citations = [c for c in numbered if getattr(c, "label", None) in by_label] if answered else []
    if answered and not citations:
        citations = list(resp.citations)
    confidence = insight.confidence if insight is not None and insight.confidence_level else \
        (resp.confidence if answered else 0)
    level = insight.confidence_level if insight is not None and answered else ""
    badge = theme.audit_badge(audit.status) if audited else ""
    st.markdown(
        f'<div class="answer-foot">{badge}{theme.confidence_pill(confidence, level) if answered else ""}'
        f'<div class="meta">{status} · {resp.total_ms / 1000:.1f} sn · {len(citations)} kaynak</div></div>',
        unsafe_allow_html=True,
    )
    if insight is not None and insight.empty_index and not answered:
        st.info("Bu tenant'ın bilgi tabanında henüz belge yok. Kenar çubuğundaki **📤 Belge yükle** bölümünden "
                f"PDF, DOCX, XLSX, TXT veya MD yükleyin ya da **{SAMPLE_TENANT}** tenant'ını seçin.", icon="📭")
    if insight is not None and insight.filter_note:
        st.markdown(theme.filter_callout(insight.filter_note, insight.filter_probe), unsafe_allow_html=True)
        if latest:  # eski mesajlarda tekrar sorma butonu gösterilmez
            st.button("🔓 Filtreyi kaldır ve tüm belgelerde sor", key=f"retry_{key}",
                      on_click=_retry_without_filter, args=(msg.get("question") or resp.question,))

    if citations or audited:
        left, right = st.columns([3, 2]) if (citations and audited) else (st.container(), st.container())
        if citations:
            with left:
                with st.expander(f"📎 Kaynaklar ({len(citations)}) — bilginin alındığı doküman ve sayfa",
                                 expanded=False):
                    for n, c in enumerate(citations, start=1):
                        st.markdown(theme.citation_card(c.label, c.source, c.page, c.department, c.relevance,
                                                        c.evidence, number=n), unsafe_allow_html=True)
        if audited:
            with right:
                render_audit_report(audit, resp, insight)
    if not audited and (insight is not None or resp.warnings):
        with st.expander("ℹ️ Doğrulama notları"):
            render_retrieval_facts(resp, insight)


def process_submission(tenant: str, ws: LocalWorkspace, question: str, departments: list[str], *,
                       on_recorded, animate: bool = True) -> None:  # noqa: ANN001
    """Yanıtı token token akıtır; akış bitince Denetçi çalışır, sonuç **çizimden önce** tam bir kez kaydedilir,
    ardından rozet, güven ve kaynak kartları akışın altına eklenir."""
    ss = st.session_state
    msgs = messages(tenant)
    history = [(m["role"], m["content"]) for m in msgs]
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🧠"):
        auditing = bool(ss.auditor_on)
        status = st.empty()
        # "Bunu mu sormak istediniz?": ana aramadan önce hızlı modelle soru denetimi (hata → öneri yok)
        status.markdown('<div class="meta">🧭 Soru kontrol ediliyor…</div>', unsafe_allow_html=True)
        suggestion = ws.suggest(question)
        render_suggestion(suggestion, key=str(len(msgs) + 1))  # kayıttan sonra yanıtın sohbet indeksi
        text_slot = st.empty()
        status.markdown('<div class="meta">🔎 Kaynaklar aranıyor…</div>', unsafe_allow_html=True)
        pending = ws.start(question, history, departments, audit=auditing)
        delay = ss.stream_delay if animate else 0.0
        shown = ""
        for chunk in pending.tokens:
            if not shown:
                status.markdown('<div class="meta">✍️ Araştırmacı yanıtı yazıyor…</div>', unsafe_allow_html=True)
            # Model parçaları zaten akar; tek parça gelen (çevrimdışı) yanıtta yazma efekti için kelimelere böl
            pieces = chunk.split(" ") if delay > 0 and len(chunk) > 40 else [chunk]
            for i, piece in enumerate(pieces):
                shown += piece + (" " if i < len(pieces) - 1 else "")
                text_slot.markdown(streaming_view(shown) + " ▌")
                if delay > 0 and len(pieces) > 1:
                    time.sleep(delay)
        if auditing:
            status.markdown('<div class="meta">🛡️ Denetçi her iddiayı kaynaklarla karşılaştırıyor…</div>',
                            unsafe_allow_html=True)
        turn = pending.finish()
        # Kayıt: finish() ile bu satırlar arasında Streamlit çağrısı yok → rerun burayı bölemez
        msgs.append({"role": "user", "content": question})
        msgs.append({"role": "assistant", "content": turn.audit.answer, "question": question,
                     "response": turn.response, "audit": turn.audit, "insight": turn.insight,
                     "suggestion": suggestion})
        on_recorded()
        status.empty()
        render_answer(msgs[-1], key=str(len(msgs) - 1), text_slot=text_slot)


# ====================================================================== sekmeler
def tab_assistant(tenant: str, ws: LocalWorkspace, departments: list[str], kpis: KpiSlot) -> None:
    ss = st.session_state
    msgs = messages(tenant)
    waiting = [s for s in ss.inbox if s["tenant"] == tenant]
    if not msgs and not waiting and not ss.demo_queue:
        st.button("▶ Canlı demoyu başlat", key="demo_main", type="primary", on_click=_start_demo,
                  help="Hazır soruları sırayla sorar: kaynaklı yanıtlar, sözleşme tablosu ve uydurma engeli.")
        st.markdown('<div class="section-title">💡 Hızlı başlangıç — örnek sorular</div>', unsafe_allow_html=True)
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, (col, label, q) in enumerate(zip(cols, SAMPLE_LABELS, SAMPLE_QUESTIONS, strict=True)):
            col.button(label, key=f"sample_{i}", width="stretch", help=q, on_click=_submit, args=(q,))
        if not ws.documents:
            st.info("Bu tenant'ın bilgi tabanı boş. Kenar çubuğundan belge yükleyin veya "
                    f"**{SAMPLE_TENANT}** tenant'ını seçin.", icon="📭")

    for i, m in enumerate(msgs):
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🧠"):
            if m["role"] == "assistant" and m.get("response") is not None:
                render_answer(m, key=str(i), latest=i == len(msgs) - 1 and not waiting)
            else:
                st.markdown(m["content"])

    # 1) Kullanıcı gönderimleri (buton / sohbet kutusu / filtresiz yeniden sor): sırayla, her biri bir kez
    for sub in waiting:
        process_submission(tenant, ws, sub["text"],
                           departments if sub["departments"] is None else list(sub["departments"]),
                           on_recorded=lambda sub=sub: ss.inbox.remove(sub))

    # 2) Canlı demo: çalıştırma başına bir soru (animasyonlu), ardından rerun
    if ss.demo_queue and tenant == SAMPLE_TENANT:
        question = ss.demo_queue[0]
        st.toast(f"🎬 Demo: {question}")
        process_submission(tenant, ws, question, [], on_recorded=lambda: ss.demo_queue.pop(0))
        kpis.render()  # demo adımları rerun ile bittiği için KPI burada güncellenir
        if ss.demo_queue:
            time.sleep(ss.demo_delay)
        else:
            st.toast("🎬 Demo tamamlandı")
        st.rerun()

    scope = ALL_DEPARTMENTS if not departments else f"{', '.join(departments)} (filtre aktif)"
    st.caption(f"🔎 Arama kapsamı: {scope}")
    st.chat_input("Kurumsal dokümanlarınıza bir soru sorun…", key=CHAT_KEY, on_submit=_on_chat_submit)


def tab_analytics(ws: LocalWorkspace) -> None:
    st.markdown(theme.SIMULATED_BANNER, unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        render_chart(charts.usage_figure(charts.sample_usage(), session_queries=len(ws.history)))
    with c2:
        render_chart(charts.department_figure(charts.department_counts(ws.documents)))
    points = charts.quality_points(ws.history)
    if points:
        render_chart(charts.quality_figure(points))
    else:
        st.caption("Soru sordukça bu oturumdaki her sorgunun güven skoru ve yanıt süresi burada görünecek.")
    if not charts.PLOTLY:
        st.caption("Plotly kurulu değil; grafikler yedek çizimle gösteriliyor. `pip install plotly` ile etkileşimli grafikleri açın.")


def tab_library(ws: LocalWorkspace) -> None:
    docs = ws.documents
    if not docs:
        st.info("Henüz belge yok. Kenar çubuğundan PDF/DOCX/XLSX/TXT/MD yükleyebilirsiniz.", icon="📭")
        return
    for d in docs:
        c1, c2 = st.columns([12, 1])
        c1.markdown(theme.doc_row(d.filename, d.department, d.page_count, d.chunk_count, d.file_type),
                    unsafe_allow_html=True)
        c2.button("🗑️", key=f"del_{d.doc_id}", help=f"{d.filename} belgesini indeksten çıkar",
                  on_click=_remove_document, args=(ws, d.doc_id))


# ====================================================================== ana akış
def main() -> None:
    _state()
    st.markdown(theme.CSS, unsafe_allow_html=True)
    tenant, departments = sidebar()
    ws = workspace(tenant)

    st.markdown(theme.hero(
        "Kurumsal bilgiye saniyeler içinde, kaynağıyla ulaşın",
        "Politika, prosedür ve sözleşmelerinizi yükleyin; çalışanlarınız doğal dilde sorsun. Her yanıt, "
        "bilginin alındığı dokümanın ve sayfanın atfıyla gelir — dokümanda olmayan bilgi asla uydurulmaz.",
        [("Hibrit arama · Vektör + BM25 + RRF", ""), ("Sayfa düzeyinde kaynak atfı", "cyan"),
         ("Halüsinasyon koruması", "green"), ("Çok kiracılı izolasyon", "")],
    ), unsafe_allow_html=True)
    kpis = KpiSlot(tenant, ws)  # yeri burada; içeriği sorgular işlendikten sonra
    st.write("")

    t1, t2, t3 = st.tabs(["💬 Asistan", "📈 Analitik", "📚 Doküman Kütüphanesi"])
    with t1:
        tab_assistant(tenant, ws, departments, kpis)
    kpis.render()
    with t2:
        tab_analytics(ws)
    with t3:
        tab_library(ws)


main()
