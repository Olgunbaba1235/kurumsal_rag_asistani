"""Kurumsal Asistan — çok kiracılı B2B Streamlit arayüzü (FastAPI backend'ine bağlı).

Çalıştırma:  streamlit run ui/streamlit_app.py
Backend:     uvicorn app.main:create_app --factory --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEFAULT_DEPARTMENTS  # noqa: E402

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
FILE_TYPES = ["pdf", "docx", "xlsx", "txt", "md"]
INTENT_LABELS = {
    "DOCUMENT_ONLY": "📄 Yalnızca doküman",
    "SQL_ONLY": "📊 Yalnızca veri (SQL)",
    "HYBRID": "🔀 Hibrit (doküman + veri)",
}


# --------------------------------------------------------------------------- API istemcisi
class ApiError(Exception):
    pass


class ApiClient:
    """API anahtarıyla (``X-API-Key``) veya kullanıcı oturumuyla (``Authorization: Bearer``) çalışır."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 180.0, *, token: str | None = None):
        self.base = base_url.rstrip("/") + "/api/v1"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {"X-API-Key": api_key or ""}
        self.timeout = timeout

    @property
    def identity(self) -> str:
        """Sonuçları kimlik başına ayırmak için anahtar (ham değer ekranda gösterilmez)."""
        return self.headers.get("X-API-Key") or self.headers.get("Authorization", "")

    @staticmethod
    def login(base_url: str, email: str, password: str, company_slug: str, timeout: float = 30.0) -> dict:
        try:
            r = httpx.post(
                base_url.rstrip("/") + "/api/v1/auth/login",
                json={"email": email, "password": password, "company_slug": company_slug},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"Backend'e ulaşılamadı: {exc}") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(str(detail))
        return r.json()

    def logout(self) -> None:
        self._request("POST", "/auth/logout")

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = httpx.request(method, self.base + path, headers=self.headers, timeout=self.timeout, **kw)
        except httpx.HTTPError as exc:
            raise ApiError(f"Backend'e ulaşılamadı ({self.base}): {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"HTTP {r.status_code}: {detail}")
        return r.json() if r.content else None

    def agent_query(self, question: str, departments: list[str], history: list[dict], intent: str | None) -> dict:
        body: dict[str, Any] = {"question": question, "departments": departments or None, "history": history}
        if intent:
            body["intent"] = intent
        return self._request("POST", "/agent/query", json=body)

    def chat_query(self, question: str, departments: list[str], history: list[dict], *, suggest: bool = True) -> dict:
        """Doküman sohbeti (`/chat/query`): kaynaklı yanıt + "Bunu mu sormak istediniz?" önerisi.
        `suggest=False`: soru bir öneriden seçildi → API yeniden öneri üretmez (döngü yok)."""
        body = {"question": question, "departments": departments or None, "history": history, "suggest": suggest}
        return self._request("POST", "/chat/query", json=body)

    def analytics_query(self, question: str, departments: list[str], mode: str) -> dict:
        body = {"question": question, "departments": departments or None, "mode": mode}
        return self._request("POST", "/analytics/query", json=body)

    def list_documents(self, department: str | None = None) -> dict:
        params = {"limit": 200, **({"department": department} if department else {})}
        return self._request("GET", "/documents", params=params)

    def upload_document(self, name: str, data: bytes, department: str) -> dict:
        return self._request(
            "POST", "/documents/upload", params={"wait": "true"},
            files={"file": (name, data, "application/octet-stream")}, data={"department": department},
        )

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def demo_data(self) -> dict:
        return self._request("POST", "/analytics/demo-data")


# --------------------------------------------------------------------------- yardımcılar
def history_payload(messages: list[dict], limit: int = 10) -> list[dict]:
    """Sohbet geçmişini API'nin `ChatTurn` biçimine çevirir (yalnızca metin turları)."""
    turns = []
    for m in messages:
        if m["role"] == "user" or m.get("kind") == "chat":
            text = m["content"]
        else:
            text = (m.get("report") or {}).get("executive_summary", "")
        if text:
            turns.append({"role": m["role"], "content": text})
    return turns[-limit:]


def rows_to_df(columns: list[str], rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


MANUAL_TENANT = "__manual__"
LOGIN_TENANT = "__login__"


def load_tenant_profiles(raw: Any) -> dict[str, dict[str, Any]]:
    """`.streamlit/secrets.toml` içindeki `[tenants.<tenant_id>]` bölümlerini okur.

    Her profil: `name`, `api_key`, `departments` (opsiyonel). Anahtarı olmayan profil atlanır.
    Tenant sunucuda yine yalnızca API anahtarından çözülür; Tenant ID arayüzde seçim içindir."""
    profiles: dict[str, dict[str, Any]] = {}
    for tenant_id, cfg in dict(raw or {}).items():
        cfg = dict(cfg)
        key = str(cfg.get("api_key") or "").strip()
        if key:
            profiles[str(tenant_id)] = {
                "name": str(cfg.get("name") or tenant_id),
                "api_key": key,
                "departments": [str(d) for d in cfg.get("departments") or []],
            }
    return profiles


def tenant_messages(state: Any, tenant_id: str) -> list[dict]:
    """Sohbet geçmişi tenant başına ayrı tutulur: tenant değişince önceki şirketin mesajları
    ne ekranda görünür ne de diğer tenant'ın API çağrısına geçmiş olarak gönderilir."""
    chats = state.setdefault("chats", {})
    return chats.setdefault(tenant_id, [])


def render_chart_builder(df: pd.DataFrame, key: str) -> None:
    """Sonuç tablosu için kullanıcı seçimli, yakınlaştırılabilir (interaktif) grafik."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if df.empty or not numeric:
        return
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Grafik", ["Çubuk", "Çizgi", "Alan", "Nokta"], key=f"{key}_kind")
    x = c2.selectbox("X ekseni", list(df.columns), key=f"{key}_x")
    y = c3.selectbox("Y ekseni", numeric, index=len(numeric) - 1, key=f"{key}_y")
    color = c4.selectbox("Renk", ["(yok)", *[c for c in df.columns if c not in (x, y)]], key=f"{key}_c")
    enc: dict[str, Any] = {"x": alt.X(x, sort="-y" if kind == "Çubuk" else None), "y": y,
                           "tooltip": list(df.columns)}
    if color != "(yok)":
        enc["color"] = color
    mark = {"Çubuk": "mark_bar", "Çizgi": "mark_line", "Alan": "mark_area", "Nokta": "mark_circle"}[kind]
    chart = getattr(alt.Chart(df), mark)(**({"point": True} if kind == "Çizgi" else {}))
    st.altair_chart(chart.encode(**enc).interactive(), width="stretch")


def render_chart(chart: dict | None) -> None:
    if chart:
        try:
            st.vega_lite_chart(spec=chart, width="stretch")
        except Exception:  # noqa: BLE001 — grafik opsiyoneldir
            st.caption("Grafik çizilemedi.")


def render_agent_message(m: dict) -> None:
    rep = m["report"]
    st.markdown(f"#### {rep['title']}")
    st.markdown(rep["executive_summary"])
    if rep["key_findings"]:
        st.markdown("**Temel bulgular**\n" + "\n".join(f"- {x}" for x in rep["key_findings"]))
    if rep["recommendations"]:
        st.markdown("**Öneriler**\n" + "\n".join(f"- {x}" for x in rep["recommendations"]))
    for c in rep["caveats"]:
        st.warning(c, icon="⚠️")
    st.caption(
        f"Niyet: {INTENT_LABELS.get(m['intent'], m['intent'])} ({m['intent_method']}) · "
        f"güven %{m['confidence']} · sentez: {rep['synthesis_method']} · {m['timings_ms'].get('total_ms', 0):.0f} ms"
    )
    if m["rag_sources"]:
        with st.expander(f"📄 Doküman kaynakları ({len(m['rag_sources'])})"):
            for s in m["rag_sources"]:
                rel = f" · ilgi %{s['relevance']:.0f}" if s.get("relevance") is not None else ""
                st.markdown(f"- **{s['label']}** — {s.get('department') or '-'}{rel}")
    sql = m.get("sql")
    if sql:
        with st.expander("🗄️ SQL özeti" + ("" if sql["success"] else " (başarısız)")):
            st.code(sql["sql"] or "-", language="sql")
            st.write(sql["summary"] or sql.get("error") or "")
            st.caption(f"Tablolar: {', '.join(sql['tables']) or '-'} · {sql['row_count']} satır · {sql['attempts']} deneme")
            for i in sql["insights"]:
                st.markdown(f"- {i}")
            if m.get("data"):
                st.dataframe(rows_to_df(sql["columns"], m["data"]["rows"]), width="stretch")
                render_chart(m["data"].get("chart"))
    if m["warnings"]:
        with st.expander("Uyarılar"):
            for w in m["warnings"]:
                st.write(f"- {w}")


_MD_SPECIAL = str.maketrans({c: "\\" + c for c in "\\`*_{}[]()<>#+-.!|$~"})


def md_escape(text: str) -> str:
    """API/model metni markdown olarak yorumlanmasın (öneri kutusu)."""
    return (text or "").translate(_MD_SPECIAL)


def suggestion_options(suggestion: dict | None) -> list[str]:
    """`/chat/query` → `suggestion` alanından tıklanabilir sorular (en çok 3; yoksa boş)."""
    if not suggestion or not suggestion.get("has_suggestion"):
        return []
    options = [o for o in suggestion.get("alternatives") or [] if isinstance(o, str) and o.strip()]
    if not options and str(suggestion.get("suggested_query") or "").strip():
        options = [suggestion["suggested_query"]]
    return list(dict.fromkeys(o.strip() for o in options))[:3]


def queue_question(chat_key: str, text: str) -> None:
    """Öneri butonu callback'i: soru bir sonraki çalıştırmada "kabul edilmiş" olarak (öneri istemeden) sorulur."""
    st.session_state.setdefault("b2b_pending", {})[chat_key] = text


def render_suggestion(suggestion: dict | None, *, key: str, chat_key: str, latest: bool) -> None:
    """"Bunu mu sormak istediniz?" kutusu yanıtın üstünde. Butonlar yalnızca son mesajda gösterilir."""
    options = suggestion_options(suggestion)
    if not options:
        return
    if len(options) == 1:
        st.info(f"Bunu mu sormak istediniz: **{md_escape(options[0].rstrip(' ?'))}**?", icon="💡")
        if latest:
            st.button("🔁 Önerilen soruyla sor", key=f"b2b_suggest_{key}_0", on_click=queue_question,
                      args=(chat_key, options[0]), help=options[0])
        return
    listed = "" if latest else "  \n" + "  \n".join(f"• {md_escape(o)}" for o in options)
    st.info("**Bunu mu sormak istediniz?** Sorunuz birden fazla konuya karşılık gelebilir; birini seçin." + listed,
            icon="🧭")
    if latest:
        for i, (col, option) in enumerate(zip(st.columns(len(options)), options, strict=True)):
            col.button(option, key=f"b2b_suggest_{key}_{i}", on_click=queue_question, args=(chat_key, option),
                       width="stretch", help=option)


def render_chat_message(m: dict, *, key: str, chat_key: str, latest: bool) -> None:
    """`/chat/query` yanıtı: öneri kutusu + kaynaklı yanıt + güven/süre + kaynaklar."""
    render_suggestion(m.get("suggestion"), key=key, chat_key=chat_key, latest=latest)
    st.markdown(m["answer"])
    st.caption(f"📄 Doküman sohbeti · güven %{m.get('confidence', 0)} · {float(m.get('total_ms') or 0):.0f} ms")
    citations = m.get("citations") or []
    if citations:
        with st.expander(f"📄 Doküman kaynakları ({len(citations)})"):
            for c in citations:
                st.markdown(f"- **{c['label']}** — {c.get('department') or '-'} · ilgi %{float(c.get('relevance') or 0):.0f}")
    if m.get("warnings"):
        with st.expander("Uyarılar"):
            for w in m["warnings"]:
                st.write(f"- {w}")


def uses_document_chat(intent: str | None, can_analytics: bool) -> bool:
    """Yalnızca doküman niyeti (veya analitik yetkisi olmayan kullanıcıda otomatik niyet — API zaten
    DOCUMENT_ONLY'ye zorlar) `/chat/query` ile yanıtlanır; öneri bu uç noktadan gelir."""
    return intent == "DOCUMENT_ONLY" or (intent is None and not can_analytics)


# --------------------------------------------------------------------------- sekmeler
def tab_assistant(api: ApiClient, departments: list[str], messages: list[dict], *, chat_key: str = "default",
                  can_analytics: bool = True) -> None:
    c1, c2 = st.columns([3, 1])
    intent_label = c1.selectbox(
        "Niyet", ["Otomatik", *INTENT_LABELS.values()], help="Otomatik: sorunun niyeti LLM/sezgisel olarak belirlenir"
    )
    if c2.button("Sohbeti temizle", width="stretch"):
        messages.clear()
        st.rerun()
    intent = next((k for k, v in INTENT_LABELS.items() if v == intent_label), None)
    pending = st.session_state.setdefault("b2b_pending", {}).pop(chat_key, None)

    last = len(messages) - 1
    for i, m in enumerate(messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            elif m.get("kind") == "chat":
                render_chat_message(m, key=str(i), chat_key=chat_key, latest=i == last and pending is None)
            else:
                render_agent_message(m)

    typed = st.chat_input("Örn. 250.000 TL üzeri siparişler hangileri ve kim onaylar?", max_chars=2000)
    question, accepted = (pending, True) if pending else (typed, False)
    if question:
        hist = history_payload(messages)
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        document_chat = accepted or uses_document_chat(intent, can_analytics)
        with st.chat_message("assistant"):
            with st.spinner("Kaynaklar aranıyor…" if document_chat else "Ajanlar çalışıyor…"):
                try:
                    if document_chat:
                        res = api.chat_query(question, departments, hist, suggest=not accepted)
                    else:
                        res = api.agent_query(question, departments, hist, intent)
                except ApiError as exc:
                    st.error(str(exc))
                    messages.pop()
                    return
            if document_chat:
                msg = {"role": "assistant", "kind": "chat", "content": res["answer"], **res}
                render_chat_message(msg, key=str(len(messages)), chat_key=chat_key, latest=True)
            else:
                msg = {"role": "assistant", "content": res["report"]["executive_summary"], **res}
                render_agent_message(msg)
        messages.append(msg)


def tab_documents(api: ApiClient, departments: list[str], all_departments: list[str]) -> None:
    st.subheader("Belge yükle")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Dosyalar", type=FILE_TYPES, accept_multiple_files=True)
        dept = st.selectbox("Departman", departments or all_departments)
        if st.form_submit_button("Yükle ve indeksle", type="primary") and files:
            for f in files:
                try:
                    r = api.upload_document(f.name, f.getvalue(), dept)
                    (st.success if r["outcome"] in ("indexed", "accepted") else st.info)(f"{f.name}: {r['message']}")
                except ApiError as exc:
                    st.error(f"{f.name}: {exc}")
    st.subheader("Belgeler")
    filt = st.selectbox("Departman filtresi", ["Tümü", *all_departments], key="doc_filter")
    try:
        docs = api.list_documents(None if filt == "Tümü" else filt)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.caption(f"Toplam {docs['total']} belge")
    if docs["items"]:
        df = pd.DataFrame(docs["items"])[
            ["filename", "department", "status", "page_count", "chunk_count", "size_bytes", "created_at"]
        ]
        st.dataframe(df, width="stretch", hide_index=True)
        victim = st.selectbox("Silinecek belge", ["—", *[d["filename"] for d in docs["items"]]])
        if victim != "—" and st.button("Belgeyi sil"):
            doc = next(d for d in docs["items"] if d["filename"] == victim)
            try:
                api.delete_document(doc["id"])
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))


def tab_analytics(api: ApiClient, departments: list[str]) -> None:
    st.subheader("Doğal dille veri sorgulama")
    if st.button("Demo satış verisi oluştur"):
        try:
            st.success(f"Demo veri: {api.demo_data()}")
        except ApiError as exc:
            st.error(str(exc))
    q = st.text_input("Soru", placeholder="2026'da en çok ciro yapan 5 müşteri kim?")
    mode = st.radio("Kaynak", ["data", "auto", "both"], horizontal=True, format_func=lambda x: {
        "data": "Yalnızca veri", "auto": "Otomatik", "both": "Doküman + veri"}[x])
    # Sonuç tenant (API anahtarı) başına saklanır: grafik ayarı değiştirmek gibi yeniden çalıştırmalarda
    # tablo kaybolmaz ve başka tenant'a geçildiğinde önceki şirketin sonucu gösterilmez.
    results = st.session_state.setdefault("analytics_results", {})
    tenant_key = api.identity
    if st.button("Sorgula", type="primary") and q.strip():
        with st.spinner("Sorgu çalışıyor…"):
            try:
                results[tenant_key] = api.analytics_query(q, departments, mode)
            except ApiError as exc:
                st.error(str(exc))
                return
    r = results.get(tenant_key)
    if r:
        st.markdown(r["answer"])
        st.caption(f"Yönlendirme: {r['route']} ({r['route_method']}) · güven %{r['confidence']}")
        d = r.get("data")
        if d and d["success"]:
            df = rows_to_df(d["columns"], d["rows"])
            st.dataframe(df, width="stretch", hide_index=True)
            render_chart(d.get("chart"))
            with st.expander("🎛️ Grafiği özelleştir", expanded=not d.get("chart")):
                render_chart_builder(df, key="analytics_chart")
            st.download_button("⬇️ CSV indir", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="sorgu_sonucu.csv", mime="text/csv")
            with st.expander("Üretilen SQL"):
                st.code(d["sql"], language="sql")
                st.write(d["explanation"])
            for i in d["insights"]:
                st.markdown(f"- {i}")
        elif d:
            st.warning("Veri sorgusu başarısız oldu.")
        for w in r["warnings"]:
            st.caption(f"⚠️ {w}")


# --------------------------------------------------------------------------- ana akış
def main() -> None:
    st.set_page_config(page_title="Kurumsal Asistan", page_icon="🏢", layout="wide")
    try:
        profiles = load_tenant_profiles(st.secrets.get("tenants"))
        api_default = st.secrets.get("api_url", DEFAULT_API_URL)
    except Exception:  # noqa: BLE001 — secrets.toml yoksa
        profiles, api_default = {}, DEFAULT_API_URL
    with st.sidebar:
        st.title("🏢 Kurumsal Asistan")
        base = st.text_input("API adresi", api_default)
        labels = {MANUAL_TENANT: "✍️ Manuel giriş (API anahtarı)", LOGIN_TENANT: "🔐 Kullanıcı girişi"}
        tenant_id = st.selectbox(
            "Tenant ID", [*profiles, MANUAL_TENANT, LOGIN_TENANT],
            format_func=lambda t: labels.get(t) or f"{t} · {profiles[t]['name']}",
            help="Profiller `.streamlit/secrets.toml` → `[tenants.<id>]` bölümlerinden okunur.",
        )
        token = None
        can_analytics = True
        if tenant_id == LOGIN_TENANT:
            sess = st.session_state.get("user_session")
            if sess is None:
                with st.form("login"):
                    company = st.text_input("Şirket kısa adı", placeholder="ornek-holding")
                    email = st.text_input("E-posta")
                    password = st.text_input("Parola", type="password")
                    if st.form_submit_button("Giriş yap", type="primary"):
                        try:
                            st.session_state["user_session"] = ApiClient.login(base, email, password, company.strip())
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))
                key, tenant_departments = "", []
            else:
                me = sess["me"]
                token = sess["token"]
                tenant_id = me["company"]["slug"]
                tenant_departments = me["departments"]
                can_analytics = me["role"] == "admin"
                key = token
                role = {"admin": "Yönetici (admin)", "manager": "Müdür", "user": "Kullanıcı"}[me["role"]]
                st.success(f"{sess['full_name']} · {me['company']['name']}\n\n{role}{' · sahip' if me['is_owner'] else ''}")
                if st.button("Çıkış yap"):
                    try:
                        ApiClient(base, token=token).logout()
                    except ApiError:
                        pass
                    st.session_state.pop("user_session", None)
                    st.rerun()
        elif tenant_id == MANUAL_TENANT:
            tenant_id = st.text_input("Tenant ID (etiket)", value="manuel").strip() or "manuel"
            key = st.text_input("Tenant API anahtarı (X-API-Key)", type="password")
            tenant_departments = list(DEFAULT_DEPARTMENTS)
        else:
            key = profiles[tenant_id]["api_key"]
            tenant_departments = profiles[tenant_id]["departments"] or list(DEFAULT_DEPARTMENTS)
        departments = st.multiselect(
            "Departmanlar", tenant_departments, default=[], placeholder="Tümü",
            help="Boş bırakırsanız tenant'ın tüm departmanları aranır.",
            key=f"departments_{tenant_id}",
        )
        st.caption("Tenant yalnızca API anahtarından çözülür; veri erişimi şirketinizle sınırlıdır.")
    if not key:
        st.info("Başlamak için kenar çubuğundan giriş yapın veya şirketinizin API anahtarını girin.")
        return
    st.caption(f"Aktif tenant: **{tenant_id}** · departman: {', '.join(departments) or 'tümü'}")
    api = ApiClient(base, token=token) if token else ApiClient(base, key)
    t1, t2, t3 = st.tabs(["🤖 Akıllı Kurumsal Asistan", "📁 Doküman Yönetimi", "📊 Canlı Veri Analitiği"])
    with t1:
        chat_key = f"{tenant_id}:{api.identity[-6:]}"
        tab_assistant(api, departments, tenant_messages(st.session_state, chat_key), chat_key=chat_key,
                      can_analytics=can_analytics)
    with t2:
        tab_documents(api, departments, tenant_departments)
    with t3:
        if can_analytics:
            tab_analytics(api, departments)
        else:
            st.info("Satış analitiği yalnızca şirket sahibi ve yöneticilere açıktır.")


if __name__ == "__main__":
    main()
