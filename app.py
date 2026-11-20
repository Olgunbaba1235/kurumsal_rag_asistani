"""Kurumsal RAG Asistanı — Streamlit arayüzü.

Çalıştırma:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `src/` düzenini kurulum gerektirmeden içe aktarılabilir yap.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.exceptions import RAGError  # noqa: E402
from rag_assistant.ingestion import SUPPORTED_EXTENSIONS  # noqa: E402
from rag_assistant.logging_config import setup_logging  # noqa: E402
from rag_assistant.pipeline import RAGPipeline, cited_indices  # noqa: E402
from rag_assistant.ui import components  # noqa: E402
from rag_assistant.ui.styles import CUSTOM_CSS  # noqa: E402

st.set_page_config(page_title="Kurumsal RAG Asistanı", page_icon="📚", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
setup_logging()

settings = get_settings()


# ------------------------------------------------------------------ kaynaklar
@st.cache_resource(show_spinner="Asistan hazırlanıyor…")
def get_pipeline(api_key: str) -> RAGPipeline:
    cfg = settings.model_copy(update={"google_api_key": SecretStr(api_key)})
    return RAGPipeline.from_settings(cfg)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)


init_state()

# ------------------------------------------------------------------ kenar çubuğu
with st.sidebar:
    st.markdown("### 🔐 Bağlantı")
    api_key = settings.api_key
    if not api_key:
        api_key = st.text_input(
            "Google API anahtarı",
            type="password",
            help="`.env` dosyasında GOOGLE_API_KEY tanımlarsanız bu alan gizlenir.",
        )
    else:
        st.success("API anahtarı `.env` üzerinden yüklendi.", icon="✅")

if not api_key:
    components.hero()
    st.info("Başlamak için kenar çubuğuna Google Gemini API anahtarınızı girin.", icon="🔑")
    st.stop()

try:
    pipeline = get_pipeline(api_key)
except RAGError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown("### 📤 Doküman Yükle")
    uploads = st.file_uploader(
        "PDF, DOCX veya TXT",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Dosya başına en fazla {settings.max_file_size_mb} MB.",
    )
    if uploads and st.button("İndeksle", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Başlıyor…")
        for i, f in enumerate(uploads, start=1):
            progress.progress((i - 1) / len(uploads), text=f"İşleniyor: {f.name}")
            try:
                res = pipeline.ingest(f.name, f.getvalue())
                if res.skipped:
                    st.toast(f"{res.filename}: zaten indekslenmiş", icon="ℹ️")
                else:
                    st.toast(f"{res.filename}: {res.message}", icon="✅")
            except RAGError as exc:
                st.error(f"{f.name}: {exc}")
        progress.progress(1.0, text="Tamamlandı")
        st.session_state.uploader_key += 1  # yükleyiciyi temizle
        st.rerun()

    documents = pipeline.store.list_documents()

    st.markdown("### 🗂️ Bilgi Tabanı")
    if not documents:
        st.caption("Henüz doküman yok.")
    selected_hashes: list[str] = []
    for doc in documents:
        c1, c2 = st.columns([0.82, 0.18], vertical_alignment="center")
        with c1:
            if st.checkbox(
                f"{doc.filename}",
                value=True,
                key=f"sel_{doc.file_hash}",
                help=f"{doc.chunk_count} parça · {doc.file_type.upper()}",
            ):
                selected_hashes.append(doc.file_hash)
        with c2:
            if st.button("🗑️", key=f"del_{doc.file_hash}", help="Dokümanı sil"):
                pipeline.delete_document(doc.file_hash)
                st.rerun()

    with st.expander("⚙️ Gelişmiş ayarlar"):
        top_k = st.slider("Getirilecek parça sayısı (top-k)", 1, 15, settings.top_k)
        temperature = st.slider("Yaratıcılık (temperature)", 0.0, 1.0, settings.llm_temperature, 0.05)
        if st.button("Sohbeti temizle", width="stretch"):
            st.session_state.messages = []
            st.rerun()

# ------------------------------------------------------------------ ana alan
components.hero()
components.stats(len(documents), pipeline.store.count(), settings.llm_model)

if not st.session_state.messages:
    st.markdown(
        "Örnek sorular: *“İzin politikası kaç gün yıllık izin tanıyor?”*, "
        "*“Satın alma onay süreci nasıl işliyor?”*"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📚"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            components.sources_panel(msg.get("sources", []), msg.get("cited", []))

placeholder = "Dokümanlarınız hakkında bir soru sorun…" if documents else "Önce bir doküman yükleyin"
if question := st.chat_input(placeholder, disabled=not documents):
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        try:
            if not selected_hashes:
                raise RAGError("Aramaya dahil edilecek en az bir doküman seçin.")
            with st.spinner("İlgili bölümler aranıyor…"):
                result = pipeline.ask_stream(
                    question,
                    history,
                    top_k=top_k,
                    temperature=temperature,
                    file_hashes=selected_hashes if len(selected_hashes) < len(documents) else None,
                )
            answer = st.write_stream(result.stream)
            answer = answer if isinstance(answer, str) else "".join(map(str, answer))
            cited = cited_indices(answer, len(result.sources))
            components.sources_panel(result.sources, cited)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": result.sources, "cited": cited}
            )
        except RAGError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()  # başarısız soruyu geçmişe yazma
