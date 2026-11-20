"""Yeniden kullanılabilir Streamlit bileşenleri."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from rag_assistant.models import RetrievedChunk

_PREVIEW_CHARS = 420


def hero() -> None:
    st.markdown(
        """
        <div class="rag-hero">
          <h1>📚 Kurumsal RAG Asistanı</h1>
          <p>Şirket dokümanlarınızı yükleyin, sorularınıza kaynak referanslı yanıtlar alın.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stats(doc_count: int, chunk_count: int, model: str) -> None:
    st.markdown(
        f"""
        <div class="rag-stats">
          <div class="rag-stat"><div class="v">{doc_count}</div><div class="l">Doküman</div></div>
          <div class="rag-stat"><div class="v">{chunk_count}</div><div class="l">Parça</div></div>
          <div class="rag-stat"><div class="v" style="font-size:1rem;padding-top:.3rem">{html.escape(model)}</div>
               <div class="l">Model</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sources_panel(sources: Sequence[RetrievedChunk], cited: Sequence[int]) -> None:
    """Yanıtın altında kaynak kartlarını gösterir; atıf yapılanlar vurgulanır."""
    if not sources:
        return
    cited_set = set(cited)
    label = f"📎 Kaynaklar ({len(cited_set) or len(sources)} atıf / {len(sources)} getirilen)"
    with st.expander(label, expanded=False):
        for i, src in enumerate(sources, start=1):
            text = src.text if len(src.text) <= _PREVIEW_CHARS else src.text[:_PREVIEW_CHARS] + "…"
            badge = '<span class="rag-badge">atıf yapıldı</span>' if i in cited_set else ""
            st.markdown(
                f"""
                <div class="rag-source {'cited' if i in cited_set else ''}">
                  <div class="hdr">
                    <span>[{i}] {html.escape(src.citation_label)}{badge}</span>
                    <span class="score">benzerlik {src.score:.0%}</span>
                  </div>
                  <div class="body">{html.escape(text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
