"""Async RAG orkestrasyonu: yeniden yazma → hibrit arama → katı bağlamlı üretim.

Çekirdek kurallar (`rewrite_query`, sistem talimatı, atıf doğrulama, güven skoru,
halüsinasyon kapısı) `src.rag_engine` ile ortaktır. Burada:

* Veritabanı oturumu **yalnızca arama süresince** açık tutulur; LLM üretimi
  (saniyeler sürebilir) sırasında havuzdan bağlantı tutulmaz.
* Senkron SDK çağrıları `asyncio.to_thread` ile event loop'u bloklamaz.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from app.core.config import ApiSettings
from app.core.db import Database
from app.services.retrieval_service import PgHybridRetriever
from app.services.tenant_service import TenantContext
from src.llm import LLM, LLMError, Message
from src.rag_engine import (
    GROUNDED_SYSTEM_PROMPT,
    RAGResponse,
    build_messages,
    finalize_response,
    rewrite_query,
)


class RAGService:
    def __init__(self, db: Database, settings: ApiSettings, retriever: PgHybridRetriever, llm: LLM | None):
        self.db = db
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    async def ask(
        self,
        tenant: TenantContext,
        question: str,
        departments: Sequence[str],
        history: Sequence[Message] = (),
        *,
        top_k: int | None = None,
    ) -> RAGResponse:
        s = self.settings
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        question = question.strip()

        # 1) Sohbet geçmişine duyarlı sorgu yeniden yazma
        t0 = time.perf_counter()
        rewriter = self.llm if s.query_rewriting else None
        rewritten = await asyncio.to_thread(rewrite_query, question, history, rewriter, max_turns=s.history_turns)
        timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 2) Hibrit arama (tenant + departman kapsamlı)
        t0 = time.perf_counter()
        if not departments:
            # Erişebildiği departman olmayan kullanıcı: arama yapılmaz, "bilgi yok" yanıtı döner
            chunks, search_timings, reranker_used = [], {}, "none"
        else:
            async with self.db.tenant_session(tenant.id, tenant.departments) as session:
                chunks, search_timings, reranker_used = await self.retriever.retrieve(
                    session,
                    tenant,
                    rewritten.standalone,
                    departments,
                    top_k=top_k or s.top_k,
                    extra_queries=rewritten.variants,
                )
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings.update(search_timings)

        top = chunks[0].rerank_score if chunks else 0.0
        ctx = {
            "question": question,
            "history": list(history),
            "departments": list(departments),
            "rewritten": rewritten,
            "chunks": chunks,
            "passed_gate": bool(chunks) and top >= s.min_relevance,
            "timings": timings,
            "t_start": t_start,
            "warnings": [],
        }

        # 3) Katı bağlamlı üretim (alaka kapısı geçildiyse)
        raw_answer = ""
        if ctx["passed_gate"] and self.llm is None:
            ctx["passed_gate"], ctx["no_llm"] = False, True
            ctx["warnings"].append("Dil modeli yapılandırılmamış (LLM_PROVIDER=none); yanıt üretilemedi.")
        elif ctx["passed_gate"]:
            t0 = time.perf_counter()
            messages = build_messages(ctx, s.history_turns)
            try:
                raw_answer = await asyncio.to_thread(
                    lambda: "".join(self.llm.stream(messages, system=GROUNDED_SYSTEM_PROMPT))
                )
            except LLMError as exc:
                ctx["llm_error"] = True
                ctx["warnings"].append(f"Yanıt üretilemedi: {exc}")
                raw_answer = "Yanıt üretilirken bir hata oluştu. Lütfen tekrar deneyin."
            timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return finalize_response(ctx, raw_answer, min_relevance=s.min_relevance, reranker=reranker_used)
