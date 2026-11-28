"""Embedding / LLM / reranker sağlayıcılarının yapılandırmadan kurulması."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ApiSettings
from src.embeddings import Embedder, GeminiEmbedder, HashingEmbedder
from src.hybrid_retriever import LexicalReranker, LLMReranker, Reranker
from src.llm import LLM, GeminiLLM


@dataclass(slots=True)
class Providers:
    embedder: Embedder
    llm: LLM | None
    reranker: Reranker


def build_providers(settings: ApiSettings) -> Providers:
    if settings.embedding_provider == "gemini":
        embedder: Embedder = GeminiEmbedder(
            settings.api_key, settings.embedding_model, settings.embedding_dimension,
            timeout=settings.llm_timeout_seconds,
        )
    else:
        # Geliştirme/test: ağ gerektirmez, anlamsal değildir
        embedder = HashingEmbedder(dimension=settings.embedding_dimension)

    llm: LLM | None = None
    if settings.llm_provider == "gemini":
        llm = GeminiLLM(settings.api_key, settings.llm_model, settings.fast_llm_model, settings.llm_temperature,
                        timeout=settings.llm_timeout_seconds, fallback_to_fast=settings.llm_fallback_to_fast)

    reranker: Reranker = LLMReranker(llm) if (llm is not None and settings.use_llm_reranker) else LexicalReranker()
    return Providers(embedder=embedder, llm=llm, reranker=reranker)
