import uuid

import pytest
from rank_bm25 import BM25Okapi

from src.config import Settings
from src.document_loader import process_document, process_file
from src.embeddings import HashingEmbedder
from src.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    LexicalReranker,
    LLMReranker,
    RetrievedChunk,
)
from src.llm import LLMError
from src.text_utils import tokenize
from src.vector_store import IN_MEMORY, SearchResult, VectorStore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(CHROMA_PERSIST_DIR=tmp_path / "chroma", QUERY_REWRITING=False, _env_file=None)


@pytest.fixture
def store(settings) -> VectorStore:
    s = VectorStore(HashingEmbedder(), settings, persist_dir=IN_MEMORY, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    s.add_document(process_file(settings.sample_docs_dir / "IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları", settings))
    s.add_document(process_file(settings.sample_docs_dir / "Finans_Harcama_Proseduru.md", "Finans", settings))
    return s


@pytest.fixture
def retriever(store, settings) -> HybridRetriever:
    return HybridRetriever(store, LexicalReranker(), settings)


# ------------------------------------------------------------------ tokenizasyon
def test_tokenize_turkish_folding_numbers_and_stopwords():
    assert tokenize("Çalışma SÜRESİ nedir?") == tokenize("calisma suresi")
    assert tokenize("İZİN") == tokenize("izin")
    assert "4500" in tokenize("Otel limiti 4.500 TL")
    assert "1000000" in tokenize("1.000.000 TL üzeri")
    assert "ve" not in tokenize("izin ve mesai")
    assert tokenize("izinleri") == tokenize("izinlerin")  # F5 kök


# ------------------------------------------------------------------ BM25
def _sr(i: str, text: str) -> SearchResult:
    return SearchResult(id=i, text=text, metadata={"source": "x.md", "page": 1}, score=0.0)


def test_bm25plus_scores_small_corpus_where_okapi_fails():
    docs = ["yıllık izin süresi on dört gün", "masraf beyanı otuz gün içinde"]
    # BM25Okapi: 2 belgelik korpusta tek belgede geçen terimin IDF'i 0 → hiçbir sinyal yok
    okapi = BM25Okapi([tokenize(d) for d in docs])
    assert max(okapi.get_scores(tokenize("masraf beyanı"))) == pytest.approx(0.0)

    hits = BM25Index([_sr("a", docs[0]), _sr("b", docs[1])]).search("masraf beyanı", top_n=5)
    assert [h[0].id for h in hits] == ["b"] and hits[0][1] > 0


def test_bm25_exact_identifier_match(retriever):
    hits = retriever.bm25_search("FİN-PRS-011", None, 5)
    assert hits[0][0].source == "Finans_Harcama_Proseduru.md"
    assert hits[0][0].page == 1


# ------------------------------------------------------------------ füzyon
def test_rrf_fusion_marks_both_arms_and_uses_formula(retriever, settings):
    fused = retriever.fuse("yıllık izin süreleri kıdem", ["İnsan Kaynakları"])
    top = fused[0]
    assert top.matched_by == {"vector", "bm25"}
    # Her iki kolda da 1. sıradaysa skor = 2 / (k + 1)
    max_possible = (settings.vector_weight + settings.bm25_weight) / (settings.rrf_k + 1)
    assert 0 < top.fused_score <= max_possible + 1e-9
    assert fused == sorted(fused, key=lambda c: c.fused_score, reverse=True)


def test_query_variants_contribute_to_fusion(retriever):
    base = {c.id: c.fused_score for c in retriever.fuse("fazla mesai", ["İnsan Kaynakları"])}
    expanded = {
        c.id: c.fused_score
        for c in retriever.fuse("fazla mesai", ["İnsan Kaynakları"], extra_queries=["fazla çalışma ücreti 270 saat"])
    }
    assert sum(expanded.values()) > sum(base.values())


# ------------------------------------------------------------------ uçtan uca arama
def test_retrieve_respects_department_and_ranks_correct_chunk(retriever):
    hits = retriever.retrieve("250.000 TL üzerindeki harcamayı kim onaylar", ["Finans"], top_k=3)
    assert all(h.department == "Finans" for h in hits)
    assert "| 250.001 – 1.000.000 TL | CFO + Genel Müdür |" in hits[0].text
    assert hits[0].label == "Finans_Harcama_Proseduru.md - Sayfa 1"
    assert hits == sorted(hits, key=lambda h: h.rerank_score, reverse=True)
    assert set(retriever.last_timings) == {"search_ms", "rerank_ms"}


def test_bm25_cache_invalidated_after_new_document(retriever, store, settings):
    assert retriever.bm25_search("otopark", ["Genel"], 5) == []
    store.add_document(process_document("otopark.md", "Otopark ücreti aylık 500 TL'dir.".encode(), "Genel", settings))
    assert retriever.bm25_search("otopark", ["Genel"], 5)[0][0].source == "otopark.md"


# ------------------------------------------------------------------ reranker'lar
def _chunk(i: str, text: str, vec: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(id=i, text=text, metadata={"source": f"{i}.md", "page": 1}, vector_score=vec)


def test_lexical_reranker_idf_weighting_prefers_rare_terms():
    common = _chunk("common", "Belgeler 15 gün içinde yüklenmelidir.")
    rare = _chunk("rare", "Masraf beyanı yapılmalıdır.")
    idf = {"gun": 0.1, "icind": 0.1, "masra": 3.0, "beyan": 3.0, "yapil": 0.5}.get
    ranked = LexicalReranker().rerank("Masraf beyanı kaç gün içinde yapılmalı?", [common, rare], lambda t: idf(t, 1.0))
    assert ranked[0].id == "rare"


class _ScoringLLM:
    def __init__(self, scores=None, fail=False):
        self.scores, self.fail = scores or {}, fail

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        if self.fail:
            raise LLMError("kota aşıldı")
        return schema.model_validate({"scores": [{"index": i, "score": s} for i, s in self.scores.items()]})

    def stream(self, *a, **k):  # pragma: no cover
        yield ""


def test_llm_reranker_orders_by_model_scores():
    cands = [_chunk("a", "konu dışı metin"), _chunk("b", "ilgili cevap metni")]
    reranker = LLMReranker(_ScoringLLM({0: 1, 1: 9}))
    ranked = reranker.rerank("soru", cands)
    assert [c.id for c in ranked] == ["b", "a"]
    assert ranked[0].rerank_score > 0.7 and reranker.last_used == "llm"


def test_llm_reranker_falls_back_to_lexical_on_error():
    reranker = LLMReranker(_ScoringLLM(fail=True))
    ranked = reranker.rerank("masraf beyanı", [_chunk("a", "izin süresi"), _chunk("b", "masraf beyanı süresi")])
    assert ranked[0].id == "b" and reranker.last_used == "lexical"
