import re
import uuid

import pytest

from src.config import Settings
from src.document_loader import process_file
from src.embeddings import HashingEmbedder
from src.hybrid_retriever import HybridRetriever, LexicalReranker, RetrievedChunk
from src.llm import LLMError
from src.rag_engine import (
    GROUNDED_SYSTEM_PROMPT,
    NO_INFO_ANSWER,
    RAGEngine,
    compute_confidence,
    extract_citations,
    is_refusal,
    rewrite_query,
)
from src.vector_store import IN_MEMORY, VectorStore

_LABEL_RE = re.compile(r'<parça etiket="\[([^\]]+)\]"')


class ScriptedLLM:
    """Test LLM'i: yanıtı bir fonksiyonla üretir, çağrıları kaydeder."""

    def __init__(self, answer_fn=None, json_response=None, json_error=False, stream_error=False):
        self.answer_fn = answer_fn or (lambda labels: f"Yıllık izin 14 iş günüdür [{labels[0]}].")
        self.json_response = json_response
        self.json_error = json_error
        self.stream_error = stream_error
        self.stream_calls: list[dict] = []
        self.json_calls: list[str] = []

    def stream(self, messages, *, system=None, temperature=None):
        self.stream_calls.append({"messages": list(messages), "system": system})
        labels = _LABEL_RE.findall(messages[-1][1])
        answer = self.answer_fn(labels)
        half = len(answer) // 2
        yield answer[:half]
        if self.stream_error:
            raise LLMError("bağlantı koptu")
        yield answer[half:]

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        self.json_calls.append(prompt)
        if self.json_error or self.json_response is None:
            raise LLMError("yapısal çıktı yok")
        return schema.model_validate(self.json_response)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(CHROMA_PERSIST_DIR=tmp_path / "chroma", QUERY_REWRITING=False, _env_file=None)


@pytest.fixture
def retriever(settings) -> HybridRetriever:
    store = VectorStore(HashingEmbedder(), settings, persist_dir=IN_MEMORY, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    store.add_document(process_file(settings.sample_docs_dir / "IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları", settings))
    store.add_document(process_file(settings.sample_docs_dir / "Finans_Harcama_Proseduru.md", "Finans", settings))
    return HybridRetriever(store, LexicalReranker(), settings)


def make_engine(retriever, settings, llm=None) -> RAGEngine:
    return RAGEngine(retriever, llm or ScriptedLLM(), settings)


# ================================================================== rewrite_query
def test_rewrite_heuristic_resolves_follow_up():
    history = [("user", "Yıllık izin süreleri nedir?"), ("assistant", "14 gün [..]")]
    r = rewrite_query("Peki 15 yılı geçince?", history)
    assert r.method == "heuristic"
    assert r.standalone.startswith("Yıllık izin süreleri nedir?")


def test_rewrite_keeps_independent_question():
    r = rewrite_query("Harcama onay limitleri nelerdir?", [("user", "Merhaba")])
    assert r.standalone == "Harcama onay limitleri nelerdir?" and r.method == "none"


def test_rewrite_with_llm_returns_standalone_and_variants():
    llm = ScriptedLLM(
        json_response={
            "standalone_query": "15 yılı aşan kıdemde yıllık izin kaç gündür?",
            "search_variants": ["yıllık ücretli izin 15 yıl kıdem", "26 iş günü izin", "fazla"],
        }
    )
    history = [("user", "Yıllık izin kaç gün?"), ("assistant", "14 iş günü.")]
    r = rewrite_query("Peki 15 yılı geçince?", history, llm)
    assert r.method == "llm"
    assert r.standalone == "15 yılı aşan kıdemde yıllık izin kaç gündür?"
    assert len(r.variants) == 2  # en fazla 2 varyant
    assert "Kullanıcı: Yıllık izin kaç gün?" in llm.json_calls[0]


def test_rewrite_falls_back_when_llm_fails():
    r = rewrite_query("Peki ya Finans?", [("user", "Onay süreci nasıl?")], ScriptedLLM(json_error=True))
    assert r.method == "heuristic"


# ================================================================== grounded yanıt
def test_grounded_answer_with_citations_confidence_and_timings(retriever, settings):
    llm = ScriptedLLM()
    r = make_engine(retriever, settings, llm).ask("Yıllık izin süreleri kıdeme göre kaç gün?", departments=["İnsan Kaynakları"])

    assert r.is_answered
    assert r.citations and r.citations[0].label.startswith("IK_Izin_ve_Calisma_Yonetmeligi.md - Sayfa ")
    assert r.citations[0].evidence  # kanıt metni dolu
    assert 0 < r.confidence <= 100
    assert any(e.cited for e in r.evidence)
    assert {"rewrite_ms", "retrieval_ms", "search_ms", "rerank_ms", "generation_ms", "total_ms"} <= r.timings_ms.keys()
    assert r.total_ms >= r.timings_ms["generation_ms"]
    # Sıkı talimat ve etiketli bağlam modele gitti
    call = llm.stream_calls[0]
    assert call["system"] == GROUNDED_SYSTEM_PROMPT and NO_INFO_ANSWER in call["system"]
    assert "<parça etiket=\"[IK_Izin_ve_Calisma_Yonetmeligi.md - Sayfa" in call["messages"][-1][1]


def test_gate_blocks_irrelevant_question_without_calling_llm(retriever, settings):
    llm = ScriptedLLM()
    r = make_engine(retriever, settings, llm).ask("Şirket otoparkının aylık ücreti ne kadar?")
    assert r.answer == NO_INFO_ANSWER
    assert not r.is_answered and r.confidence == 0 and r.citations == []
    assert llm.stream_calls == []
    assert any("model çağrılmadı" in w for w in r.warnings)


def test_department_isolation_refuses_out_of_scope_question(retriever, settings):
    llm = ScriptedLLM()
    r = make_engine(retriever, settings, llm).ask("Masraf beyanı kaç gün içinde yapılmalı?", departments=["İnsan Kaynakları"])
    assert r.answer == NO_INFO_ANSWER and llm.stream_calls == []
    assert all(e.department == "İnsan Kaynakları" for e in r.evidence)


def test_model_refusal_is_normalized(retriever, settings):
    llm = ScriptedLLM(answer_fn=lambda labels: "Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır")
    r = make_engine(retriever, settings, llm).ask("Yıllık izin kaç gün?", departments=["İnsan Kaynakları"])
    assert r.answer == NO_INFO_ANSWER and not r.is_answered and r.confidence == 0


def test_hallucinated_citation_is_removed_with_warning(retriever, settings):
    llm = ScriptedLLM(answer_fn=lambda labels: f"İzin 14 gündür [{labels[0]}]. Ek prim vardır [Uydurma.pdf - Sayfa 9].")
    r = make_engine(retriever, settings, llm).ask("Yıllık izin kaç gün?", departments=["İnsan Kaynakları"])
    assert "Uydurma.pdf" not in r.answer
    assert [c.label for c in r.citations] == [r.citations[0].label]
    assert any("Uydurma.pdf - Sayfa 9" in w for w in r.warnings)


def test_uncited_answer_gets_low_confidence_and_warning(retriever, settings):
    llm = ScriptedLLM(answer_fn=lambda labels: "Yıllık izin süresi kıdeme göre değişmektedir.")
    r = make_engine(retriever, settings, llm).ask("Yıllık izin kaç gün?", departments=["İnsan Kaynakları"])
    assert r.is_answered and r.confidence <= 30
    assert any("kaynak atfı içermiyor" in w for w in r.warnings)


def test_history_is_used_for_rewrite_and_passed_to_model(retriever, settings):
    llm = ScriptedLLM()
    history = [("user", "Harcama onay yetkileri nelerdir?"), ("assistant", "Tutara göre değişir.")]
    r = make_engine(retriever, settings, llm).ask("Peki ya 1.000.000 TL üzeri?", history, ["Finans"])
    assert r.rewritten_query.startswith("Harcama onay yetkileri nelerdir?")
    assert llm.stream_calls[0]["messages"][:2] == history
    assert r.evidence[0].source == "Finans_Harcama_Proseduru.md"


# ================================================================== akış
def test_streaming_yields_tokens_then_populates_response(retriever, settings):
    stream = make_engine(retriever, settings).ask_stream("Yıllık izin kaç gün?", departments=["İnsan Kaynakları"])
    assert stream.response is None
    text = "".join(stream)
    assert stream.response is not None and stream.response.answer == text.strip()


def test_stream_error_is_reported(retriever, settings):
    llm = ScriptedLLM(stream_error=True)
    r = make_engine(retriever, settings, llm).ask("Yıllık izin kaç gün?", departments=["İnsan Kaynakları"])
    assert any("yarıda kesildi" in w for w in r.warnings)


# ================================================================== yardımcılar
def _rc(source: str, page: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=f"{source}{page}", text="x", metadata={"source": source, "page": page}, rerank_score=score)


def test_extract_citations_tolerates_format_variations():
    chunks = [_rc("IK_Yonetmelik.md", 2, 0.9), _rc("Finans.xlsx", 1, 0.8)]
    answer = "A [ik_yonetmelik - sayfa 2]. B [Finans.xlsx - Sayfa 1; IK_Yonetmelik.md - Sayfa 2]. C [Yok.pdf - Sayfa 3]"
    valid, invalid = extract_citations(answer, chunks)
    assert valid == ["IK_Yonetmelik.md - Sayfa 2", "Finans.xlsx - Sayfa 1"]
    assert invalid == ["Yok.pdf - Sayfa 3"]


def test_confidence_scales_with_evidence_and_coverage():
    chunks = [_rc("a.md", 1, 0.9), _rc("b.md", 1, 0.3)]
    strong = compute_confidence("İzin 14 gündür [a.md - Sayfa 1]. Prim yoktur [a.md - Sayfa 1].", chunks, ["a.md - Sayfa 1"])
    weak = compute_confidence("İzin 14 gündür [b.md - Sayfa 1]. Ayrıca başka kurallar da var.", chunks, ["b.md - Sayfa 1"])
    none = compute_confidence("İzin 14 gündür.", chunks, [])
    assert strong > weak > 0 and none <= 30 and strong <= 100


def test_is_refusal_variants():
    assert is_refusal("yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır")
    assert not is_refusal("Yıllık izin 14 gündür.")


def test_invalid_department_raises(retriever, settings):
    with pytest.raises(ValueError):
        make_engine(retriever, settings).ask("izin", departments=["Pazarlama"])
