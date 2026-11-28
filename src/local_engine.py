"""Kurulum gerektirmeyen yerel (gömülü) RAG çalışma alanı — Streamlit vitrin uygulaması için.

* Veritabanı, Docker veya ChromaDB gerekmez: parçalar bellekte tutulur, vektör araması
  `HashingEmbedder` ile, sözcüksel arama ve reranking üretimdeki `HybridRetriever` ile yapılır.
* `GOOGLE_API_KEY` varsa yanıtları Gemini üretir; yoksa **çevrimdışı özetleyici** devreye girer:
  bağlamdan soruyla en ilgili cümleleri seçip `[Dosya - Sayfa X]` atıflarıyla birleştirir
  (yeni bilgi üretmez). Her iki durumda da atıf doğrulama ve güven skoru üretim motoruyla aynıdır.
* Her Streamlit oturumu kendi çalışma alanını kullanır; bir kullanıcının yüklediği belge
  başka bir oturumda görünmez.
* Bileşik sorular (ör. "X nasıl hesaplanır ve Y nedir?") alt sorulara ayrılır: dokümanda karşılığı olan kısım
  kaynaklı yanıtlanır, olmayan kısım için açık not düşülür (bkz. `src.compound`).
* Her yanıtla birlikte bir `AnswerInsight` döner: en iyi alaka skoru, eşik, alt soru sonuçları ve departman
  filtresi bir reddin nedeniyse bunu açıklayan not (filtre dışı belgelerde yapılan ön kontrolle).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.auditor import AuditResult, GeminiAuditor, RuleBasedAuditor, skipped
from src.compound import compose_response, merge_audits, split_compound
from src.config import Settings, get_settings
from src.document_loader import ProcessedDocument, process_document
from src.embeddings import HashingEmbedder
from src.hybrid_retriever import HybridRetriever, LexicalReranker
from src.llm import GeminiLLM, LLMError, Message
from src.rag_engine import RAGEngine, RAGResponse, citation_coverage, rewrite_query
from src.search_types import SearchResult
from src.suggestions import QuerySuggester, QuerySuggestion
from src.text_utils import fold, tokenize

logger = logging.getLogger(__name__)

OFFLINE_MODE = "Çevrimdışı özetleyici"
SHOWCASE_DEPARTMENTS = ("Hukuk",)
GEMINI_MODE = "Gemini"


# =============================================================== Bellek içi vektör deposu
@dataclass(slots=True)
class LocalDocument:
    doc_id: str
    filename: str
    department: str
    file_type: str
    page_count: int
    chunk_count: int
    added_at: float = field(default_factory=time.time)


class InMemoryVectorStore:
    """`HybridRetriever`'ın beklediği depo arayüzü: `search`, `get_chunks`, `revision`, `settings`."""

    def __init__(self, settings: Settings, dimension: int = 512):
        self.settings = settings
        self.embedder = HashingEmbedder(dimension=dimension)
        self.revision = 0
        self._chunks: list[SearchResult] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self.documents: dict[str, LocalDocument] = {}

    def add_document(self, doc: ProcessedDocument) -> bool:
        """Belgeyi indeksler. Aynı içerik zaten varsa False döner (tekrar eklenmez)."""
        if doc.doc_id in self.documents:
            return False
        texts = [c.text for c in doc.chunks]
        vectors = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
        self._chunks.extend(SearchResult(c.id, c.text, dict(c.metadata), 0.0) for c in doc.chunks)
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors
        self.documents[doc.doc_id] = LocalDocument(
            doc.doc_id, doc.filename, doc.department, doc.file_type, doc.page_count, len(doc.chunks)
        )
        self.revision += 1
        return True

    def remove_document(self, doc_id: str) -> None:
        if doc_id not in self.documents:
            return
        keep = [i for i, c in enumerate(self._chunks) if c.metadata.get("doc_id") != doc_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self._vectors.shape[1]), dtype=np.float32)
        del self.documents[doc_id]
        self.revision += 1

    def _mask(self, departments: Sequence[str] | None) -> list[int]:
        if not departments:
            return list(range(len(self._chunks)))
        wanted = set(departments)
        return [i for i, c in enumerate(self._chunks) if c.department in wanted]

    def get_chunks(self, departments: Sequence[str] | None = None) -> list[SearchResult]:
        return [self._chunks[i] for i in self._mask(departments)]

    def search(self, query: str, departments: Sequence[str] | None = None, top_k: int = 20) -> list[SearchResult]:
        idx = self._mask(departments)
        if not idx:
            return []
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        scores = self._vectors[idx] @ q
        order = np.argsort(-scores)[:top_k]
        out = []
        for j in order:
            c = self._chunks[idx[j]]
            out.append(SearchResult(c.id, c.text, c.metadata, float(max(0.0, scores[j]))))
        return out

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# =============================================================== Çevrimdışı "LLM"
_BLOCK_RE = re.compile(r'<parça etiket="\[([^\]]+)\]"[^>]*>\n(.*?)\n</parça>', re.DOTALL)
_QUESTION_RE = re.compile(r"\nSORU: (.+?)(?:\n|$)")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


_TITLE_RE = re.compile(r"^(?:Madde|Bölüm|Kısım|Ek)\s+\d+[\w.]*\s*[–—:-]", re.IGNORECASE)


class OfflineSummarizer:
    """API anahtarı olmadan çalışan, yalnızca bağlamdan cümle seçen yanıtlayıcı.

    `src.llm.LLM` protokolünü uygular. Yeni bilgi üretmez: bağlam parçalarındaki cümleleri,
    sorudaki terimlerle örtüşmesine göre seçer ve kaynak etiketiyle döndürür.
    """

    name = OFFLINE_MODE

    def __init__(self, max_sentences: int = 3, max_sources: int = 2):
        self.max_sentences = max_sentences
        self.max_sources = max_sources

    @staticmethod
    def _clean(sentence: str) -> str:
        s = re.sub(r"^[#>\-*\s]+", "", sentence).replace("**", "").strip()
        return s

    def answer(self, prompt: str) -> str:
        blocks = _BLOCK_RE.findall(prompt)
        m = _QUESTION_RE.search(prompt)
        terms = set(tokenize(m.group(1))) if m else set()
        if not blocks or not terms:
            return "Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır."

        scored: list[tuple[float, int, str, str]] = []
        for rank, (label, text) in enumerate(blocks[: self.max_sources]):
            for sent in _SENT_RE.split(text):
                clean = self._clean(sent).rstrip(":").strip()
                if len(clean) < 25 or (len(clean.split()) < 6 and not clean.startswith("|")):
                    continue  # başlık/kısa parça
                letters = [ch for ch in clean if ch.isalpha()]
                if _TITLE_RE.match(clean) or (letters and sum(ch.isupper() for ch in letters) > 0.8 * len(letters)):
                    continue  # "Madde 5 – Hizmet Kredisi ve Cezai Şart" gibi madde/bölüm başlıkları
                if clean.startswith("|"):  # tablo satırı: okunur hale getir
                    cells = [c.strip() for c in clean.strip("|").split("|") if c.strip() and set(c.strip()) != {"-"}]
                    if not cells or not any(ch.isdigit() for ch in clean):
                        continue  # boş ya da başlık satırı
                    clean = " · ".join(cells)
                overlap = len(terms & set(tokenize(clean)))
                if overlap:
                    scored.append((overlap - 0.15 * rank, -len(scored), clean, label))
        if not scored:
            label, text = blocks[0]
            first = next((self._clean(s) for s in _SENT_RE.split(text) if len(self._clean(s)) >= 25), "")
            return f"{first} [{label}]" if first else "Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır."

        best = sorted(scored, reverse=True)[: self.max_sentences]
        lines = [f"- {sent.rstrip('.')}. [{label}]" for _, _, sent, label in best]
        return "Dokümanlarda ilgili hükümler şöyledir:\n" + "\n".join(lines)

    def stream(self, messages: Sequence[Message], *, system: str | None = None,
               temperature: float | None = None) -> Iterator[str]:
        yield self.answer(messages[-1][1])

    def generate_json(self, prompt, schema, *, system=None, fast=True):  # noqa: ANN001, ANN201
        raise LLMError("Çevrimdışı modda yapısal çıktı yok")  # sorgu yeniden yazma sezgisel yönteme düşer


# =============================================================== Çalışma alanı
@dataclass(slots=True)
class QueryLog:
    question: str
    confidence: int
    answered: bool
    total_ms: float
    citations: int
    audit_status: str = "skipped"
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class PartOutcome:
    """Bileşik sorunun bir alt sorusu: yanıtlanıp yanıtlanmadığı ve en iyi alaka skoru (0..100)."""

    text: str
    answered: bool
    relevance: float


@dataclass(slots=True)
class AnswerInsight:
    """Yanıtın arama tarafı açıklaması (Denetçi Kontrol Raporu ve red nedeni için)."""

    best_relevance: float  # 0..100 — kullanılan kanıt parçalarının en yüksek alaka skoru
    threshold: float  # 0..100 — alaka kapısı (MIN_RELEVANCE)
    departments: list[str] = field(default_factory=list)  # aktif filtre ([] = tüm departmanlar)
    filter_note: str = ""  # filtre bir reddin/eksik kısmın olası nedeniyse açıklama
    filter_probe: str = ""  # filtre dışı belgelerde yapılan ön kontrolün sonucu
    outside_departments: list[str] = field(default_factory=list)
    parts: list[PartOutcome] = field(default_factory=list)
    empty_index: bool = False  # tenant'ın bilgi tabanında hiç belge yok
    confidence: int = 0  # kullanıcıya gösterilen, Denetçi kararını da içeren güven skoru (%)
    confidence_level: str = ""  # "Yüksek güvenilirlik" / "Orta güvenilirlik" / "Düşük güvenilirlik" / "Bilgi yok"

    @property
    def missing_parts(self) -> list[PartOutcome]:
        return [p for p in self.parts if not p.answered]


STRONG_RELEVANCE = 0.60  # bu alaka skoruna ulaşan kanıt "tam güçlü" sayılır
VERIFICATION_WEIGHT = {"approved": 1.0, "revised": 0.6, "skipped": 0.55}


def calibrated_confidence(response: RAGResponse, audit: AuditResult,
                          parts: Sequence[PartOutcome] = ()) -> tuple[int, str]:
    """Kullanıcıya gösterilen güven skoru (%) ve seviyesi.

    Araştırmacının ham skoru yalnızca arama/atıf sinyallerine bakar ve Gemini'nin kısa, her cümlesi atıflı
    yanıtlarında yapay olarak düşük kalır (%50 civarı). Burada Denetçi kararı da hesaba katılır:

    * kanıt gücü  (0.35): atıf yapılan en alakalı parçanın skoru / 0.60 (üstü tam puan)
    * atıf kapsamı (0.25): olgusal cümlelerin atıflı oranı
    * doğrulama    (0.40): Denetçi onayı 1.0 · düzeltme 0.6 · denetlenmedi 0.55

    Bileşik soruda yanıtlanmayan alt soru oranı kadar düşürülür; skor hiçbir zaman %97'yi aşmaz.
    """
    if response.error or not response.is_answered or audit.status == "rejected":
        return 0, "Bilgi yok"
    cited = [e for e in response.evidence if e.cited] or response.evidence
    evidence = min(1.0, max((e.relevance for e in cited), default=0.0) / 100 / STRONG_RELEVANCE)
    coverage = citation_coverage(audit.answer if audit.status != "skipped" else response.answer)
    score = 0.35 * evidence + 0.25 * coverage + 0.40 * VERIFICATION_WEIGHT.get(audit.status, 0.55)
    if parts:
        score *= 0.6 + 0.4 * sum(p.answered for p in parts) / len(parts)
    value = round(min(0.97, score) * 100)
    level = "Yüksek güvenilirlik" if value >= 85 else "Orta güvenilirlik" if value >= 65 else "Düşük güvenilirlik"
    return value, level


@dataclass(slots=True)
class Turn:
    """Tek bir kullanıcı sorusunun sonucu: Araştırmacı yanıtı, Denetçi kararı ve arama açıklaması."""

    response: RAGResponse
    audit: AuditResult
    insight: AnswerInsight


class PendingTurn:
    """Akışlı yanıt: önce `tokens` tüketilir (ekrana akar), sonra `finish()` Denetçi'yi çalıştırıp `Turn` döner.

    `finish()` oturum geçmişine tam bir kez kaydeder; tekrar çağrılırsa aynı sonucu döndürür.
    """

    def __init__(self, tokens: Iterator[str], finish: Callable[[], Turn]):
        self.tokens = tokens
        self._finish = finish
        self._turn: Turn | None = None

    def finish(self) -> Turn:
        if self._turn is None:
            for _ in self.tokens:  # akış yarıda bırakıldıysa tamamla
                pass
            self._turn = self._finish()
        return self._turn


class LocalWorkspace:
    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        base = settings or get_settings()
        # Lexical reranker ile kalibre edilmiş varsayılanlar; LLM reranker çevrimdışı çalışmaz
        # Vitrin: örnek sözleşme için "Hukuk" departmanı eklenir (API'nin varsayılan departmanları değişmez)
        departments = list(dict.fromkeys([*base.departments, *SHOWCASE_DEPARTMENTS]))
        self.settings = base.model_copy(update={"use_llm_reranker": False, "departments": departments})
        self.store = InMemoryVectorStore(self.settings)
        self.llm, self.mode = self._make_llm(api_key or self.settings.api_key)
        if self.mode == OFFLINE_MODE:
            self.settings = self.settings.model_copy(update={"query_rewriting": False})
        self.engine = RAGEngine(HybridRetriever(self.store, LexicalReranker(), self.settings), self.llm, self.settings)
        if self.mode == OFFLINE_MODE:
            # Çevrimdışı özetleyicide yapısal çıktı yok: öneriler yalnızca sezgisel (yazım + başlık niyetleri)
            self.engine.suggester = QuerySuggester(None, enabled=self.settings.query_suggestions)
        # Denetçi: Gemini modunda hızlı Flash modeliyle, aksi halde çevrimdışı kural tabanlı
        self.auditor = GeminiAuditor(self.llm) if self.mode == GEMINI_MODE else RuleBasedAuditor()
        self.history: list[QueryLog] = []

    @staticmethod
    def _make_llm(api_key: str | None):  # noqa: ANN205
        if api_key:
            try:
                s = get_settings()
                return GeminiLLM(api_key, s.llm_model, s.fast_llm_model, s.llm_temperature,
                                 timeout=s.llm_timeout_seconds,
                                 fallback_to_fast=s.llm_fallback_to_fast), GEMINI_MODE
            except Exception as exc:  # noqa: BLE001 — paket yok/anahtar geçersiz: demo yine çalışsın
                logger.warning("Gemini başlatılamadı, çevrimdışı moda geçiliyor: %s", exc)
        return OfflineSummarizer(), OFFLINE_MODE

    # ------------------------------------------------------------ belgeler
    def add_bytes(self, filename: str, data: bytes, department: str) -> tuple[ProcessedDocument, bool]:
        doc = process_document(filename, data, department, self.settings)
        return doc, self.store.add_document(doc)

    def sample_documents(self) -> list[ProcessedDocument]:
        """Örnek klasördeki belgeleri işler (önbelleğe alınabilir; bkz. app.py `_cached_samples`)."""
        docs: list[ProcessedDocument] = []
        mapping = {"ik": "İnsan Kaynakları", "fin": "Finans", "tek": "Teknik", "huk": "Hukuk"}
        for path in sorted(Path(self.settings.sample_docs_dir).glob("*.*")):
            if path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx", ".xlsx"}:
                continue
            prefix = fold(path.stem.split("_", 1)[0])
            dept = next((d for k, d in mapping.items() if prefix.startswith(k)), self.settings.default_department)
            if dept not in self.settings.departments:
                dept = self.settings.default_department
            docs.append(process_document(path.name, path.read_bytes(), dept, self.settings))
        return docs

    def load_sample_documents(self, processed: Sequence[ProcessedDocument] | None = None) -> int:
        return sum(self.store.add_document(doc) for doc in (processed if processed is not None
                                                             else self.sample_documents()))

    @property
    def documents(self) -> list[LocalDocument]:
        return sorted(self.store.documents.values(), key=lambda d: d.added_at)

    # ------------------------------------------------------------ soru-cevap
    def suggest(self, question: str, departments: Sequence[str] | None = None) -> QuerySuggestion:
        """"Bunu mu sormak istediniz?" önerisi; yalnızca `departments` kapsamındaki belgelerden ipucu alır.
        Ana aramayla paralel (ayrı iş parçacığında) çağrılabilir; hiçbir hata yükseltmez."""
        return self.engine.suggest(question, list(departments or []) or None)

    def start(self, question: str, history: Sequence[Message] = (),
              departments: Sequence[str] | None = None, *, audit: bool = True) -> PendingTurn:
        """Yanıtı akışlı başlatır. Tek parçalı ve alaka kapısını geçen soruda model çıktısı token token akar;
        bileşik/kurtarma/red yollarında yanıt hazırlanıp tek parça halinde verilir."""
        depts = list(departments or [])
        if len(split_compound(question)) < 2:
            answer = self.engine.ask_stream(question, history, depts or None)
            if answer.will_generate:
                def finish() -> Turn:
                    full = answer.response
                    assert full is not None
                    if full.is_answered or full.error:
                        return self._complete(question, full, None, [], depts, audit)
                    response, parts, outcomes = self._research(question, history, depts, full=full)
                    return self._complete(question, response, parts, outcomes, depts, audit)

                return PendingTurn(iter(answer), finish)

        state: dict = {}

        def tokens() -> Iterator[str]:
            state["research"] = self._research(question, history, depts)
            yield state["research"][0].answer

        return PendingTurn(tokens(), lambda: self._complete(question, *state["research"], depts, audit))

    def run(self, question: str, history: Sequence[Message] = (),
            departments: Sequence[str] | None = None, *, audit: bool = True) -> Turn:
        """Bir kullanıcı sorusunu uçtan uca yanıtlar ve oturum geçmişine **tam bir kez** kaydeder.

        Araştırmacı (RAG motoru) taslağı üretir; Denetçi yalnızca getirilen parçalarla doğrular. Bileşik
        sorularda her alt soru ayrı aranır ve ayrı denetlenir; sonuç tek yanıtta birleştirilir.
        """
        return self.start(question, history, departments, audit=audit).finish()

    def _complete(self, question: str, response: RAGResponse, parts, outcomes: list[PartOutcome],  # noqa: ANN001
                  depts: list[str], audit: bool) -> Turn:
        if response.error:  # servis hatası: denetlenecek yanıt yok; "bilgi yok" reddi gibi gösterilmez
            result = AuditResult("skipped", response.answer, "Yanıt üretilemediği için denetim yapılmadı.",
                                 "—", 0.0, [])
        elif parts is None:
            result = self.auditor.audit(response) if audit else skipped(response)
        elif audit:
            result = merge_audits(
                [(i, p, self.auditor.audit(r)) for i, p, r in parts],
                [o.text for o in outcomes if not o.answered],
                total_parts=len(outcomes), method=self.auditor.name,
            )
        else:
            result = skipped(response)
        insight = self._insight(question, response, result, depts, outcomes)
        insight.confidence, insight.confidence_level = calibrated_confidence(response, result, outcomes)
        # Oturum metriği kullanıcının gördüğüyle aynı: Denetçi'nin reddettiği yanıt "cevapsız, güven %0" sayılır
        answered = response.is_answered and result.status != "rejected" and not response.error
        self.history.append(QueryLog(question, insight.confidence, answered, response.total_ms,
                                     len(response.citations) if answered else 0, result.status))
        return Turn(response, result, insight)

    def ask(self, question: str, history: Sequence[Message] = (),
            departments: Sequence[str] | None = None) -> RAGResponse:
        return self.run(question, history, departments, audit=False).response

    def ask_audited(self, question: str, history: Sequence[Message] = (),
                    departments: Sequence[str] | None = None, *, audit: bool = True) -> tuple[RAGResponse, AuditResult]:
        turn = self.run(question, history, departments, audit=audit)
        return turn.response, turn.audit

    def reset_history(self) -> None:
        """Oturum metriklerini sıfırlar (sohbet temizlendiğinde / demo yeniden başladığında)."""
        self.history.clear()

    # ------------------------------------------------------------ iç adımlar
    def _probe(self, text: str, history: Sequence[Message], departments: Sequence[str] | None) -> float:
        """Yalnızca arama + reranking (LLM yok): alt sorunun alaka kapısını geçip geçmeyeceğini ölçer."""
        query = rewrite_query(text, history, None, max_turns=self.settings.history_turns).standalone
        chunks = self.engine.retriever.retrieve(query, departments or None, self.settings.top_k)
        return chunks[0].rerank_score if chunks else 0.0

    def _research(self, question: str, history: Sequence[Message], depts: list[str],  # noqa: ANN202
                  *, full: RAGResponse | None = None):
        """(yanıt, alt soru yanıtları | None, alt soru sonuçları). Alt soru yanıtı yoksa tek parça akışıdır.

        Bileşik soruda önce yalnızca arama yapılır (LLM yok): her alt soru alaka kapısını geçiyorsa soru
        bütün olarak tek seferde yanıtlanır; bir kısmı geçiyorsa yalnızca o alt sorular yanıtlanır ve
        geçmeyenler için not düşülür. Tek soru reddedilirse "kurtarma" ayrımıyla aynı işlem denenir.
        """
        t0 = time.perf_counter()
        scope = depts or None
        parts = split_compound(question)
        if len(parts) < 2:
            full = full or self.engine.ask(question, history, scope)
            if full.is_answered or full.error:
                return full, None, []
            parts = split_compound(question, relaxed=True)  # kurtarma: soru sözcüğü şartı olmadan dene
            if len(parts) < 2:
                return full, None, []

        threshold = self.settings.min_relevance
        probes = [(p, self._probe(p, history, depts)) for p in parts]
        passing = sum(score >= threshold for _, score in probes)
        if passing == 0 or (passing == len(parts) and full is None):
            full = full or self.engine.ask(question, history, scope)
            if passing == 0 or full.is_answered or full.error:
                return full, None, []  # hiçbiri yok → red; tamamı var → soru bütün olarak yanıtlandı

        answered: list[tuple[int, str, RAGResponse]] = []
        outcomes: list[PartOutcome] = []
        for i, (part, score) in enumerate(probes, start=1):
            if score >= threshold:
                r = self.engine.ask(part, history, scope)
                if r.error:
                    return r, None, []  # servis hatası: kısmi yanıt kurulmaz, tekrar deneme önerilir
                if r.is_answered:
                    answered.append((i, part, r))
                    best = max((e.relevance for e in r.evidence), default=0.0)
                    outcomes.append(PartOutcome(part, True, best))
                    continue
            outcomes.append(PartOutcome(part, False, round(score * 100, 1)))
        if not answered:
            return full or self.engine.ask(question, history, scope), None, []
        missing = [o.text for o in outcomes if not o.answered]
        composed = compose_response(question, answered, missing, total_parts=len(parts), departments=depts,
                                    total_ms=(time.perf_counter() - t0) * 1000)
        return composed, answered, outcomes

    def _outside_matches(self, texts: Sequence[str], depts: list[str]) -> tuple[list[str], float]:
        """Filtre dışındaki departmanlarda eşiği geçen içerik var mı? (departmanlar, en iyi alaka 0..100)"""
        hits: dict[str, float] = {}
        for text in texts:
            for c in self.engine.retriever.retrieve(text, None, self.settings.top_k):
                if c.rerank_score >= self.settings.min_relevance and c.department not in depts:
                    hits[c.department] = max(hits.get(c.department, 0.0), c.rerank_score)
        ordered = sorted(hits, key=lambda d: hits[d], reverse=True)
        return ordered, round(max(hits.values(), default=0.0) * 100, 1)

    def _insight(self, question: str, response: RAGResponse, result: AuditResult, depts: list[str],
                 outcomes: list[PartOutcome]) -> AnswerInsight:
        insight = AnswerInsight(
            best_relevance=max((e.relevance for e in response.evidence), default=0.0),
            threshold=round(self.settings.min_relevance * 100, 1),
            departments=depts,
            parts=outcomes,
            empty_index=self.store.chunk_count == 0,
        )
        refused = not response.is_answered or result.status == "rejected"
        missing = insight.missing_parts
        if depts and (refused or missing) and not response.error:
            texts = [question] if refused else [o.text for o in missing]
            outside, best = self._outside_matches(texts, depts)
            insight.outside_departments = outside
            insight.filter_note = (
                f"Aktif departman filtresi ({', '.join(depts)}) nedeniyle diğer belgeler taranmadı. "
                "Tüm belgelerde aramak için filtreyi kaldırın."
            )
            insight.filter_probe = (
                f"Ön kontrol: ilgili içerik filtre dışındaki {', '.join(outside)} belgelerinde bulunuyor "
                f"(en iyi alaka %{best:.0f})." if outside
                else "Ön kontrol: filtre dışındaki belgelerde de eşiği geçen içerik bulunamadı."
            )
        return insight
