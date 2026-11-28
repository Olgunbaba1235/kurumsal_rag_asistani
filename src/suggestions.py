"""Kurumsal bağlama duyarlı sorgu önerisi ("Bunu mu sormak istediniz?").

Akış (`QuerySuggester.suggest`):

1. **Kapsam ipucu** (`CorpusHint`): yalnızca kullanıcının ERİŞEBİLDİĞİ parçalardan (tenant + izinli departmanlar;
   API'de ayrıca RLS) belge adları, departmanlar, bölüm başlıkları ve sözcük dağarcığı çıkarılır. İpucu, aramanın
   zaten kurduğu kapsamlı BM25 indeksinden türetilir ve o indeks nesnesinde önbelleklenir → kapsam dışındaki hiçbir
   belge, başlık veya terim ipucuna (dolayısıyla modele ve öneriye) giremez.
2. **Hızlı ön analiz (LLM'siz)**: sorudaki içerik sözcükleri dağarcıkta var mı, yakın yazımı (ör. "izn" → "izin")
   var mı, soru tek/iki terimlik ve soru kalıbı içermiyor mu ("izin", "masraf formu")? Net soru → model
   çağrılmadan "öneri yok" (ek gecikme ve maliyet yok).
3. **fast_model**: ipucu + ön analiz sinyalleriyle yazım hatasında tek düzeltilmiş soru, belirsiz soruda 2–3 niyet.
4. **Sezgisel yedek**: model yoksa (çevrimdışı) veya hata verirse yakın yazım düzeltmesi ve başlıklardan niyetler.
5. **Doğrulama**: özgün soruyla aynı (yalnızca harf/noktalama/Türkçe karakter farkı), boş, aşırı uzun, yinelenen ve
   kapsamda olmayan bir dosya adını anan öneriler atılır.

Sonuçlar (kapsam anahtarı, ipucu parmak izi, normalize soru) anahtarıyla LRU önbellekte tutulur; kapsam anahtarı
çağıranın verdiği tenant + departman kümesidir, yani daha geniş yetkili bir kullanıcının önbelleğe aldığı öneri dar
yetkili kullanıcıya asla dönmez. Bu modül hiçbir koşulda istisna yükseltmez: her hata "öneri yok"tur.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from src.llm import LLM, LLMError, StructuredOutput
from src.search_types import SearchResult
from src.text_utils import STOPWORDS, fold, tokenize, turkish_lower

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 300  # bundan uzun sorular zaten ayrıntılıdır → öneri yok
MAX_SUGGESTION_CHARS = 200
MAX_ALTERNATIVES = 3
MAX_DOCUMENTS = 30
MAX_TOPICS = 60
MAX_HINT_CHARS = 2000
FUZZY_CUTOFF = 0.8

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
# Soru sözcükleri: kesme işaretinden sonraki ek ayrı sözcük sayılmaz ("Mars'taki" → "mars")
_QUERY_WORD_RE = re.compile(r"([^\W\d_]+)(?:['’][^\W\d_]+)?", re.UNICODE)
_FILENAME_RE = re.compile(r"[\w.\-]+\.(?:pdf|docx?|xlsx?|txt|md|csv)\b", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_TOPIC_PREFIX_RE = re.compile(r"^\s*(?:madde\s*)?\d+(?:\.\d+)*[.)]?\s*[-–—:]?\s*", re.IGNORECASE)
# Soru kalıbı: bunlardan biri varsa kısa soru da "tam soru" sayılır (katlanmış biçim)
_QUESTION_CUES = frozenset(
    "ne neler nedir nelerdir nasil nasildir kac kaci hangi hangisi kim kime kimin nerede nereye neden niye "
    "mi mu midir mudur misin musun gerekir gerekiyor var yok ne zaman".split()
)
# Her belgede bulunan genel bölümler niyet önerisi olarak anlamsızdır (modele ipucu olarak yine verilir)
_GENERIC_TOPICS = frozenset({
    "amac", "kapsam", "amac ve kapsam", "tanimlar", "dayanak", "yururluk", "yurutme", "genel", "genel ilkeler",
    "genel hukumler", "giris", "ekler", "yaptirimlar", "degisiklikler",
})


# =============================================================== veri tipleri
@dataclass(frozen=True, slots=True)
class QuerySuggestion:
    """Öneri sonucu. JSON karşılığı `to_dict()` (``has_suggestion``, ``suggested_query`` + ``alternatives``).

    * ``kind="typo"``      → tek düzeltilmiş soru (``suggested_query``)
    * ``kind="ambiguous"`` → 2–3 tıklanabilir niyet (``alternatives``; ``suggested_query`` ilkidir)
    """

    has_suggestion: bool = False
    suggested_query: str = ""
    alternatives: tuple[str, ...] = ()
    kind: str = "none"
    method: str = field(default="none", compare=False)  # llm | heuristic | skipped | cache | disabled

    @property
    def options(self) -> list[str]:
        """Arayüzde gösterilecek tıklanabilir sorular (en az 1, en çok 3)."""
        if not self.has_suggestion:
            return []
        return list(self.alternatives) or [self.suggested_query]

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_suggestion": self.has_suggestion,
            "suggested_query": self.suggested_query,
            "alternatives": list(self.alternatives),
            "kind": self.kind,
        }


NO_SUGGESTION = QuerySuggestion()


@dataclass(frozen=True, slots=True, eq=False)
class CorpusHint:
    """Kullanıcının erişebildiği belgelerin özeti (modele bağlam ipucu + sezgisel analiz için)."""

    documents: tuple[str, ...]
    departments: tuple[str, ...]
    topics: tuple[str, ...]
    words: frozenset[str]  # katlanmış tam sözcükler
    prefixes: frozenset[str]  # sözcüklerin ≥4 harfli önekleri (çekimli/kısaltılmış yazımı tanımak için)
    stems: frozenset[str]  # tokenize kökleri
    surface: Mapping[str, str]  # katlanmış sözcük → belgedeki yazımı (Türkçe karakterli, küçük harf)
    topic_terms: Mapping[str, frozenset[str]]  # başlık → başlığın ve altındaki metnin kökleri
    topic_docs: Mapping[str, frozenset[str]]  # başlık → geçtiği belge adlarının kökleri ("ik", "izin"…)
    fingerprint: str
    chunk_count: int = 0
    topic_department: Mapping[str, str] = field(default_factory=dict)  # başlık → departman
    # departman → eşleşme anahtarları: kısaltma ("ik", "bt") ve ad sözcüklerinin kökleri ("insan", "kayna")
    department_terms: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.chunk_count == 0

    def render(self, max_chars: int = MAX_HINT_CHARS) -> str:
        lines = [
            "Belgeler: " + "; ".join(self.documents),
            "Departmanlar: " + ", ".join(self.departments),
            "Konu başlıkları: " + "; ".join(self.topics),
        ]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


EMPTY_HINT = CorpusHint((), (), (), frozenset(), frozenset(), frozenset(), {}, {}, {}, "empty", 0)


def _clean_topic(raw: str) -> str:
    topic = re.sub(r"\s+", " ", raw).strip()
    # Belge başlığındaki kurum öneki: "Örnek Holding A.Ş. — İzin ve Çalışma Yönetmeliği" → "İzin ve Çalışma …"
    topic = re.split(r"\s[—–]\s|\s-\s", topic)[-1]
    topic = _TOPIC_PREFIX_RE.sub("", topic).strip(" -–—:#")
    return topic if 2 < len(topic) <= 80 else ""


def build_corpus_hint(chunks: Sequence[SearchResult]) -> CorpusHint:
    """Parçalardan kapsam ipucu üretir. Yalnızca VERİLEN parçalar kullanılır; çağıran kapsamı belirler."""
    if not chunks:
        return EMPTY_HINT
    documents: dict[str, None] = {}
    departments: dict[str, None] = {}
    topics: dict[str, str] = {}  # katlanmış → görünen
    topic_terms: dict[str, set[str]] = {}
    topic_docs: dict[str, set[str]] = {}
    topic_department: dict[str, str] = {}
    surfaces: dict[str, Counter] = {}
    words: set[str] = set()
    stems: set[str] = set()
    digest = hashlib.sha1()

    def add_words(text: str) -> None:
        for raw in _WORD_RE.findall(turkish_lower(text)):
            folded = fold(raw)
            if len(folded) >= 2:
                words.add(folded)
                surfaces.setdefault(folded, Counter())[raw] += 1

    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8", "ignore"))
        if chunk.source:
            documents.setdefault(chunk.source, None)
            add_words(re.sub(r"[_\-.]+", " ", chunk.source))
        if chunk.department:
            departments.setdefault(chunk.department, None)
        add_words(chunk.text)
        chunk_stems = set(tokenize(chunk.text))
        stems.update(chunk_stems)
        headings = [str(chunk.metadata.get(k) or "") for k in ("section", "sheet_name")]
        headings += _MD_HEADING_RE.findall(chunk.text)  # parça içindeki alt başlıklar ("### 4.1 Teklif …")
        for raw_topic in headings:
            topic = _clean_topic(raw_topic)
            if not topic:
                continue
            folded = fold(topic)
            if folded not in topics and len(topics) < MAX_TOPICS:
                topics[folded] = topic
                add_words(topic)
            if folded in topics:
                topic_terms.setdefault(folded, set()).update(chunk_stems | set(tokenize(topic)))
                topic_docs.setdefault(folded, set()).update(tokenize(re.sub(r"[_\-.]+", " ", chunk.source)))
                if chunk.department:
                    topic_department.setdefault(folded, chunk.department)
    stems.update(tokenize(" ".join(topics.values())))
    surface = {w: c.most_common(1)[0][0] for w, c in surfaces.items()}
    prefixes = {w[:k] for w in words if len(w) >= 4 for k in range(4, len(w) + 1)}
    docs = tuple(list(documents)[:MAX_DOCUMENTS])
    digest.update(str(len(chunks)).encode())
    return CorpusHint(
        documents=docs,
        departments=tuple(departments),
        topics=tuple(topics.values()),
        words=frozenset(words),
        prefixes=frozenset(prefixes),
        stems=frozenset(stems),
        surface=surface,
        topic_terms={topics[k]: frozenset(v) for k, v in topic_terms.items()},
        topic_docs={topics[k]: frozenset(v) for k, v in topic_docs.items()},
        fingerprint=digest.hexdigest(),
        chunk_count=len(chunks),
        topic_department={topics[k]: v for k, v in topic_department.items()},
        department_terms={d: _department_keys(d) for d in departments},
    )


def _department_keys(department: str) -> frozenset[str]:
    """"İnsan Kaynakları" → {"ik", "insan", "kayna"}: kullanıcı departmanı kısaltma veya adla anabilir."""
    words = [fold(w) for w in _WORD_RE.findall(turkish_lower(department))]
    keys = set(tokenize(department)) | {w for w in words if len(w) >= 3}
    if len(words) >= 2:
        keys.add("".join(w[0] for w in words))
    return frozenset(keys)


_HINT_LOCK = threading.Lock()


def hint_from_index(index: Any) -> CorpusHint:
    """Kapsamlı BM25 indeksinden (``index.chunks``) ipucu üretir ve indeks nesnesinde önbellekler.

    İndeks zaten tenant + departman (+ API'de RLS) kapsamlıdır ve kapsam/sürüm anahtarıyla önbelleklenir; ipucu
    indeksle aynı ömre sahip olduğundan belge eklenince/silinince kendiliğinden yenilenir.
    """
    hint = getattr(index, "_corpus_hint", None)
    if hint is None:
        hint = build_corpus_hint(list(getattr(index, "chunks", []) or []))
        with _HINT_LOCK:
            try:
                index._corpus_hint = hint
            except AttributeError:  # pragma: no cover — slots'lu bir indeks: önbelleksiz devam
                pass
    return hint


# =============================================================== ön analiz (LLM'siz)
@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    content: tuple[str, ...]  # katlanmış içerik sözcükleri
    unknown: tuple[str, ...]  # dağarcıkta olmayanlar
    corrections: Mapping[str, str]  # yanlış yazılmış (katlanmış) → belgedeki yazım
    has_cue: bool
    short: bool

    @property
    def ambiguous(self) -> bool:
        """Tek/iki terimlik ve soru kalıbı olmayan sorgu ("izin", "masraf formu")."""
        return self.short and not self.has_cue

    @property
    def needs_review(self) -> bool:
        """Model çağrısına değer mi? Net, dağarcıkla örtüşen sorular atlanır."""
        return bool(self.corrections) or self.ambiguous

    @property
    def clean(self) -> bool:
        return not self.needs_review


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein (+ bitişik harf yer değiştirme) mesafesi; `limit`'i aşınca erken çıkar."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _is_known(word: str, hint: CorpusHint) -> bool:
    """Belgede geçiyor mu? Türkçe eklemeli olduğu için çekimli ("almada" ← "alma") ve kısaltılmış ("hesap" →
    "hesaplanır") yazımlar da bilinen sayılır."""
    if word in hint.words or any(s in hint.stems for s in tokenize(word)):
        return True
    if len(word) >= 4 and word in hint.prefixes:  # sözcüğün kendisi bir belge sözcüğünün öneki
        return True
    return any(word[:k] in hint.words for k in range(len(word) - 1, 2, -1))  # belge sözcüğü + ek


def _closest(word: str, hint: CorpusHint) -> str | None:
    """Belgedeki en yakın yazım (kısa sözcüklerde 1, uzunlarda 2 düzenleme). Eşitlikte uzunluğu yakın olan."""
    if len(word) < 3:
        return None
    limit = 2 if len(word) >= 8 else 1
    best: tuple[int, int, str] | None = None
    for cand in hint.words:
        if (len(cand) < 3 or cand[0] != word[0] or abs(len(cand) - len(word)) > limit
                or cand in STOPWORDS or cand in _QUESTION_CUES):
            continue
        if cand.startswith(word) or word.startswith(cand):  # kök ↔ çekimli biçim ("gün" / "günü"): hata değil
            continue
        # Kısa sözcükte tek harf değişimi başka gerçek bir sözcük üretir ("izin" → "için"): yalnızca eksik/fazla
        # harf veya yer değiştirmiş harf düzeltilir
        if len(word) <= 4 and len(cand) == len(word) and sorted(cand) != sorted(word):
            continue
        dist = _edit_distance(word, cand, limit)
        if dist <= limit:
            key = (dist, abs(len(cand) - len(word)), cand)
            best = key if best is None or key < best else best
    return best[2] if best else None


def analyze_query(question: str, hint: CorpusHint) -> QueryAnalysis:
    originals = [m.group(1) for m in _QUERY_WORD_RE.finditer(question)]
    words = [fold(turkish_lower(w)) for w in originals]
    content = tuple(w for w in words if len(w) >= 2 and w not in STOPWORDS and w not in _QUESTION_CUES)
    # Cümle içinde büyük harfle başlayan sözcük özel isimdir ("Mars", "Ankara"): yazım hatası sayılmaz
    proper = {fold(turkish_lower(o)) for i, o in enumerate(originals) if i > 0 and o[:1].isupper()}
    has_cue = "?" in question or any(w in _QUESTION_CUES for w in words)
    unknown: list[str] = []
    corrections: dict[str, str] = {}
    for w in content:
        if _is_known(w, hint):
            continue
        unknown.append(w)
        if w in proper:
            continue
        match = _closest(w, hint)
        if match and match != w:
            corrections[w] = hint.surface.get(match, match)
    return QueryAnalysis(content, tuple(unknown), corrections, has_cue, short=0 < len(content) <= 2)


def _apply_corrections(question: str, corrections: Mapping[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        word = m.group(0)
        fixed = corrections.get(fold(turkish_lower(word)))
        if not fixed:
            return word
        return fixed[:1].upper() + fixed[1:] if word[:1].isupper() else fixed

    return _WORD_RE.sub(repl, question)


def _topic_alternatives(terms: Iterable[str], hint: CorpusHint) -> list[str]:
    """Terimlerle örtüşen başlıklardan niyet soruları (başlıkta geçenler önce, sonra içerikte geçenler).

    Terimlerden biri bir departmanı anıyorsa ("ik prosedürü", "finans") niyetler o departmanın başlıklarıyla
    sınırlanır; kalan terim başlıklarla örtüşmüyorsa departmanın ana başlıkları belge sırasıyla önerilir."""
    terms = [t for t in terms if t]
    keys = {fold(t) for t in terms} | {s for t in terms for s in tokenize(t)}
    named = {d for d, dk in hint.department_terms.items() if dk & keys}
    dept_keys = set().union(*(hint.department_terms[d] for d in named)) if named else set()
    stems = {s for t in terms if fold(t) not in dept_keys for s in tokenize(t)} - dept_keys
    if not stems and not named:
        return []
    scored = []
    for order, topic in enumerate(hint.topics):
        if fold(topic) in _GENERIC_TOPICS:
            continue
        if named and hint.topic_department.get(topic) not in named:
            continue
        if not stems:  # yalnızca departman anıldı → belge sırası
            scored.append((0, 0, order, topic))
            continue
        title = stems & set(tokenize(topic))
        body = stems & hint.topic_terms.get(topic, frozenset())
        docs = stems & hint.topic_docs.get(topic, frozenset())
        covered = title | body | docs
        if covered or named:
            # Önce sorudaki terimlerden kaçını kapsadığı, sonra nerede geçtiği (başlık > belge adı > metin)
            scored.append((-len(covered), -(3 * len(title) + 2 * len(docs) + len(body)), order, topic))
    scored.sort()
    return [_intent_question(topic) for *_, topic in scored]


def _intent_question(topic: str) -> str:
    """Başlıktan doğal bir niyet sorusu: "Teklif Alma Kuralları" → "Teklif Alma Kuralları nelerdir?"."""
    folded = fold(topic)
    if re.search(r"\b(kural|sure|limit|yetki|takvim|gereklilik|ilke)", folded):
        return f"{topic} nelerdir?"
    return f"{topic} ile ilgili kurallar nelerdir?"


def heuristic_suggestion(question: str, hint: CorpusHint, analysis: QueryAnalysis) -> QuerySuggestion:
    corrected = _apply_corrections(question, analysis.corrections) if analysis.corrections else question
    if analysis.ambiguous:
        terms = [analysis.corrections.get(w, w) for w in analysis.content]
        options = _finalize_options(_topic_alternatives(terms, hint), question, hint)
        if len(options) >= 2:
            return QuerySuggestion(True, options[0], tuple(options), "ambiguous", "heuristic")
        if options:
            return QuerySuggestion(True, options[0], (), "typo", "heuristic")
    if corrected != question:
        options = _finalize_options([corrected], question, hint)
        if options:
            return QuerySuggestion(True, options[0], (), "typo", "heuristic")
    return QuerySuggestion(method="heuristic")


# =============================================================== doğrulama
def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", fold(text)).strip()


def same_question(a: str, b: str) -> bool:
    """Yalnızca büyük/küçük harf, noktalama veya Türkçe karakter farkı → aynı soru (öneri gereksiz)."""
    return _norm(a) == _norm(b)


def _finalize_options(candidates: Iterable[str], question: str, hint: CorpusHint) -> list[str]:
    allowed = {fold(d) for d in hint.documents}
    out: list[str] = []
    seen = {_norm(question)}
    for raw in candidates:
        option = re.sub(r"\s+", " ", raw or "").strip()
        key = _norm(option)
        if not option or len(option) > MAX_SUGGESTION_CHARS or not key or key in seen:
            continue
        # Derinlemesine savunma: kapsamda olmayan bir dosya adını anan öneri asla gösterilmez
        if any(fold(name) not in allowed for name in _FILENAME_RE.findall(option)):
            logger.warning("Kapsam dışı dosya adı içeren öneri atıldı.")
            continue
        seen.add(key)
        out.append(option)
        if len(out) == MAX_ALTERNATIVES:
            break
    return out


# =============================================================== LLM
class _QuerySuggestionOut(StructuredOutput):
    has_suggestion: bool = Field(
        description="Soruda yazım hatası, anlam bozukluğu, eksik ifade veya belirsizlik varsa true; soru zaten açık, "
        "net ve dilbilgisi açısından düzgünse false"
    )
    suggested_query: str = Field(description="Tek ve en olası düzeltilmiş soru; öneri yoksa boş metin")
    alternatives: list[str] = Field(
        description="Soru birden fazla konuya karşılık gelebiliyorsa en fazla 3 farklı niyet (her biri tam soru); "
        "aksi halde boş liste"
    )


_SUGGEST_SYSTEM = """Sen kurumsal bir doküman asistanında soru kalitesi denetleyicisisin. Soruyu YANITLAMA.
Görevin, kullanıcının sorusunu KURUM İÇİ BAĞLAM'daki belge adları, departmanlar ve konu başlıklarıyla birebir
örtüşen en doğru aramaya dönüştürmenin gerekip gerekmediğine karar vermektir.

- Yazım hatası, eksik ifade veya anlam bozukluğu varsa: has_suggestion=true; suggested_query alanına bağlamdaki
  kurumsal terimleri kullanan TEK, net, dilbilgisi açısından düzgün bir soru yaz; alternatives boş liste.
- Soru çok kısa/belirsizse (ör. yalnızca "izin" veya "fatura") ve bağlamda birden fazla ilgili konu varsa:
  has_suggestion=true; alternatives alanına bağlamdaki FARKLI konulara karşılık gelen 2-3 tam soru yaz;
  suggested_query bunların en olasısı olsun.
- Soru zaten açık, net ve dilbilgisi açısından düzgünse: has_suggestion=false, suggested_query="", alternatives=[].
  Büyük/küçük harf, noktalama veya Türkçe karakter eksikliği (ı/i, ş/s…) için öneri ÜRETME.
- YALNIZCA bağlamdaki belge, departman ve başlıklara dayan; bağlamda olmayan belge, departman, rakam, tarih veya
  varsayım EKLEME. Kullanıcının niyetini değiştirme. Kullanıcının dilini koru (varsayılan Türkçe)."""


def _prompt(question: str, hint: CorpusHint, analysis: QueryAnalysis) -> str:
    signals = []
    if analysis.corrections:
        signals.append("Belgelerde geçmeyen sözcükler (olası doğru yazım): " + ", ".join(
            f"{w} → {c}" for w, c in analysis.corrections.items()))
    if analysis.ambiguous:
        signals.append("Soru kısa ve soru kalıbı içermiyor; birden fazla niyete karşılık gelebilir.")
    return (
        f"KURUM İÇİ BAĞLAM (kullanıcının erişebildiği belgeler):\n{hint.render()}\n\n"
        f"ÖN ANALİZ:\n{chr(10).join(signals) or '-'}\n\n"
        f"KULLANICI SORUSU: {question}"
    )


def _from_llm(out: _QuerySuggestionOut, question: str, hint: CorpusHint) -> QuerySuggestion:
    if not out.has_suggestion:
        return QuerySuggestion(method="llm")
    options = _finalize_options([*out.alternatives, out.suggested_query] if out.alternatives
                                else [out.suggested_query], question, hint)
    if len(options) >= 2:
        return QuerySuggestion(True, options[0], tuple(options), "ambiguous", "llm")
    if options:
        return QuerySuggestion(True, options[0], (), "typo", "llm")
    return QuerySuggestion(method="llm")


# =============================================================== motor
class QuerySuggester:
    """İş parçacığı güvenli öneri motoru. Her kapsam (tenant + departmanlar) için ayrı önbellek anahtarı kullanır."""

    def __init__(self, llm: LLM | None, *, enabled: bool = True, cache_size: int = 512):
        self.llm = llm
        self.enabled = enabled
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, QuerySuggestion] = OrderedDict()
        self._lock = threading.Lock()

    def suggest(self, question: str, hint: CorpusHint, *, scope: Hashable = ()) -> QuerySuggestion:
        try:
            return self._suggest(question, hint, scope)
        except Exception:  # noqa: BLE001 — öneri isteğe bağlıdır; hiçbir hata aramayı/yanıtı engellememeli
            logger.exception("Soru önerisi üretilemedi; öneri gösterilmeyecek.")
            return NO_SUGGESTION

    def _suggest(self, question: str, hint: CorpusHint, scope: Hashable) -> QuerySuggestion:
        question = re.sub(r"\s+", " ", question or "").strip()
        if not self.enabled:
            return QuerySuggestion(method="disabled")
        if not question or len(question) > MAX_QUESTION_CHARS or hint.empty:
            return QuerySuggestion(method="skipped")
        analysis = analyze_query(question, hint)
        if analysis.clean:
            return QuerySuggestion(method="skipped")  # net soru: model çağrılmaz

        key = (scope, hint.fingerprint, _norm(question))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return QuerySuggestion(cached.has_suggestion, cached.suggested_query, cached.alternatives,
                                       cached.kind, "cache")

        result, cacheable = None, True
        if self.llm is not None:
            try:
                out = self.llm.generate_json(_prompt(question, hint, analysis), _QuerySuggestionOut,
                                             system=_SUGGEST_SYSTEM, fast=True)
                result = _from_llm(out, question, hint)
            except Exception as exc:  # noqa: BLE001 — LLMError, şema/JSON hatası, beklenmeyen SDK hatası
                level = logging.INFO if isinstance(exc, (LLMError, ValueError)) else logging.WARNING
                logger.log(level, "Soru önerisi modelden alınamadı (%s); sezgisel yönteme düşülüyor.", exc)
                cacheable = False  # geçici hata: bir sonraki soruda model tekrar denensin
        if result is None:
            result = heuristic_suggestion(question, hint, analysis)

        if cacheable:
            with self._lock:
                self._cache[key] = result
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def suggest_query(question: str, llm: LLM | None, hint: CorpusHint | None = None,
                  *, scope: Hashable = ()) -> QuerySuggestion:
    """Önbelleksiz tek seferlik öneri (yardımcı/test). Üretimde `QuerySuggester` örneği kullanılır."""
    return QuerySuggester(llm, cache_size=0).suggest(question, hint or EMPTY_HINT, scope=scope)
