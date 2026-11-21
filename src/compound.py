"""Bileşik (çok parçalı) sorular: alt sorulara ayırma ve kısmi yanıtların birleştirilmesi.

Neden gerekli? Alaka kapısı, sorunun ayırt edici terimlerinin IDF ağırlıklı kapsamına bakar. "Fazla mesai
nasıl hesaplanır ve Mars ofisinin adresi nedir?" gibi bir soruda dokümanda hiç geçmeyen "Mars/ofis/adres"
terimleri en yüksek ağırlığı aldığından skor eşiğin altına düşer ve dokümanda karşılığı olan kısım da
reddedilir. Burada soru bağımsız alt sorulara ayrılır; her alt soru kendi alaka kapısından geçer.
Karşılığı olanlar kaynak atfıyla yanıtlanır, olmayanlar için yanıta açık bir not düşülür.

Ayırma kuralları bilinçli olarak temkinlidir:

* Soru işareti, noktalı virgül ve satır sonu cümleleri ayırır.
* "ve", "ayrıca", "bir de", virgül gibi bağlaçlar yalnızca her iki taraf da kendi soru sözcüğünü taşıyorsa
  ayırır ("Uzaktan çalışma kuralları ve çekirdek saatler nelerdir?" tek soru olarak kalır).
* ``relaxed=True`` (tüm soru reddedildiğinde kurtarma denemesi) soru sözcüğü şartını kaldırır; yine de her
  parçada en az iki içerik terimi bulunmalıdır.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.auditor import AuditResult
from src.rag_engine import NO_INFO_ANSWER, Citation, Evidence, RAGResponse
from src.text_utils import fold, tokenize

MAX_PARTS = 4
MIN_PART_TERMS = 2

# Soru sözcükleri ve soru ekleri (katlanmış biçim). `tokenize` bunları stopword saydığı için ayrıca aranır.
_QUESTION_WORDS = frozenset(
    """
    ne neler nedir nelerdir nasil nasildir kac kactir kaca kacta kacinci hangi hangisi hangileri nerede
    neresi nereye nereden nerededir kim kimdir kimler kime kimin neden nicin niye mi mu midir mudur misin
    musun miyim muyum miyiz muyuz misiniz musunuz
    """.split()
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=\?)\s+|\s*[;\n]+\s*")
_CONJ = r"(?:ve de|hem de|bir de|ayrıca|ayrica|peki|ve)"
_CLAUSE_RE = re.compile(rf"(\s*,\s*(?:{_CONJ}\s+)?|\s+{_CONJ}\s+)", re.IGNORECASE)
_MD_CTRL_RE = re.compile(r"[\[\]*_`#<>|]")


# ====================================================================== ayırma
def _terms(text: str) -> int:
    return len(tokenize(text))


def _has_question_word(text: str) -> bool:
    return text.rstrip().endswith("?") or any(w in _QUESTION_WORDS for w in _WORD_RE.findall(fold(text)))


def _closes(clause: str, relaxed: bool) -> bool:
    return _terms(clause) >= MIN_PART_TERMS and (relaxed or _has_question_word(clause))


def _split_clauses(sentence: str, relaxed: bool) -> list[str]:
    segments = _CLAUSE_RE.split(sentence)
    pieces, seps = segments[0::2], segments[1::2]
    out: list[str] = []
    buf, joiner = pieces[0], ""
    for sep, piece in zip(seps, pieces[1:], strict=True):
        if _closes(buf, relaxed):
            out.append(buf)
            buf, joiner = piece, sep
        else:
            buf = f"{buf}{sep}{piece}"
    if out and not _closes(buf, relaxed):
        out[-1] = f"{out[-1]}{joiner}{buf}"  # yüklemsiz kuyruk önceki parçaya aittir
    else:
        out.append(buf)
    return [p.strip(" ,") for p in out if p.strip(" ,")]


def split_compound(question: str, *, relaxed: bool = False) -> list[str]:
    """Soruyu bağımsız alt sorulara ayırır. Ayrılamıyorsa tek elemanlı liste döner."""
    original = " ".join(question.split())
    parts: list[str] = []
    for sentence in _SENTENCE_RE.split(question.strip()):
        sentence = " ".join(sentence.split())
        if sentence:
            parts.extend(_split_clauses(sentence, relaxed))

    merged: list[str] = []
    for p in parts:  # çok kısa parçalar ("Peki?") komşusuna eklenir
        if merged and _terms(p) < MIN_PART_TERMS:
            merged[-1] = f"{merged[-1]} {p}"
        elif merged and _terms(merged[-1]) < MIN_PART_TERMS:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    unique = list(dict.fromkeys(merged))
    if len(unique) < 2 or len(unique) > MAX_PARTS or any(_terms(p) < MIN_PART_TERMS for p in unique):
        return [original] if original else []
    return unique


# ====================================================================== birleştirme
def _plain(part: str) -> str:
    """Kullanıcı metnini Markdown/atıf sözdiziminden arındırır (not ve başlıklarda güvenle gösterilir)."""
    return _MD_CTRL_RE.sub("", part).strip(" ?.!") or part.strip()


def scope_note(part: str) -> str:
    """Dokümanda karşılığı olmayan alt soru için yanıta eklenen açık not."""
    return f"ℹ️ **Not:** “{_plain(part)}” kısmına ilişkin bilgi yüklenen dokümanlarda yer almamaktadır."


def part_heading(index: int, part: str) -> str:
    return f"**{index}) {_plain(part)}?**"


def compose_answer(answered: Sequence[tuple[int, str, str]], missing: Sequence[str]) -> str:
    """Alt soru yanıtlarını (sıra no, alt soru, metin) ve eksik kısım notlarını tek metinde birleştirir."""
    blocks: list[str] = []
    multi = len(answered) > 1
    for index, part, text in answered:
        blocks.append(f"{part_heading(index, part)}\n\n{text.strip()}" if multi else text.strip())
    blocks.extend(scope_note(p) for p in missing)
    return "\n\n".join(blocks)


def _unique_by_label(items: Sequence[Citation] | Sequence[Evidence]) -> list:
    seen: dict[str, object] = {}
    for item in items:
        seen.setdefault(item.label, item)
    return list(seen.values())


def compose_response(
    question: str,
    answered: Sequence[tuple[int, str, RAGResponse]],
    missing: Sequence[str],
    *,
    total_parts: int,
    departments: Sequence[str] | None,
    total_ms: float,
) -> RAGResponse:
    """Kısmi yanıtı tek bir `RAGResponse` olarak kurar (atıflar/kanıtlar alt yanıtlardan birleştirilir)."""
    responses = [r for _, _, r in answered]
    mean_conf = sum(r.confidence for r in responses) / len(responses)
    # Kapsam cezası: yanıtlanan alt soru oranı düştükçe güven de düşer (tamamı yanıtlandıysa ceza yok)
    confidence = round(mean_conf * (0.6 + 0.4 * len(answered) / total_parts))
    warnings = [
        f"Bileşik soru {total_parts} alt soruya ayrılarak arandı: "
        f"{len(answered)} alt soru kaynaklarla yanıtlandı"
        + (f", {len(missing)} alt soru yüklenen dokümanlarda bulunamadı." if missing else ".")
    ]
    for r in responses:
        warnings.extend(w for w in r.warnings if w not in warnings)
    return RAGResponse(
        question=question.strip(),
        answer=compose_answer([(i, p, r.answer) for i, p, r in answered], missing),
        is_answered=True,
        confidence=confidence,
        citations=_unique_by_label([c for r in responses for c in r.citations]),
        evidence=_unique_by_label([e for r in responses for e in r.evidence]),
        rewritten_query=question.strip(),
        search_queries=list(dict.fromkeys(q for r in responses for q in r.search_queries)),
        departments=list(departments) if departments else None,
        timings_ms={"total_ms": round(total_ms, 1), "parts": float(total_parts)},
        warnings=warnings,
        reranker=responses[0].reranker,
    )


def merge_audits(
    part_audits: Sequence[tuple[int, str, AuditResult]],
    missing: Sequence[str],
    *,
    total_parts: int,
    method: str,
) -> AuditResult:
    """Alt soru denetimlerini tek karara indirger; Denetçi'nin reddettiği alt soru da eksik kısım sayılır."""
    kept = [(i, p, a) for i, p, a in part_audits if a.status != "rejected"]
    rejected = [p for _, p, a in part_audits if a.status == "rejected"]
    all_missing = [*missing, *rejected]
    checks = [c for _, _, a in part_audits for c in a.checks]
    duration = round(sum(a.duration_ms for _, _, a in part_audits), 2)
    if not kept:
        summary = f"Bileşik sorunun {total_parts} alt sorusunun hiçbiri kaynakta doğrulanamadı; yanıt reddedildi."
        return AuditResult("rejected", NO_INFO_ANSWER, summary, method, duration, checks)
    removed = sum(1 for c in checks if not c.supported)
    status = "approved" if all(a.status == "approved" for _, _, a in kept) and not rejected else "revised"
    summary = (f"Bileşik soru ({total_parts} alt soru): {len(kept)} alt sorunun yanıtı "
               "kaynak parçalarıyla doğrulandı")
    if removed:
        summary += f", {removed} desteklenmeyen iddia çıkarıldı"
    if all_missing:
        summary += f"; {len(all_missing)} alt soru dokümanlarda bulunamadı (yanıtta not olarak belirtildi)"
    answer = compose_answer([(i, p, a.answer) for i, p, a in kept], all_missing)
    return AuditResult(status, answer, summary + ".", method, duration, checks)
