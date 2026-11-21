"""Denetçi (Auditor) ajanı: Araştırmacı'nın (RAG motoru) yanıtını yalnızca getirilen kaynak parçalarıyla karşılaştırır.

Karar türleri:
* ``approved`` — her iddia kaynakla destekleniyor; yanıt aynen geçer.
* ``revised``  — desteklenmeyen iddialar (kaynakta geçmeyen sayı/tarih/oran ya da kaynakla örtüşmeyen cümle)
  çıkarılır veya düzeltilir; kalan yanıt kaynaklıdır.
* ``rejected`` — dokümanda karşılığı yok ya da hiçbir iddia desteklenmiyor; sabit "bilgi yok" yanıtı döner.
* ``skipped``  — Denetçi Modu kapalı.

İki uygulama vardır:
* ``RuleBasedAuditor`` (çevrimdışı): iddia–kaynak sözcük örtüşmesi + sayısal/alıntı tutarlılığı. Ağ gerektirmez.
* ``GeminiAuditor``: hızlı/ekonomik Gemini Flash modeliyle yapısal karar; ardından aynı kural tabanlı sayısal
  kontrol güvenlik ağı olarak yeniden uygulanır. LLM hatasında kural tabanlı denetçiye düşer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.llm import LLM, LLMError
from src.rag_engine import NO_INFO_ANSWER, RAGResponse
from src.text_utils import fold, tokenize

AuditStatus = Literal["approved", "revised", "rejected", "skipped"]

_LABEL_RE = re.compile(r"\[([^\[\]\n]{3,200})\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,:/]\d+)*")
_QUOTE_RE = re.compile(r"[\"“”«»]([^\"“”«»]{4,200})[\"“”«»]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\-])")
_LIST_MARKER_RE = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
# "…yüklenen dokümanlarda yer almamaktadır" türü kapsam notları (katlanmış biçimde aranır)
_ABSENCE_RE = re.compile(r"\b(?:yer almamaktadir|yer almiyor|bulunmamaktadir|bulunmuyor|yer verilmemistir)\b")
_DOC_REF_RE = re.compile(r"\b(?:dokuman|belge|kaynak|baglam)")


@dataclass(slots=True)
class ClaimCheck:
    text: str
    supported: bool
    reason: str
    overlap: float


@dataclass(slots=True)
class AuditResult:
    status: AuditStatus
    answer: str
    summary: str
    method: str
    duration_ms: float
    checks: list[ClaimCheck] = field(default_factory=list)

    @property
    def removed(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.supported]


def normalize_number(raw: str) -> str:
    """`1.200` → `1200`, `1,5` → `1.5`; saat/tarih biçimleri (`10:00`, `15.02.2026`) korunur."""
    if ":" in raw or raw.count(".") >= 2 or "/" in raw:
        return raw
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
        return raw.replace(".", "")
    return raw.replace(",", ".")


def numbers_in(text: str) -> set[str]:
    return {normalize_number(n) for n in _NUMBER_RE.findall(text)}


def _strip_labels(text: str) -> str:
    return _LABEL_RE.sub("", text).strip(" -•*\t")


_MD_MARK_RE = re.compile(r"\*\*|__|`|^#{1,6}\s*")
_SPACE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")


def display_text(claim: str) -> str:
    """Rapor için okunur iddia metni: atıf etiketleri, Markdown işaretleri ve noktalama öncesi boşluk temizlenir."""
    text = _MD_MARK_RE.sub("", _strip_labels(claim))
    text = _SPACE_PUNCT_RE.sub(r"\1", " ".join(text.split()))
    return text.strip(" -•*:") or text


_BOLD_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*:?$")


def _is_heading(claim: str) -> bool:
    """Markdown başlığı ("### Hibrit Model") veya yalnızca kalın kısa satır ("**Ekipman**") iddia değildir.

    Sayı içeren ya da cümle gibi biten satırlar başlık sayılmaz; böylece başlık kılığında bir iddia kaçamaz.
    """
    body = _LABEL_RE.sub("", claim).strip()
    if numbers_in(body) or body.rstrip("*").endswith((".", "!", "?")):
        return False
    return body.startswith("#") or (bool(_BOLD_ONLY_RE.match(body)) and len(body.split()) <= 8)


def _is_scope_note(body: str) -> bool:
    """"…yüklenen dokümanlarda yer almamaktadır." gibi kapsam notları bilgi iddiası değildir.

    Yalnızca sayı içermeyen ve doküman/kaynak/bağlama atıf yapan yokluk cümleleri sayılır; böylece
    "…yer almamaktadır" kalıbı içine gizlenmiş bir sayı ya da olgu denetimden kaçamaz.
    """
    folded = fold(_QUOTE_RE.sub("", body))
    return not numbers_in(folded) and bool(_ABSENCE_RE.search(folded)) and bool(_DOC_REF_RE.search(folded))


def _is_connective(claim: str) -> bool:
    """Bilgi taşımayan bağlaç satırları ("…şöyledir:") ve kapsam notları denetlenmez."""
    body = _strip_labels(claim)
    if _is_scope_note(body) or _is_heading(claim):
        return True
    return not numbers_in(body) and (body.endswith(":") or len(tokenize(body)) < 3)


class RuleBasedAuditor:
    """Çevrimdışı denetçi: kaynak metinle örtüşme + sayı/alıntı tutarlılığı."""

    name = "Kural tabanlı doğrulayıcı"

    def __init__(self, min_overlap: float = 0.6):
        self.min_overlap = min_overlap

    def check_claim(self, claim: str, sources: Sequence[str]) -> ClaimCheck:
        body = _strip_labels(claim)
        shown = display_text(claim)
        source_text = "\n".join(sources)
        missing = sorted(numbers_in(body) - numbers_in(source_text))
        if missing:
            return ClaimCheck(shown, False, f"Kaynakta geçmeyen sayı/tarih/oran: {', '.join(missing)}", 0.0)
        for q in _QUOTE_RE.findall(body):
            if q.strip().lower() not in source_text.lower():
                return ClaimCheck(shown, False, f"Alıntı kaynakta birebir yok: “{q[:60]}”", 0.0)
        claim_tokens = set(tokenize(body))
        if not claim_tokens:
            return ClaimCheck(shown, True, "Bilgi içermeyen ifade", 1.0)
        overlap = len(claim_tokens & set(tokenize(source_text))) / len(claim_tokens)
        if overlap < self.min_overlap:
            return ClaimCheck(shown, False, f"Kaynakla örtüşme düşük (%{overlap * 100:.0f})", overlap)
        return ClaimCheck(shown, True, f"Kaynakla örtüşme %{overlap * 100:.0f}", overlap)

    def _sources_for(self, claim: str, response: RAGResponse) -> list[str]:
        labels = set(_LABEL_RE.findall(claim))
        by_label = [e.text for e in response.evidence if e.label in labels]
        return by_label or [e.text for e in response.evidence]

    def audit(self, response: RAGResponse) -> AuditResult:
        t0 = time.perf_counter()
        if not response.is_answered:
            return AuditResult("rejected", NO_INFO_ANSWER,
                               "Dokümanlarda soruyu karşılayan bilgi yok; yanıt üretilmesine izin verilmedi.",
                               self.name, _ms(t0))

        # Satır satır, satır içinde cümle cümle denetlenir; düzeltilen yanıtta paragraf/madde yapısı korunur.
        lines: list[str] = []
        checks: list[ClaimCheck] = []
        for line in response.answer.split("\n"):
            if not line.strip():
                lines.append("")
                continue
            m = _LIST_MARKER_RE.match(line)
            marker = m.group(1) if m else ""
            kept: list[str] = []
            for raw in _SENTENCE_SPLIT_RE.split(line[len(marker):]):
                claim = raw.strip()
                if not claim:
                    continue
                if _is_connective(claim):
                    kept.append(claim)
                    continue
                check = self.check_claim(claim, self._sources_for(claim, response))
                checks.append(check)
                if check.supported:
                    kept.append(claim)
            if kept:
                lines.append(marker + " ".join(kept))

        dropped = [c for c in checks if not c.supported]
        supported = [c for c in checks if c.supported]
        if not dropped:
            return AuditResult("approved", response.answer,
                               f"{len(checks)} iddianın tamamı kaynak parçalarıyla destekleniyor.",
                               self.name, _ms(t0), checks)
        if not supported:
            return AuditResult("rejected", NO_INFO_ANSWER,
                               f"{len(dropped)} iddianın hiçbiri kaynakta doğrulanamadı; yanıt reddedildi.",
                               self.name, _ms(t0), checks)
        revised = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return AuditResult("revised", revised,
                           f"{len(dropped)} desteklenmeyen iddia çıkarıldı, {len(supported)} iddia doğrulandı.",
                           self.name, _ms(t0), checks)


class _AuditVerdict(BaseModel):
    verdict: Literal["approved", "revised", "rejected"]
    corrected_answer: str = Field(description="Yalnızca kaynakla desteklenen, [Dosya - Sayfa X] atıflı yanıt")
    issues: list[str] = Field(default_factory=list, description="Desteklenmeyen iddialar (kısa)")


_AUDIT_SYSTEM = """Sen titiz bir denetçisin. Sana bir YANIT ve onu üreten KAYNAK PARÇALARI verilir.
Yanıttaki her iddiayı yalnızca kaynak parçalarıyla karşılaştır. Kaynakta geçmeyen sayı, tarih, oran, isim veya
bilgi varsa onu çıkar ya da kaynaktaki doğru değerle düzelt. Atıf etiketlerini ([Dosya - Sayfa X]) koru.
Hepsi destekleniyorsa verdict=approved ve yanıtı aynen döndür; düzelttiysen verdict=revised;
hiçbir iddia desteklenmiyorsa verdict=rejected. Yeni bilgi EKLEME."""


class GeminiAuditor:
    """Gemini Flash ile denetim; sonuç yine kural tabanlı sayısal kontrolden geçirilir."""

    name = "Gemini Flash denetçi"

    def __init__(self, llm: LLM, fallback: RuleBasedAuditor | None = None):
        self.llm = llm
        self.rules = fallback or RuleBasedAuditor()

    def audit(self, response: RAGResponse) -> AuditResult:
        t0 = time.perf_counter()
        if not response.is_answered:
            return self.rules.audit(response)
        context = "\n\n".join(f"[{e.label}]\n{e.text}" for e in response.evidence)
        prompt = f"KAYNAK PARÇALARI:\n{context}\n\nYANIT:\n{response.answer}"
        try:
            v = self.llm.generate_json(prompt, _AuditVerdict, system=_AUDIT_SYSTEM, fast=True)
        except LLMError:
            result = self.rules.audit(response)
            result.summary += " (Gemini denetçiye ulaşılamadı; kural tabanlı doğrulama kullanıldı.)"
            return result
        if v.verdict == "rejected" or not v.corrected_answer.strip():
            return AuditResult("rejected", NO_INFO_ANSWER, "Denetçi hiçbir iddiayı kaynakta doğrulayamadı.",
                               self.name, _ms(t0))
        # Güvenlik ağı: LLM'in onayladığı/düzelttiği metin de sayısal tutarlılık kontrolünden geçer
        candidate = RAGResponse(**{**_as_dict(response), "answer": v.corrected_answer})
        safety = self.rules.audit(candidate)
        status: AuditStatus = "revised" if v.verdict == "revised" or safety.status != "approved" else "approved"
        if safety.status == "rejected":
            status = "rejected"
        issues = "; ".join(v.issues[:3])
        summary = (safety.summary if status != "revised" or not issues
                   else f"Düzeltilen noktalar: {issues}. {safety.summary}")
        return AuditResult(status, safety.answer, summary, self.name, _ms(t0), safety.checks)


def _as_dict(r: RAGResponse) -> dict:
    return {f: getattr(r, f) for f in r.__dataclass_fields__}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def skipped(response: RAGResponse) -> AuditResult:
    return AuditResult("skipped", response.answer, "Denetçi Modu kapalı; yanıt denetlenmedi.", "—", 0.0)
