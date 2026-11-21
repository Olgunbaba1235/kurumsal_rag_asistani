"""Yanıt metnindeki uzun atıf etiketlerini numaralı işaretlere çevirir.

Model ve Denetçi atıfları ``[IK_Izin_ve_Calisma_Yonetmeligi.md - Sayfa 4]`` biçiminde yazar (doğrulama bu
etiketlere dayanır). Ekranda her cümlenin sonunda aynı uzun dosya adı tekrar etmesin diye etiketler ilk
geçiş sırasına göre ``[1]``, ``[2]`` … olur; ayrıntılar yanıtın altındaki numaralı kaynak kartlarındadır.
Yalnızca görüntü katmanıdır: saklanan yanıt metni ve doğrulama etiketli metinle çalışmaya devam eder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.text_utils import fold

_BRACKET_RE = re.compile(r"\[([^\[\]\n]{3,300})\]")
_PART_SPLIT_RE = re.compile(r"\s*[;|]\s*")
_LABEL_PART_RE = re.compile(r"^(?P<src>.+?)\s*[-–—,]\s*(?:sayfa|s\.|sf\.)\s*(?P<page>\d+)$", re.IGNORECASE)
_TRAILING_OPEN_RE = re.compile(r"\[[^\]\n]*$")


def _stem(source: str) -> str:
    source = fold(source).strip()
    return source.rsplit(".", 1)[0] if "." in source else source


def number_citations(text: str, citations: Sequence[Any] = ()) -> tuple[str, list[Any]]:
    """Metindeki atıf etiketlerini ``[n]`` yapar. Dönen liste n'inci işaretin kaynağıdır (Citation nesnesi;
    eşleşmeyen etiket için etiket metni). Atıf olmayan köşeli parantezlere dokunulmaz."""
    exact = {fold(c.label).strip(): c for c in citations}
    loose: dict[tuple[str, int | None], list[Any]] = {}
    for c in citations:
        loose.setdefault((_stem(c.source), c.page), []).append(c)
    numbers: dict[str, int] = {}
    order: list[Any] = []

    def resolve(part: str, m: re.Match) -> tuple[str, Any]:
        hit = exact.get(fold(part).strip())
        if hit is None:
            candidates = loose.get((_stem(m.group("src")), int(m.group("page"))), [])
            hit = candidates[0] if len(candidates) == 1 else None
        return (hit.label, hit) if hit is not None else (fold(part).strip(), part.strip())

    def replace(match: re.Match) -> str:
        found: list[int] = []
        for part in _PART_SPLIT_RE.split(match.group(1).strip()):
            m = _LABEL_PART_RE.match(part.strip())
            if not m:
                return match.group(0)  # atıf değil (ör. "[Not]") → olduğu gibi
            key, item = resolve(part, m)
            if key not in numbers:
                numbers[key] = len(numbers) + 1
                order.append(item)
            if numbers[key] not in found:
                found.append(numbers[key])
        return "".join(f"[{n}]" for n in found)

    return _BRACKET_RE.sub(replace, text), order


def streaming_view(text: str, citations: Sequence[Any] = ()) -> str:
    """Akış sırasında gösterilecek metin: tamamlanmış etiketler numaralanır, yarım kalmış ``[IK_Iz…`` gizlenir."""
    numbered, _ = number_citations(text, citations)
    return _TRAILING_OPEN_RE.sub("", numbered)
