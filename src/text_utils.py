"""Türkçe metin normalizasyonu ve tokenizasyon (BM25 ve sözcüksel reranking için).

* Türkçe'ye özgü küçük harf dönüşümü (I → ı, İ → i)
* Diakritik katlama: "çalışma" ve "calisma" aynı token'a düşer (kullanıcılar
  sıklıkla Türkçe karakter kullanmadan yazar)
* Sayı normalizasyonu: "4.500" → "4500", "1.000.000" → "1000000"
* Stopword temizliği
* Kısaltma eşitleme: "15 dk" ile "15 dakika" aynı token'a düşer (bkz. ``ABBREVIATIONS``). Hem sorguya hem
  belgeye uygulandığı için eşleşme simetriktir; kısaltma farkı alaka skorunu eşiğin altına düşürmez.
* F5 kök bulma: alfabetik token'ların ilk 5 karakteri. Eklemeli bir dil olan
  Türkçe'de bilgi erişimi için basit ama etkili olduğu gösterilmiş bir yöntemdir
  ("izinleri", "izinlerin", "izni" → "izin"/"izinl" … yakın köklere düşer).
"""

from __future__ import annotations

import re

_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_TOKEN_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?|[^\W\d_]+", re.UNICODE)
_STEM_LEN = 5

STOPWORDS = frozenset(
    """
    acaba ama ancak artik aslinda az bana bazi belki ben beni benim beri bile bir biri birkac birsey biz bize
    bizi bizim bu buna bunda bundan bunlar bunlari bunlarin bunu bunun burada cok cunku da daha dahi de defa
    diye eger en gibi hem hep hepsi her hic icin ile ise kac kadar ki kim kime kimi mi mu ne neden nedir nasil
    nerede nereye niye o olan olarak oldu olsun olur on ona ondan onlar onlari onlarin onu onun oyle pek
    sadece sanki sen senin siz sizin su suna sunda sundan sunu sunun tarafindan tum ve veya ya yani yine
    hangi hangisi var yok midir mudur peki lutfen nelerdir
    """.split()
)


# Katlanmış (fold) biçimde yaygın kısaltma → tam sözcük. Bilinçli olarak dar tutulur: yalnızca
# kurumsal belgelerde sık görülen ve başka anlama gelmeyen birim kısaltmaları.
ABBREVIATIONS = {
    "dk": "dakika",
    "dak": "dakika",
    "sn": "saniye",
    "sa": "saat",
    "km": "kilometre",
    "kg": "kilogram",
    "lt": "litre",
}


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def fold(text: str) -> str:
    """Türkçe küçük harfe çevirip diakritikleri ASCII'ye katlar."""
    return turkish_lower(text).translate(_FOLD)


def _normalize_number(tok: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", tok):
        tok = tok.replace(".", "")
    return tok.replace(",", ".")


def tokenize(text: str, *, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(fold(text)):
        if raw[0].isdigit():
            tokens.append(_normalize_number(raw))
            continue
        raw = ABBREVIATIONS.get(raw, raw)
        if remove_stopwords and raw in STOPWORDS:
            continue
        if len(raw) < 2:
            continue
        tokens.append(raw[:_STEM_LEN] if stem else raw)
    return tokens
