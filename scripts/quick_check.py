"""Örnek dokümanlarla terminalden hızlı uçtan uca doğrulama.

Kullanım:
    python scripts/quick_check.py            # GOOGLE_API_KEY varsa Gemini, yoksa çevrimdışı mod
    python scripts/quick_check.py --offline  # her zaman çevrimdışı mod

Çevrimdışı mod: HashingEmbedder + sözcüksel reranker + sezgisel sorgu yeniden yazma.
Yanıt üretimi için *alıntı yapan bir taklit LLM* kullanılır (en alakalı satırı
etiketiyle döndürür) — arama, halüsinasyon kapısı, atıf doğrulama, güven skoru
ve süre ölçümü gerçek kodla çalışır; yalnızca dil üretimi taklittir.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.document_loader import process_file  # noqa: E402
from src.hybrid_retriever import HybridRetriever, LexicalReranker  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.rag_engine import RAGEngine  # noqa: E402
from src.text_utils import tokenize  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


class ExtractiveStubLLM:
    """ÇEVRİMDIŞI TAKLİT: bağlamdaki en alakalı satırı etiketiyle aynen alıntılar."""

    _BLOCK_RE = re.compile(r'<parça etiket="\[(?P<label>[^\]]+)\]"[^>]*>\n(?P<body>.*?)\n</parça>', re.S)

    def stream(self, messages, *, system=None, temperature=None):
        prompt = messages[-1][1]
        question = prompt.rsplit("SORU:", 1)[-1]
        q = set(tokenize(question))
        best = (0.0, "", "")
        # Bağlam parçaları rerank sırasıyla gelir → yalnızca en alakalı parçadan alıntı yap
        top = self._BLOCK_RE.search(prompt)
        for line in (top.group("body").split("\n") if top else []):
            if line.strip().startswith("#") or set(line.strip()) <= set("|- "):
                continue  # başlık ve tablo ayraç satırlarını atla
            overlap = len(q & set(tokenize(line))) / (len(q) or 1)
            if overlap > best[0]:
                best = (overlap, line.strip(" -|"), top.group("label"))
        if best[0] == 0:
            yield "Yüklenen kurumsal dokümanlarda bu bilgi yer almamaktadır."
            return
        yield f"{best[1]} [{best[2]}]"

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        raise LLMError("çevrimdışı modda yapısal çıktı yok")


QUESTIONS: list[tuple[str, list[str] | None, list[tuple[str, str]]]] = [
    ("7 yıldır çalışıyorum, kaç gün yıllık izin hakkım var?", ["İnsan Kaynakları"], []),
    (
        "Peki 15 yılı geçince?",
        ["İnsan Kaynakları"],
        [("user", "Yıllık izin süreleri kıdeme göre nasıl?"), ("assistant", "Kıdeme göre değişir [..]")],
    ),
    ("250.000 TL üzerindeki bir harcamayı kim onaylar?", ["Finans"], []),
    ("FİN-PRS-011 dokümanının sahibi kimdir?", ["Finans"], []),
    ("Yıllık fazla mesai sınırı kaç saat?", None, []),
    ("Şirket otoparkının aylık ücreti ne kadar?", None, []),
    ("Masraf beyanı kaç gün içinde yapılmalı?", ["İnsan Kaynakları"], []),  # departman izolasyonu
]


def build_engine(offline: bool) -> tuple[RAGEngine, str]:
    settings = get_settings()
    if not offline and settings.api_key:
        engine = RAGEngine.from_settings(settings)
        # Kalıcı üretim deposunu kirletmemek için geçici depo kullan
        engine.retriever.store = VectorStore(
            engine.retriever.store.embedder, settings, persist_dir=tempfile.mkdtemp(prefix="rag_check_")
        )
        return engine, f"GEMINI ({settings.llm_model} / {settings.embedding_model})"

    from src.embeddings import HashingEmbedder

    # Çevrimdışı: LLM tabanlı yeniden yazma yok → doğrudan sezgisel yönteme düş
    settings = settings.model_copy(update={"query_rewriting": False})
    store = VectorStore(HashingEmbedder(), settings, persist_dir=":memory:", collection_name="quick_check")
    retriever = HybridRetriever(store, LexicalReranker(), settings)
    return RAGEngine(retriever, ExtractiveStubLLM(), settings), "ÇEVRİMDIŞI (hashing embedder + taklit LLM)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    engine, mode = build_engine(args.offline)
    print(f"\n=== Kurumsal RAG — hızlı doğrulama · Mod: {mode} ===\n")

    for name, dept in [("IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları"), ("Finans_Harcama_Proseduru.md", "Finans")]:
        doc = process_file(settings.sample_docs_dir / name, dept, settings)
        res = engine.retriever.store.add_document(doc)
        print(f"  + {name:<38} [{dept}] {res.message}")
    print()

    for question, departments, history in QUESTIONS:
        r = engine.ask(question, history, departments)
        scope = ", ".join(departments) if departments else "Tüm departmanlar"
        print("─" * 100)
        print(f"SORU      : {question}   ({scope})")
        if r.rewritten_query != question:
            print(f"YENİDEN   : {r.rewritten_query}")
        print("YANIT     : " + textwrap.fill(r.answer, 100, subsequent_indent=" " * 12))
        print(f"GÜVEN     : %{r.confidence}  ·  cevaplandı={r.is_answered}  ·  reranker={r.reranker}")
        t = r.timings_ms
        print(
            f"SÜRE      : toplam {t['total_ms']} ms (yeniden yazma {t['rewrite_ms']}, arama {t.get('search_ms')}, "
            f"rerank {t.get('rerank_ms')}, üretim {t.get('generation_ms', 0)})"
        )
        for c in r.citations:
            print(f"KAYNAK    : [{c.label}] · {c.department} · alaka %{c.relevance}")
        for e in r.evidence[:3]:
            mark = "✓" if e.cited else " "
            print(f"  {mark} kanıt  %{e.relevance:5.1f} {'+'.join(e.matched_by):<12} [{e.label}]  {e.text[:70]!r}…")
        for w in r.warnings:
            print(f"UYARI     : {w}")
    print("─" * 100)


if __name__ == "__main__":
    main()
