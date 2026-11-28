from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatQuery(BaseModel):
    question: str = Field(min_length=2, max_length=2000, examples=["Kaç gün yıllık izin hakkım var?"])
    departments: list[str] | None = Field(
        default=None,
        description="Aranacak departmanlar. Boşsa tenant'ın tüm departmanları aranır.",
        examples=[["İnsan Kaynakları"]],
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    top_k: int | None = Field(default=None, ge=1, le=20)
    suggest: bool = Field(
        default=True,
        description="\"Bunu mu sormak istediniz?\" önerisi üretilsin mi? Kullanıcı bir öneriyi seçip tekrar sorduysa "
        "`false` gönderin (öneri → yeni öneri döngüsü oluşmaz).",
    )


class CitationOut(BaseModel):
    label: str = Field(description="Atıf etiketi: 'Dosya Adı - Sayfa X'")
    source: str
    page: int | None
    department: str
    relevance: float = Field(description="0-100 alaka skoru")
    evidence: str = Field(description="Atıf yapılan parçanın metni (kanıt)")


class EvidenceOut(BaseModel):
    label: str
    source: str
    page: int | None
    department: str
    text: str
    relevance: float
    vector_score: float
    bm25_score: float
    matched_by: list[str]
    cited: bool


class SuggestionOut(BaseModel):
    """Yalnızca kullanıcının erişebildiği departmanların belge adı/başlık/terimlerinden türetilir."""

    has_suggestion: bool
    suggested_query: str = Field(description="Önerilen (düzeltilmiş) soru; belirsiz soruda ilk niyet")
    alternatives: list[str] = Field(
        description="Belirsiz soruda tıklanabilir 2–3 niyet (ilki `suggested_query`); yazım düzeltmesinde boş"
    )
    kind: Literal["typo", "ambiguous", "none"]


class ChatResponse(BaseModel):
    answer: str
    is_answered: bool
    confidence: int = Field(ge=0, le=100, description="Güven skoru (%)")
    citations: list[CitationOut]
    evidence: list[EvidenceOut]
    rewritten_query: str
    search_queries: list[str]
    departments: list[str]
    reranker: str
    timings_ms: dict[str, float]
    total_ms: float
    warnings: list[str]
    suggestion: SuggestionOut | None = Field(default=None, description="\"Bunu mu sormak istediniz?\" (yoksa null)")

    @classmethod
    def from_result(cls, result: Any, departments: list[str], suggestion: Any = None) -> ChatResponse:
        """`src.rag_engine.RAGResponse` (+ `src.suggestions.QuerySuggestion`) → API yanıtı."""
        payload = asdict(result)
        payload.pop("question", None)
        payload["departments"] = departments
        payload["total_ms"] = result.total_ms
        if suggestion is not None and suggestion.has_suggestion:
            payload["suggestion"] = suggestion.to_dict()
        return cls.model_validate(payload)
