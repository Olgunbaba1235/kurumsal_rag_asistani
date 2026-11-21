"""Vitrin grafikleri (Plotly). Plotly kurulu değilse aynı veriler Vega-Lite ile çizilir.

Örnek (simüle) veriler grafik başlığında açıkça "örnek veri" olarak etiketlenir; gerçek oturum
verisi olan grafikler (departman dağılımı, sorgu kalitesi) bu oturumun indeksinden ve sorgularından gelir.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

try:  # pragma: no cover - ortam bağımlı
    import plotly.graph_objects as go

    PLOTLY = True
except ImportError:  # pragma: no cover
    go = None
    PLOTLY = False

PALETTE = ["#7C83FF", "#22D3EE", "#34D399", "#FBBF24", "#F472B6", "#A78BFA"]
_FONT = {"color": "#CBD5E1", "family": "Inter, Segoe UI, sans-serif"}

# Plotly'nin tarih ekseni ay adlarını İngilizce yazar ("Nov 7"); Türkçe etiketler burada üretilir.
TR_MONTHS = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
TR_MONTHS_LONG = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim",
                  "Kasım", "Aralık"]
TR_WEEKDAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def tr_day_label(iso_date: str) -> str:
    """`2026-11-07` → `7 Kas`"""
    d = dt.date.fromisoformat(iso_date)
    return f"{d.day} {TR_MONTHS[d.month - 1]}"


def tr_long_date(iso_date: str) -> str:
    """`2026-11-07` → `7 Kasım 2026, Cumartesi` (fare üzerine gelince gösterilir)"""
    d = dt.date.fromisoformat(iso_date)
    return f"{d.day} {TR_MONTHS_LONG[d.month - 1]} {d.year}, {TR_WEEKDAYS[d.weekday()]}"


def sample_usage(days: int = 14, seed: int = 7, today: dt.date | None = None) -> list[dict[str, Any]]:
    """Sunum için deterministik örnek kullanım serisi (gerçek ölçüm değildir)."""
    rng = random.Random(seed)
    today = today or dt.date.today()
    rows = []
    for i in range(days, 0, -1):
        day = today - dt.timedelta(days=i)
        weekday = day.weekday() < 5
        queries = int((140 if weekday else 45) * (1 + 0.035 * (days - i)) * rng.uniform(0.85, 1.15))
        answered = int(queries * rng.uniform(0.86, 0.95))
        rows.append({"date": day.isoformat(), "queries": queries, "answered": answered})
    return rows


def _layout(title: str, height: int = 320) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#E2E8F0"}, "x": 0.02},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": _FONT,
        "height": height,
        # Sol boşluk eksen başlığı ve tik etiketlerine yetecek kadar geniş; automargin taşmayı önler
        "margin": {"l": 72, "r": 20, "t": 50, "b": 48},
        "legend": {"orientation": "h", "y": -0.2, "font": {"color": "#CBD5E1"}},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.06)", "zeroline": False, "automargin": True},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.06)", "zeroline": False, "automargin": True},
    }


# ------------------------------------------------------------------ veri hazırlama (Plotly'den bağımsız, test edilir)
def department_counts(documents: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in documents:
        counts[d.department] += d.chunk_count
    return dict(counts.most_common())


def quality_points(history: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"n": i + 1, "ms": round(h.total_ms, 1), "confidence": h.confidence, "answered": h.answered,
         "question": h.question[:60]}
        for i, h in enumerate(history)
    ]


# ------------------------------------------------------------------ Plotly figürleri
def usage_figure(rows: list[dict[str, Any]], session_queries: int = 0):  # noqa: ANN201
    title = "Günlük sorgu hacmi — örnek veri" + (f" · bu oturum: {session_queries}" if session_queries else "")
    if not PLOTLY:
        return _vl_line(rows, title)
    labels = [tr_day_label(r["date"]) for r in rows]
    hover = [tr_long_date(r["date"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=[r["queries"] for r in rows], name="Sorgu", customdata=hover,
        hovertemplate="%{customdata}<br>Sorgu: %{y}<extra></extra>",
        mode="lines+markers", line={"color": PALETTE[0], "width": 3, "shape": "spline"},
        fill="tozeroy", fillcolor="rgba(124,131,255,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[r["answered"] for r in rows], name="Kaynaklı yanıt", customdata=hover,
        hovertemplate="%{customdata}<br>Kaynaklı yanıt: %{y}<extra></extra>",
        mode="lines", line={"color": PALETTE[1], "width": 2, "dash": "dot"},
    ))
    layout = _layout(title)
    ticks = labels[:: 1 if len(labels) <= 10 else 2]
    # Kategori ekseni: etiketler aynen (Türkçe) yazılır; Plotly tarih biçimlendirmesi devreye girmez
    layout["xaxis"] = {**layout["xaxis"], "type": "category", "tickmode": "array", "tickvals": ticks,
                       "ticktext": ticks, "tickangle": 0}
    layout["yaxis"] = {**layout["yaxis"], "title": {"text": "Sorgu", "standoff": 10}}
    fig.update_layout(**layout)
    return fig


def department_figure(counts: dict[str, int]):  # noqa: ANN201
    title = "Bilgi tabanı: departman bazında parça dağılımı"
    if not PLOTLY:
        return _vl_bar(counts, title)
    total = sum(counts.values()) or 1
    shares = [f"%{round(v / total * 100)}" for v in counts.values()] or [""]  # Türkçe yazım: %53
    fig = go.Figure(go.Pie(
        labels=list(counts) or ["Belge yok"], values=list(counts.values()) or [1], hole=0.62,
        marker={"colors": PALETTE, "line": {"color": "#0B1222", "width": 2}},
        customdata=shares, texttemplate="%{customdata}" if counts else "%{label}", textposition="outside",
        automargin=True, sort=False,
        hovertemplate="%{label}: %{value} parça (%{customdata})<extra></extra>",
    ))
    # Uzun departman adları dilimin yanına sığmayıp kesilmesin: adlar altta lejantta
    layout = _layout(title)
    layout["legend"] = {**layout["legend"], "y": -0.08, "x": 0.5, "xanchor": "center"}
    fig.update_layout(**layout, showlegend=bool(counts))
    return fig


def quality_figure(points: list[dict[str, Any]]):  # noqa: ANN201
    title = "Bu oturumdaki sorgular: güven skoru ve yanıt süresi"
    if not PLOTLY:
        return _vl_scatter(points, title)
    fig = go.Figure(go.Scatter(
        x=[p["n"] for p in points], y=[p["confidence"] for p in points], mode="markers+lines",
        marker={"size": [max(10, min(34, p["ms"] / 4)) for p in points],
                "color": [PALETTE[2] if p["answered"] else "#F87171" for p in points],
                "line": {"color": "#0B1222", "width": 1}},
        line={"color": "rgba(124,131,255,0.35)"},
        text=[f'{p["question"]}<br>{p["ms"]} ms' for p in points], hovertemplate="%{text}<br>Güven %{y}%<extra></extra>",
    ))
    layout = _layout(title)
    layout["margin"] = {**layout["margin"], "l": 84}  # "Güven (%)" başlığı soldan kesilmesin
    layout["yaxis"] = {**layout["yaxis"], "range": [0, 105], "title": {"text": "Güven (%)", "standoff": 12}}
    layout["xaxis"] = {**layout["xaxis"], "title": {"text": "Sorgu sırası"}, "dtick": 1}
    fig.update_layout(**layout)
    return fig


# ------------------------------------------------------------------ Vega-Lite yedekleri (Plotly yoksa)
def _vl_base(title: str) -> dict[str, Any]:
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "title": title,
            "background": "transparent", "config": {"view": {"stroke": None}}}


def _vl_line(rows: list[dict[str, Any]], title: str) -> dict[str, Any]:
    values = [{**r, "gün": tr_day_label(r["date"])} for r in rows]
    return {**_vl_base(title), "data": {"values": values},
            "mark": {"type": "area", "line": True, "opacity": 0.35},
            "encoding": {"x": {"field": "gün", "type": "ordinal", "sort": None, "title": None,
                               "axis": {"labelAngle": 0}},
                         "y": {"field": "queries", "type": "quantitative", "title": "Sorgu"}}}


def _vl_bar(counts: dict[str, int], title: str) -> dict[str, Any]:
    return {**_vl_base(title), "data": {"values": [{"department": k, "chunks": v} for k, v in counts.items()]},
            "mark": "bar", "encoding": {"x": {"field": "department", "type": "nominal", "title": None},
                                        "y": {"field": "chunks", "type": "quantitative", "title": "Parça"}}}


def _vl_scatter(points: list[dict[str, Any]], title: str) -> dict[str, Any]:
    return {**_vl_base(title), "data": {"values": points}, "mark": {"type": "point", "filled": True},
            "encoding": {"x": {"field": "n", "type": "quantitative", "title": "Sorgu sırası"},
                         "y": {"field": "confidence", "type": "quantitative", "title": "Güven (%)"}}}
