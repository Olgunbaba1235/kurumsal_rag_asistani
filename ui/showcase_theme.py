"""Vitrin arayüzünün görsel dili: koyu tema, buzlu cam (glassmorphism) kartlar ve HTML bileşenleri.

Tüm dinamik metinler `html.escape` ile kaçışlanır (belge içeriği ve kullanıcı girdisi HTML olarak
yorumlanmaz). Bileşenler **tek satırlık** HTML üretir: Streamlit'in Markdown ayrıştırıcısı boş satır
gördüğünde HTML bloğunu kapatıp kalanını Markdown olarak yorumlar (ham "##" başlıklar, kartın içine
taşan tablolar). Kaynak parçalarındaki Markdown bu yüzden burada güvenli HTML'e çevrilir.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from typing import Any

ACCENT = "#7C83FF"
ACCENT_2 = "#22D3EE"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"

CSS = """
<style>
:root {
  --glass-bg: rgba(255, 255, 255, 0.055);
  --glass-border: rgba(255, 255, 255, 0.12);
  --text-dim: #9CA3AF;
  --accent: #7C83FF;
  --accent-2: #22D3EE;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(124, 131, 255, 0.22), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34, 211, 238, 0.14), transparent 55%),
    linear-gradient(180deg, #0A0F1E 0%, #0B1222 55%, #0A0F1E 100%);
  background-attachment: fixed;
}
.block-container { padding-top: 1.4rem; max-width: 1380px; }
[data-testid="stSidebar"] {
  background: rgba(10, 15, 30, 0.72);
  backdrop-filter: blur(16px);
  border-right: 1px solid var(--glass-border);
}
[data-testid="stHeader"] { background: transparent; }
/* Streamlit geliştirici araç çubuğu (Deploy, ana menü) son kullanıcıya gösterilmez */
[data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"], #MainMenu,
[data-testid="stToolbarActions"], [data-testid="stDecoration"] { display: none !important; }

.glass {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

/* --- Karşılama --- */
.hero { padding: 26px 30px; margin-bottom: 18px; position: relative; overflow: hidden; }
.hero h1 {
  font-size: 2.05rem; line-height: 1.2; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #FFFFFF 0%, #C7CBFF 45%, #67E8F9 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: #CBD5E1; margin: 0 0 14px 0; font-size: 1.02rem; max-width: 900px; }
.badge {
  display: inline-block; padding: 4px 11px; margin: 0 6px 6px 0; border-radius: 999px; font-size: 0.78rem;
  color: #E0E7FF; background: rgba(124, 131, 255, 0.16); border: 1px solid rgba(124, 131, 255, 0.35);
}
.badge.cyan { background: rgba(34, 211, 238, 0.12); border-color: rgba(34, 211, 238, 0.35); color: #CFFAFE; }
.badge.green { background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.35); color: #D1FAE5; }

/* --- KPI kartları --- */
/* Sabit yükseklik + satır sınırı: içerik uzunluğu ne olursa olsun beş kart aynı hizada durur */
.kpi {
  position: relative; box-sizing: border-box; height: 146px; padding: 16px 18px; overflow: hidden;
  display: flex; flex-direction: column; transition: transform .15s ease, border-color .15s ease;
}
.kpi:hover { transform: translateY(-2px); border-color: rgba(124, 131, 255, 0.45); }
.kpi .label {
  color: var(--text-dim); font-size: 0.8rem; text-transform: uppercase; letter-spacing: .06em;
  padding-right: 34px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi .value {
  color: #F8FAFC; font-size: 1.7rem; font-weight: 750; margin-top: 4px; line-height: 1.15;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi .value.small {
  font-size: 1.15rem; line-height: 1.25; white-space: normal;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.kpi .sub {
  color: #A5B4FC; font-size: 0.8rem; line-height: 1.35; margin-top: auto; padding-top: 6px; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.kpi .icon { position: absolute; top: 14px; right: 16px; font-size: 1.35rem; opacity: .9; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }

/* --- Kaynak (atıf) kartları --- */
.cite { padding: 14px 16px; margin: 10px 0; border-left: 3px solid var(--accent); }
.cite .head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
.cite .file { color: #F1F5F9; font-weight: 650; font-size: 0.95rem; }
.pill {
  font-size: 0.72rem; padding: 2px 9px; border-radius: 999px; color: #E2E8F0;
  background: rgba(148, 163, 184, 0.14); border: 1px solid rgba(148, 163, 184, 0.28);
}
.pill.page { background: rgba(124, 131, 255, 0.18); border-color: rgba(124, 131, 255, 0.45); color: #E0E7FF; }
.bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,0.08); overflow: hidden; margin: 4px 0 10px 0; }
.bar > span { display:block; height:100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.cite .quote {
  color: #CBD5E1; font-size: 0.86rem; line-height: 1.5; border-radius: 10px; padding: 10px 12px;
  background: rgba(0, 0, 0, 0.22); max-height: 340px; overflow-y: auto;
}
.cite .quote p { margin: 0 0 6px 0; font-size: 0.86rem; }
.cite .quote .cite-h {
  color: #E0E7FF; font-size: 0.84rem; font-weight: 700; letter-spacing: .01em; margin: 8px 0 4px 0;
}
.cite .quote .cite-h:first-child { margin-top: 0; }
.cite .quote .cite-li { padding-left: 14px; text-indent: -12px; margin: 2px 0; }
.cite .quote .cite-note { color: var(--text-dim); font-size: 0.8rem; border-left: 2px solid var(--glass-border);
  padding-left: 8px; margin: 4px 0; }
.cite .quote .cite-more { color: var(--text-dim); margin-top: 4px; }
.cite .quote table { width: 100%; border-collapse: collapse; margin: 6px 0; font-size: 0.8rem; }
.cite .quote th, .cite .quote td {
  border: 1px solid rgba(255, 255, 255, 0.10); padding: 4px 8px; text-align: left; vertical-align: top;
}
.cite .quote th { background: rgba(124, 131, 255, 0.14); color: #E0E7FF; font-weight: 650; }
.cite .cite-label {
  color: var(--text-dim); font-size: 0.78rem; margin-top: 8px; padding-top: 8px;
  border-top: 1px dashed rgba(255, 255, 255, 0.12);
}
.cite .cite-label code { background: transparent; color: #C7D2FE; font-size: 0.78rem; padding: 0; }
.meta { color: var(--text-dim); font-size: 0.8rem; margin-top: 4px; }

/* --- Denetçi Kontrol Raporu: arama metrikleri, filtre uyarısı, alt sorular --- */
.facts { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 4px 0; }
.fact { padding: 6px 10px; border-radius: 10px; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); }
.fact .k { display: block; color: var(--text-dim); font-size: 0.7rem; text-transform: uppercase; letter-spacing: .05em; }
.fact .v { color: #F1F5F9; font-size: 0.92rem; font-weight: 650; }
.fact .v.ok { color: #6EE7B7; } .fact .v.bad { color: #FCA5A5; }
.gauge { position: relative; height: 6px; border-radius: 6px; background: rgba(255,255,255,0.08); margin: 6px 0 10px 0; }
.gauge > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.gauge > i { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #FBBF24; border-radius: 2px; }
.callout {
  border-radius: 12px; padding: 10px 14px; margin: 8px 0; font-size: 0.86rem; line-height: 1.45;
  color: #FDE68A; background: rgba(251, 191, 36, 0.10); border: 1px solid rgba(251, 191, 36, 0.40);
}
.callout .probe { color: #CBD5E1; font-size: 0.8rem; margin-top: 4px; }
.part-row { font-size: 0.82rem; color: #CBD5E1; margin: 3px 0; }
.part-row .ok { color: #6EE7B7; } .part-row .bad { color: #FCA5A5; }

/* --- Sohbet --- */
[data-testid="stChatMessage"] {
  background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px;
  backdrop-filter: blur(10px); padding: 10px 14px; margin-bottom: 10px;
}
.section-title { color: #E2E8F0; font-weight: 700; font-size: 1.05rem; margin: 6px 0 10px 0; }
.doc-row { display:flex; justify-content:space-between; align-items:center; padding: 12px 16px; margin: 8px 0; }
.doc-row .name { color:#F1F5F9; font-weight:600; }
.audit-badge {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; margin: 2px 0 8px 0;
  border-radius: 999px; font-size: 0.8rem; font-weight: 650; letter-spacing: .01em;
  backdrop-filter: blur(10px); border: 1px solid;
}
.audit-badge.approved { color: #A7F3D0; background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.45); }
.audit-badge.revised  { color: #FDE68A; background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.45); }
.audit-badge.rejected { color: #FECACA; background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.5); }
.audit-row { font-size: 0.82rem; color: #CBD5E1; margin: 4px 0; }
.audit-row .bad { color: #FCA5A5; }
.stButton > button { border-radius: 12px; border: 1px solid var(--glass-border); }
/* Hızlı başlangıç butonları: metin kesilmez, gerekirse iki satıra iner; dört buton aynı yükseklikte */
[class*="st-key-sample_"] button { min-height: 54px; padding: 6px 12px; }
[class*="st-key-sample_"] button * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  line-height: 1.25; text-align: center;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #6366F1, #06B6D4); border: none; font-weight: 650;
}
</style>
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def hero(title: str, subtitle: str, badges: Iterable[tuple[str, str]]) -> str:
    chips = "".join(f'<span class="badge {esc(cls)}">{esc(text)}</span>' for text, cls in badges)
    return f'<div class="glass hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>{chips}</div>'


def kpi_card(icon: str, label: str, value: str, sub: str = "", *, dot: str | None = None,
             small: bool = False) -> str:
    dot_html = f'<span class="dot" style="background:{esc(dot)}"></span>' if dot else ""
    value_cls = "value small" if small else "value"
    return (
        f'<div class="glass kpi"><span class="icon">{esc(icon)}</span>'
        f'<div class="label">{esc(label)}</div>'
        f'<div class="{value_cls}" title="{esc(value)}">{dot_html}{esc(value)}</div>'
        f'<div class="sub" title="{esc(sub)}">{esc(sub)}</div></div>'
    )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(?:[-*•+]|(\d+)[.)])\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _inline(text: str) -> str:
    """Kaçışlanmış metinde **kalın** ve *italik* işaretlerini HTML'e çevirir; diğer her şey düz metindir."""
    out = esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", out)
    return out


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _clip(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[:budget].rsplit(" ", 1)[0] + " …"


def evidence_html(text: str, max_chars: int = 1400) -> str:
    """Kaynak parçasının (Markdown) güvenli, satır sonu içermeyen HTML önizlemesi.

    Başlık diyezleri gösterilmez (başlıklar kart içi küçük başlık olur), Markdown tabloları gerçek
    ``<table>`` olur, uzun parçalar satır/hücre bütünlüğü korunarak kısaltılır.
    """
    lines = [ln.strip() for ln in _COMMENT_RE.sub("", text.replace("\r", "")).split("\n")]
    out: list[str] = []
    para: list[str] = []
    budget = max_chars
    truncated = False

    def flush() -> None:
        nonlocal budget, truncated
        if para and not truncated:
            joined = " ".join(para)
            if len(joined) > budget:
                joined, truncated = _clip(joined, max(budget, 40)), True
            budget -= len(joined)
            out.append(f"<p>{_inline(joined)}</p>")
        para.clear()

    i = 0
    while i < len(lines) and not truncated:
        line = lines[i]
        if not line:
            flush()
        elif line.startswith("|"):
            flush()
            rows: list[str] = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            header = _cells(rows[0]) if len(rows) > 1 and _TABLE_SEP_RE.match(rows[1]) else None
            body = rows[2:] if header else rows
            html_rows = []
            if header:
                html_rows.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
            for row in body:
                if budget <= 0:
                    truncated = True
                    break
                budget -= len(row)
                html_rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _cells(row)) + "</tr>")
            out.append(f"<table>{''.join(html_rows)}</table>")
            continue
        elif m := _HEADING_RE.match(line):
            flush()
            out.append(f'<div class="cite-h">{_inline(m.group(2).strip("# "))}</div>')
            budget -= len(m.group(2))
        elif m := _LIST_RE.match(line):
            flush()
            bullet = f"{m.group(1)}." if m.group(1) else "•"
            item = _clip(m.group(2), max(budget, 40))
            truncated = len(item) < len(m.group(2))
            budget -= len(item)
            out.append(f'<div class="cite-li">{bullet} {_inline(item)}</div>')
        elif line.startswith(">"):
            flush()
            note = line.lstrip("> ").strip()
            budget -= len(note)
            out.append(f'<div class="cite-note">{_inline(note)}</div>')
        else:
            para.append(line)
        if budget <= 0 and i < len(lines) - 1:
            truncated = True
        i += 1
    flush()
    if truncated or i < len(lines):
        out.append('<div class="cite-more">…</div>')
    return "".join(out)


def citation_card(label: str, source: str, page: int | None, department: str, relevance: float,
                  evidence: str, max_chars: int = 1400) -> str:
    """Kaynak kartı: başlık satırı, alaka çubuğu, parça önizlemesi ve (ayrı kapta) atıf etiketi."""
    rel = max(0.0, min(100.0, float(relevance)))
    page_pill = f'<span class="pill page">Sayfa {esc(page)}</span>' if page else ""
    return (
        '<div class="glass cite">'
        f'<div class="head"><span class="file">📄 {esc(source)}</span>{page_pill}'
        f'<span class="pill">{esc(department or "Genel")}</span>'
        f'<span class="pill">Alaka %{rel:.0f}</span></div>'
        f'<div class="bar"><span style="width:{rel:.0f}%"></span></div>'
        f'<div class="quote">{evidence_html(evidence, max_chars)}</div>'
        f'<div class="cite-label">Atıf etiketi: <code>[{esc(label)}]</code></div></div>'
    )


AUDIT_LABELS = {
    "approved": "✓ Denetçi Onaylı",
    "revised": "✎ Denetçi Tarafından Düzeltildi",
    "rejected": "✕ Kaynakta Bulunamadı — Uydurma Engellendi",
}


def audit_badge(status: str) -> str:
    if status not in AUDIT_LABELS:
        return ""
    return f'<span class="audit-badge {esc(status)}">{esc(AUDIT_LABELS[status])}</span>'


def audit_row(text: str, reason: str, ok: bool) -> str:
    icon, cls = ("✓", "") if ok else ("✕", "bad")
    short = text if len(text) <= 140 else text[:140].rsplit(" ", 1)[0] + " …"
    return f'<div class="audit-row"><span class="{cls}">{icon} {esc(reason)}</span> — {esc(short)}</div>'


def doc_row(filename: str, department: str, pages: int, chunks: int, file_type: str) -> str:
    return (
        f'<div class="glass doc-row"><div><div class="name">📄 {esc(filename)}</div>'
        f'<div class="meta">{esc(department)} · {esc(file_type.upper())}</div></div>'
        f'<div><span class="pill">{esc(pages)} sayfa</span> <span class="pill page">{esc(chunks)} parça</span></div></div>'
    )


def retrieval_facts(best: float, threshold: float, departments: Sequence[str]) -> str:
    """En iyi alaka skoru, eşik ve arama kapsamı (eşik işaretli alaka göstergesiyle)."""
    best = max(0.0, min(100.0, float(best)))
    threshold = max(0.0, min(100.0, float(threshold)))
    passed = best >= threshold
    scope = ", ".join(departments) if departments else "Tüm departmanlar"
    gate = "Geçildi" if passed else "Geçilemedi · model çağrılmadı"
    return (
        '<div class="facts">'
        f'<div class="fact"><span class="k">En iyi alaka skoru</span><span class="v">%{best:.0f}</span></div>'
        f'<div class="fact"><span class="k">Eşik değeri</span><span class="v">%{threshold:.0f}</span></div>'
        f'<div class="fact"><span class="k">Alaka kapısı</span>'
        f'<span class="v {"ok" if passed else "bad"}">{gate}</span></div>'
        f'<div class="fact"><span class="k">Arama kapsamı</span><span class="v">{esc(scope)}</span></div>'
        '</div>'
        f'<div class="gauge" title="Alaka %{best:.0f} · eşik %{threshold:.0f}">'
        f'<span style="width:{best:.0f}%"></span><i style="left:{threshold:.0f}%"></i></div>'
    )


def filter_callout(note: str, probe: str = "") -> str:
    """Filtrenin bir reddin nedeni olduğunu açıkça gösteren uyarı kutusu (yanıtın hemen altında)."""
    probe_html = f'<div class="probe">{esc(probe)}</div>' if probe else ""
    return f'<div class="callout">🔒 {esc(note)}{probe_html}</div>'


def filter_fact(departments: Sequence[str], probe: str = "") -> str:
    """Rapor içindeki kısa filtre uyarısı (yanıt altındaki kutunun tekrarı değil, özet satırı)."""
    detail = f" {esc(probe)}" if probe else ""
    return (f'<div class="part-row"><span class="bad">⚠ Filtre uyarısı:</span> yalnızca '
            f'{esc(", ".join(departments))} belgeleri tarandı.{detail}</div>')


def part_row(text: str, answered: bool, relevance: float) -> str:
    icon, cls, state = ("✓", "ok", "kaynaklı yanıtlandı") if answered else ("✕", "bad", "dokümanlarda yok")
    return (f'<div class="part-row"><span class="{cls}">{icon} {state}</span> · alaka %{relevance:.0f} — '
            f'{esc(text)}</div>')


def uploader_css(max_mb: int, types: Sequence[str]) -> str:
    """Dosya yükleme alanındaki Streamlit'in İngilizce metinlerini Türkçeleştirir ("Upload", "… per file").

    Streamlit bu metinler için bir yerelleştirme ayarı sunmadığından metin CSS ile gizlenip yerine
    Türkçesi yazılır; sınır ve uzantılar çalışma zamanındaki gerçek ayarlardan üretilir.
    """
    kinds = " · ".join(t.upper() for t in types)
    limit = f"Dosya başına en fazla {int(max_mb)} MB"
    return (
        "<style>"
        '[data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0 !important; white-space: normal !important; }'
        '[data-testid="stFileUploaderDropzoneInstructions"] span::after {'
        f' content: "{limit}\\A {kinds}"; white-space: pre-line; font-size: 0.78rem; line-height: 1.35; }}'
        '[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] '
        '[data-testid="stMarkdownContainer"] { display: none !important; }'
        '[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] [data-has-shortcut]::after {'
        ' content: "Dosya seç"; }'
        '[data-testid="stFileUploaderDropzone"] > div:not([data-testid]) > span { font-size: 0 !important; }'
        '[data-testid="stFileUploaderDropzone"] > div:not([data-testid]) > span::after {'
        ' content: "Dosyaları buraya bırakın"; font-size: 0.875rem; }'
        "</style>"
    )


def service_error_card(message: str) -> str:
    """Yapay zekâ servisi hatası (429/503…): uydurma engeli rozetinden ayrı, tekrar denenebilir uyarı."""
    return (
        '<div class="callout"><b>⚠️ Yanıt şu anda üretilemedi</b>'
        f'<div class="probe">{esc(message)}</div></div>'
    )

