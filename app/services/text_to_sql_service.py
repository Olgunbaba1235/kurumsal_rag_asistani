"""Text-to-SQL ve veri analitiği ajanı.

Akış
----
1. **Üretim** — LLM, şema açıklaması + soru (+ önceki denemelerin hataları) ile
   yapısal JSON üretir: ``{sql, explanation, chart}``.
2. **Doğrulama** — `SQLGuard` (yalnızca tek SELECT, izinli tablolar/fonksiyonlar,
   DML/DDL yok, tenant_id yok, satır sınırı). Çalıştırılan SQL ağaçtan yeniden üretilir.
3. **Güvenli çalıştırma** — tenant kapsamlı oturumda:
   ``SET TRANSACTION READ ONLY`` · ``SET LOCAL statement_timeout`` ·
   ``SET LOCAL ROLE rag_readonly`` (yalnızca 4 tabloda SELECT yetkisi) · RLS.
   Doğrulayıcı atlatılsa bile veritabanı yazma/silme işlemini reddeder.
4. **Kendini düzeltme** — doğrulama veya çalıştırma hatası LLM'e geri verilir,
   en fazla ``ANALYTICS_MAX_ATTEMPTS`` deneme yapılır.
5. **Python analizi** — sayısal sütun istatistikleri, en büyük pay, dönemsel
   değişim gibi deterministik içgörüler; otomatik Vega-Lite grafik spesifikasyonu.
6. **Özet** — LLM yalnızca sorgu sonucundaki rakamlarla kısa bir özet yazar;
   LLM yoksa/başarısızsa deterministik içgörüler kullanılır.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from app.core.config import ApiSettings
from app.core.db import Database
from app.models.sql_models import ANALYTICS_TABLES, CUSTOMER_SEGMENTS, ORDER_STATUSES, SALES_CHANNELS
from app.services.sql_guard import SQLGuard, UnsafeSQLError, ValidatedSQL
from app.services.tenant_service import TenantContext
from src.llm import LLM, LLMError, Message, StructuredOutput

logger = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = f"""PostgreSQL veritabanı — B2B toptan satış ve üretim firması. Tutarlar TRY, KDV hariç.

customers (müşteriler)
  id BIGINT PK · code TEXT (müşteri kodu) · name TEXT (unvan)
  segment TEXT ∈ {list(CUSTOMER_SEGMENTS)} · city TEXT · region TEXT (coğrafi bölge)
  credit_limit NUMERIC (kredi limiti) · payment_terms_days INT (vade, gün) · since DATE (müşteri olma tarihi)
  is_active BOOLEAN

products (ürünler)
  id BIGINT PK · sku TEXT · name TEXT · category TEXT · unit TEXT (birim)
  unit_price NUMERIC (liste satış fiyatı) · unit_cost NUMERIC (birim maliyet)
  stock_qty INT (mevcut stok) · reorder_level INT (kritik stok seviyesi) · is_active BOOLEAN

orders (siparişler)
  id BIGINT PK · order_no TEXT · customer_id BIGINT → customers.id
  order_date DATE · shipped_date DATE (NULL olabilir)
  status TEXT ∈ {list(ORDER_STATUSES)} · channel TEXT ∈ {list(SALES_CHANNELS)}
  discount_rate NUMERIC (yüzde iskonto) · total_amount NUMERIC (sipariş net tutarı)

order_items (sipariş kalemleri)
  id BIGINT PK · order_id BIGINT → orders.id · product_id BIGINT → products.id
  quantity INT · unit_price NUMERIC (iskontolu birim fiyat) · line_total NUMERIC (satır tutarı)

Hesaplama kuralları:
  ciro/satış = SUM(orders.total_amount) veya SUM(order_items.line_total); iptal siparişler (status = 'iptal') HARİÇ tutulur
  brüt kâr = SUM(order_items.line_total - order_items.quantity * products.unit_cost)
  kritik stok = products.stock_qty < products.reorder_level"""

_SQL_SYSTEM = f"""Sen kıdemli bir veri analisti ve PostgreSQL uzmanısın. Kullanıcının sorusunu yanıtlayan TEK bir \
salt-okunur SELECT sorgusu yaz.

{SCHEMA_DESCRIPTION}

KURALLAR:
- Yalnızca SELECT (gerekirse WITH/CTE). INSERT/UPDATE/DELETE/DDL ASLA.
- Yalnızca yukarıdaki 4 tablo ve sütunlar. tenant_id sütununu ASLA kullanma (izolasyon otomatik).
- Açık sütun adları ve anlamlı Türkçe takma adlar kullan (ör. AS toplam_ciro). SELECT * kullanma.
- Sıralama gerekiyorsa ORDER BY, liste sorularında LIMIT (en fazla 100) kullan.
- Tarih gruplamaları için date_trunc('month', order_date) veya to_char(order_date, 'YYYY-MM') kullan.
- Tutarları ROUND(..., 2) ile yuvarla.
- Soru bu veriyle yanıtlanamıyorsa sql alanını boş bırak ve explanation'da nedenini yaz.
- 'chart' alanında sonucu en iyi gösterecek grafiği öner: x = kategori/zaman sütunu, y = sayısal sütun \
(takma adlarla). Zaman serisi → line, kategorik karşılaştırma → bar, ≤6 parçalı pay → pie, aksi halde table."""

_SUMMARY_SYSTEM = """Sen bir iş analistisin. Yalnızca sana verilen SORGU SONUCU ve İSTATİSTİKLERdeki rakamları \
kullanarak soruyu 2-4 cümleyle yanıtla. Sonuçta olmayan hiçbir rakam veya iddia ekleme. Para birimini TRY \
olarak belirt. 'highlights' alanına en önemli 1-3 bulguyu yaz."""


# =============================================================== LLM şemaları
class ChartHint(StructuredOutput):
    type: Literal["bar", "line", "pie", "table", "none"] = "table"
    x: str | None = None
    y: str | None = None
    title: str | None = None


class SQLGeneration(StructuredOutput):
    sql: str = Field(default="", description="Tek bir PostgreSQL SELECT sorgusu; yanıtlanamıyorsa boş")
    explanation: str = Field(default="", description="Sorgunun ne hesapladığının kısa Türkçe açıklaması")
    chart: ChartHint = Field(default_factory=ChartHint)


class DataSummary(StructuredOutput):
    summary: str
    highlights: list[str] = Field(default_factory=list)


# =============================================================== Sonuç tipleri
@dataclass(slots=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]  # JSON-uyumlu
    kinds: dict[str, str]  # sütun → "number" | "temporal" | "text" | "bool"
    row_count: int
    truncated: bool


@dataclass(slots=True)
class Attempt:
    sql: str
    error: str | None = None


@dataclass(slots=True)
class SQLAnswer:
    question: str
    success: bool
    sql: str = ""
    explanation: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    tables: list[str] = field(default_factory=list)
    numeric_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    chart: dict[str, Any] | None = None
    summary: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# =============================================================== Python analizi
def _to_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return value


def _kind(values: Sequence[Any]) -> str:
    sample = [v for v in values if v is not None]
    if not sample:
        return "text"
    if all(isinstance(v, bool) for v in sample):
        return "bool"
    if all(isinstance(v, int | float | Decimal) and not isinstance(v, bool) for v in sample):
        return "number"
    if all(isinstance(v, dt.date | dt.datetime) for v in sample):
        return "temporal"
    return "text"


def _fmt(n: float) -> str:
    s = f"{n:,.2f}" if abs(n) < 1e15 else str(n)
    return s.replace(",", "X").replace(".", ",").replace("X", ".").removesuffix(",00")


def _is_identifier(col: str) -> bool:
    c = col.lower()
    return c == "id" or c.endswith("_id") or c.endswith("_no") or c in ("code", "sku")


def analyze(result: QueryResult) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Sayısal sütun istatistikleri ve deterministik içgörüler (rakamlar doğrudan veriden)."""
    stats: dict[str, dict[str, float]] = {}
    insights: list[str] = []
    if not result.rows:
        return stats, ["Sorgu sonucu boş: koşullara uyan kayıt bulunamadı."]

    idx = {c: i for i, c in enumerate(result.columns)}
    for col, kind in result.kinds.items():
        if kind != "number" or _is_identifier(col):
            continue
        vals = [float(r[idx[col]]) for r in result.rows if r[idx[col]] is not None]
        if vals:
            stats[col] = {
                "count": len(vals),
                "sum": round(sum(vals), 2),
                "mean": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }

    label_col = next((c for c in result.columns if result.kinds.get(c) in ("text", "temporal")), None)
    measure = next(iter(stats), None)
    if label_col and measure and len(result.rows) > 1:
        li, mi = idx[label_col], idx[measure]
        pairs = [(r[li], float(r[mi])) for r in result.rows if r[mi] is not None]
        total = sum(v for _, v in pairs)
        if result.kinds[label_col] == "temporal":
            ordered = sorted(pairs, key=lambda p: str(p[0]))
            first, last = ordered[0], ordered[-1]
            peak = max(ordered, key=lambda p: p[1])
            if first[1]:
                change = (last[1] - first[1]) / abs(first[1]) * 100
                insights.append(
                    f"{measure}: {first[0]} → {last[0]} döneminde %{change:+.1f} değişim "
                    f"({_fmt(first[1])} → {_fmt(last[1])})."
                )
            insights.append(f"En yüksek {measure}: {peak[0]} ({_fmt(peak[1])}).")
        else:
            top = max(pairs, key=lambda p: p[1])
            share = f" — toplamın %{top[1] / total * 100:.1f}'i" if total > 0 else ""
            insights.append(f"En yüksek {measure}: {top[0]} ({_fmt(top[1])}{share}).")
            if len(pairs) >= 5 and total > 0:
                top3 = sum(sorted((v for _, v in pairs), reverse=True)[:3])
                insights.append(f"İlk 3 kayıt toplam {measure} değerinin %{top3 / total * 100:.1f}'ini oluşturuyor.")
        insights.append(f"Toplam {measure}: {_fmt(total)} ({len(pairs)} satır).")
    elif measure and len(result.rows) == 1:
        insights.append(f"{measure}: {_fmt(stats[measure]['sum'])}.")
    if result.truncated:
        insights.append(f"Sonuç ilk {result.row_count} satırla sınırlandırıldı.")
    return stats, insights


def build_chart(result: QueryResult, hint: ChartHint | None, title: str) -> dict[str, Any] | None:
    """Vega-Lite v5 spesifikasyonu (ön yüzden bağımsız; vega-embed ile doğrudan çizilir)."""
    if not result.rows or len(result.columns) < 2:
        return None
    kinds = result.kinds
    numeric = [c for c in result.columns if kinds.get(c) == "number" and not _is_identifier(c)]
    labels = [c for c in result.columns if kinds.get(c) in ("text", "temporal")]
    hint = hint or ChartHint()
    if hint.type in ("none", "table") and hint.x is None:
        hint = ChartHint()  # öneri yoksa otomatik seç
    x = hint.x if hint.x in result.columns else (labels[0] if labels else None)
    y = hint.y if hint.y in numeric else (numeric[0] if numeric else None)
    if x is None or y is None or x == y:
        return None

    mark = hint.type if hint.type in ("bar", "line", "pie") else None
    if mark is None:
        mark = "line" if kinds.get(x) == "temporal" else "bar"
    if mark == "pie" and len(result.rows) > 8:
        mark = "bar"

    xi, yi = result.columns.index(x), result.columns.index(y)
    values = [{x: r[xi], y: r[yi]} for r in result.rows[:200]]
    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": hint.title or title,
        "data": {"values": values},
        "width": "container",
    }
    if mark == "pie":
        spec["mark"] = {"type": "arc", "tooltip": True}
        spec["encoding"] = {
            "theta": {"field": y, "type": "quantitative"},
            "color": {"field": x, "type": "nominal"},
        }
    elif mark == "line":
        x_type = "temporal" if kinds.get(x) == "temporal" or _looks_like_period(values, x) else "ordinal"
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": {"field": x, "type": x_type},
            "y": {"field": y, "type": "quantitative"},
        }
    else:
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "x": {"field": x, "type": "nominal", "sort": "-y"},
            "y": {"field": y, "type": "quantitative"},
        }
    return spec


def _looks_like_period(values: list[dict], x: str) -> bool:
    return all(isinstance(v[x], str) and len(v[x]) >= 7 and v[x][:4].isdigit() and v[x][4] == "-" for v in values)


# =============================================================== Servis
class TextToSQLService:
    def __init__(self, db: Database, settings: ApiSettings, llm: LLM | None):
        self.db = db
        self.settings = settings
        self.llm = llm
        self.guard = SQLGuard(ANALYTICS_TABLES, max_rows=settings.analytics_max_rows)

    # ------------------------------------------------------------ çalıştırma
    async def execute(self, tenant: TenantContext, validated: ValidatedSQL) -> QueryResult:
        """Doğrulanmış SQL'i salt-okunur, zaman aşımlı, en az yetkili rolle çalıştırır."""
        s = self.settings
        async with self.db.tenant_session(tenant.id, tenant.departments) as session, session.begin():
            conn = await session.connection()
            await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            await conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(s.analytics_statement_timeout_ms)}")
            await conn.exec_driver_sql(f'SET LOCAL ROLE "{s.analytics_readonly_role}"')
            # exec_driver_sql: SQL olduğu gibi gönderilir (':' içeren literal'ler bind parametresi sanılmaz)
            result = await conn.exec_driver_sql(validated.sql)
            columns = list(result.keys())
            raw = result.fetchmany(s.analytics_max_rows + 1)

        truncated = len(raw) > s.analytics_max_rows
        raw = raw[: s.analytics_max_rows]
        # SELECT * ile gelebilecek tenant_id sütununu sonuçtan çıkar
        keep = [i for i, c in enumerate(columns) if c.lower() != "tenant_id"]
        columns = [columns[i] for i in keep]
        raw_rows = [[row[i] for i in keep] for row in raw]
        kinds = {c: _kind([r[i] for r in raw_rows]) for i, c in enumerate(columns)}
        rows = [[_to_json(v) for v in r] for r in raw_rows]
        return QueryResult(columns, rows, kinds, len(rows), truncated)

    # ------------------------------------------------------------ üretim
    def _prompt(self, question: str, history: Sequence[Message], attempts: list[Attempt]) -> str:
        parts = []
        if history:
            recent = "\n".join(f"{'Kullanıcı' if r == 'user' else 'Asistan'}: {t[:300]}" for r, t in list(history)[-6:])
            parts.append(f"SOHBET GEÇMİŞİ:\n{recent}")
        parts.append(f"SORU: {question}")
        if attempts:
            errors = "\n".join(f"- Deneme {i}: {a.sql}\n  HATA: {a.error}" for i, a in enumerate(attempts, 1))
            parts.append(f"ÖNCEKİ BAŞARISIZ DENEMELER (bu hataları düzelt):\n{errors}")
        return "\n\n".join(parts)

    async def _summarize(self, question: str, answer: SQLAnswer) -> str:
        fallback = " ".join(answer.insights) or "Sorgu tamamlandı."
        if self.llm is None or not answer.rows:
            return fallback
        sample = [dict(zip(answer.columns, r, strict=True)) for r in answer.rows[:50]]
        prompt = (
            f"SORU: {question}\n\nSORGU AÇIKLAMASI: {answer.explanation}\n\n"
            f"SORGU SONUCU ({answer.row_count} satır{', kısaltılmış' if answer.truncated else ''}, ilk 50):\n{sample}\n\n"
            f"İSTATİSTİKLER: {answer.numeric_stats}\n\nİÇGÖRÜLER: {answer.insights}"
        )
        try:
            out = await asyncio.to_thread(
                self.llm.generate_json, prompt, DataSummary, system=_SUMMARY_SYSTEM, fast=True
            )
            return out.summary.strip() or fallback
        except LLMError as exc:
            answer.warnings.append(f"Özet üretilemedi, deterministik içgörüler kullanıldı: {exc}")
            return fallback

    async def run(self, tenant: TenantContext, question: str, history: Sequence[Message] = ()) -> SQLAnswer:
        t_start = time.perf_counter()
        answer = SQLAnswer(question=question.strip(), success=False)
        if self.llm is None:
            answer.warnings.append("Dil modeli yapılandırılmamış (LLM_PROVIDER=none); SQL üretilemedi.")
            return answer

        generation: SQLGeneration | None = None
        result: QueryResult | None = None
        t_gen = t_exec = 0.0
        for _ in range(self.settings.analytics_max_attempts):
            t0 = time.perf_counter()
            try:
                generation = await asyncio.to_thread(
                    self.llm.generate_json,
                    self._prompt(question, history, answer.attempts),
                    SQLGeneration,
                    system=_SQL_SYSTEM,
                    fast=False,
                )
            except LLMError as exc:
                answer.warnings.append(f"SQL üretilemedi: {exc}")
                break
            finally:
                t_gen += time.perf_counter() - t0

            if not generation.sql.strip():
                answer.explanation = generation.explanation
                answer.warnings.append("Soru mevcut satış verisiyle yanıtlanamıyor.")
                break

            attempt = Attempt(sql=generation.sql.strip())
            answer.attempts.append(attempt)
            t0 = time.perf_counter()
            try:
                validated = self.guard.validate(attempt.sql)
                attempt.sql = validated.sql
                result = await self.execute(tenant, validated)
                answer.tables = list(validated.tables)
                break
            except UnsafeSQLError as exc:
                attempt.error = f"Güvenlik doğrulaması: {exc}"
                # Log'a yalnızca ret nedeni yazılır; sorgu metni müşteri verisi içerebilir
                logger.warning("[%s] Güvensiz SQL reddedildi: %s", tenant.slug, str(exc).split(":")[0][:80])
            except (DBAPIError, SQLAlchemyError) as exc:
                msg = str(getattr(exc, "orig", exc)).splitlines()[0][:300]
                attempt.error = f"Veritabanı hatası: {msg}"
            finally:
                t_exec += time.perf_counter() - t0

        answer.timings_ms["sql_generation_ms"] = round(t_gen * 1000, 1)
        answer.timings_ms["sql_execution_ms"] = round(t_exec * 1000, 1)

        if result is None or generation is None:
            if answer.attempts and answer.attempts[-1].error:
                answer.warnings.append(f"{len(answer.attempts)} denemede geçerli sorgu üretilemedi.")
            answer.timings_ms["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return answer

        t0 = time.perf_counter()
        answer.success = True
        answer.sql = answer.attempts[-1].sql
        answer.explanation = generation.explanation
        answer.columns, answer.rows = result.columns, result.rows
        answer.row_count, answer.truncated = result.row_count, result.truncated
        answer.numeric_stats, answer.insights = analyze(result)
        answer.chart = build_chart(result, generation.chart, title=question.strip()[:80])
        answer.timings_ms["analysis_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        answer.summary = await self._summarize(question, answer)
        answer.timings_ms["summary_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        answer.timings_ms["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return answer
