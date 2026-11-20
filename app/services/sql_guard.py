"""Text-to-SQL güvenlik doğrulayıcısı (deny-by-default).

LLM'in ürettiği SQL **güvenilmeyen girdi** kabul edilir. Bu modül, sorguyu
çalıştırmadan önce sqlglot ile ayrıştırır ve yalnızca aşağıdakilerin hepsi
sağlanıyorsa kabul eder:

* Tam olarak **bir** ifade (``SELECT 1; DROP TABLE …`` reddedilir)
* Kök ifade SELECT veya küme işlemi (UNION/INTERSECT/EXCEPT)
* Ağacın **hiçbir yerinde** DML/DDL/komut düğümü yok (CTE içine gizlenmiş
  ``DELETE … RETURNING`` dahil), ``SELECT … INTO`` ve ``FOR UPDATE`` yok
* Yalnızca izinli analitik tablolar (+ sorgunun kendi CTE adları); şema
  niteleyicisi yok/`public` → ``pg_catalog``, ``information_schema`` ve RAG
  tabloları (documents, tenants…) erişilemez
* ``tenant_id`` sütununa açık referans yok (izolasyon veritabanı RLS'indedir;
  ``SELECT *`` ile gelen tenant_id sütunu çalıştırıcıda sonuçtan çıkarılır)
* Fonksiyonlar: sqlglot'un tanıdığı tipli fonksiyonlar serbest; tanınmayan
  (anonim) fonksiyonlar yalnızca beyaz listeden; ``pg_*``, ``set_config``,
  ``current_setting``, ``dblink``, ``lo_*`` vb. her durumda yasak

Çalıştırılan SQL, modelin yazdığı metin değil **ayrıştırma ağacından yeniden
üretilen** SQL'dir: yorumlar (`--`, `/* */`) ve gizli ekler bu yolla düşer.
Satır sınırı zorla uygulanır.

Bu, savunma katmanlarından yalnızca biridir; sorgu ayrıca salt-okunur rol,
READ ONLY transaction, statement_timeout ve RLS ile çalıştırılır.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class UnsafeSQLError(ValueError):
    """Sorgu güvenlik politikasını ihlal ediyor."""


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "Command", "Copy",
        "Grant", "Revoke", "Transaction", "Commit", "Rollback", "Set", "Into", "Lock", "Use",
        "Pragma", "TruncateTable", "Analyze", "Kill", "LoadData", "Describe", "Show",
    )
    if hasattr(exp, name)
)

# Tanınmayan (Anonymous) fonksiyonlar için beyaz liste
_ALLOWED_ANONYMOUS = frozenset(
    {
        "date_trunc", "date_part", "to_char", "to_date", "age", "make_date", "justify_interval",
        "initcap", "split_part", "btrim", "ltrim", "rtrim", "lpad", "rpad", "unaccent",
        "width_bucket", "percentile_cont", "percentile_disc", "mode", "ntile", "cume_dist",
        "percent_rank", "dense_rank", "rank", "row_number", "first_value", "last_value", "lag", "lead",
        "nullif", "generate_series", "trunc", "div", "sign", "floor", "ceil", "ceiling", "abs", "round",
        "string_agg", "array_agg", "bool_and", "bool_or", "every",
    }
)
_FORBIDDEN_FUNC_RE = re.compile(
    r"^(pg_|lo_|dblink|set_config|current_setting|current_user|session_user|query_to_xml|"
    r"table_to_xml|xpath|txid_|version$|inet_|has_.*_privilege|nextval|setval|currval|"
    r"copy|read_|write_|file_|system)",
    re.IGNORECASE,
)
_FORBIDDEN_COLUMNS = frozenset({"tenant_id"})


@dataclass(frozen=True, slots=True)
class ValidatedSQL:
    sql: str  # çalıştırılacak, yeniden üretilmiş SQL
    tables: tuple[str, ...]
    limited_to: int


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name)
    return node.sql_name()


class SQLGuard:
    def __init__(self, allowed_tables: Iterable[str], max_rows: int = 500):
        self.allowed_tables = frozenset(t.lower() for t in allowed_tables)
        self.max_rows = max_rows

    def validate(self, sql: str) -> ValidatedSQL:
        if not sql or not sql.strip():
            raise UnsafeSQLError("Boş sorgu.")
        if len(sql) > 10_000:
            raise UnsafeSQLError("Sorgu çok uzun.")

        try:
            statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
        except ParseError as exc:
            raise UnsafeSQLError(f"SQL ayrıştırılamadı: {str(exc).splitlines()[0]}") from exc
        if len(statements) != 1:
            raise UnsafeSQLError("Yalnızca tek bir SELECT ifadesine izin verilir.")
        root = statements[0]

        if not isinstance(root, exp.Select | exp.SetOperation):
            raise UnsafeSQLError(f"Yalnızca SELECT sorgularına izin verilir ({type(root).__name__} reddedildi).")

        for node in root.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise UnsafeSQLError(f"Yasaklı ifade: {type(node).__name__.upper()}.")

        cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
        tables: set[str] = set()
        for table in root.find_all(exp.Table):
            name = table.name.lower()
            schema = (table.db or "").lower()
            if table.catalog or (schema and schema != "public"):
                raise UnsafeSQLError(f"Şema/katalog erişimi yasak: {table.sql(dialect='postgres')}.")
            if name in cte_names and not schema:
                continue
            if name not in self.allowed_tables:
                raise UnsafeSQLError(
                    f"'{name}' tablosuna erişim yok. İzinli tablolar: {', '.join(sorted(self.allowed_tables))}."
                )
            tables.add(name)
        if not tables:
            raise UnsafeSQLError("Sorgu izinli tablolardan en az birini kullanmalıdır.")

        for col in root.find_all(exp.Column):
            if col.name.lower() in _FORBIDDEN_COLUMNS:
                raise UnsafeSQLError("tenant_id sütununa erişilemez (izolasyon veritabanı tarafından sağlanır).")

        for func in root.find_all(exp.Func):
            name = _func_name(func).lower()
            if _FORBIDDEN_FUNC_RE.match(name):
                raise UnsafeSQLError(f"Yasaklı fonksiyon: {name}().")
            if isinstance(func, exp.Anonymous) and name not in _ALLOWED_ANONYMOUS:
                raise UnsafeSQLError(f"İzin verilmeyen fonksiyon: {name}().")

        limited = self._enforce_limit(root)
        rendered = limited.sql(dialect="postgres", comments=False)
        return ValidatedSQL(sql=rendered, tables=tuple(sorted(tables)), limited_to=self.max_rows)

    def _enforce_limit(self, root: exp.Expression) -> exp.Expression:
        cap = self.max_rows
        if isinstance(root, exp.Select):
            limit = root.args.get("limit")
            current = None
            if limit is not None:
                expr = limit.expression
                if isinstance(expr, exp.Literal) and expr.is_int:
                    current = int(expr.this)
                else:
                    raise UnsafeSQLError("LIMIT sabit bir tam sayı olmalıdır.")
            return root.limit(min(current, cap) if current is not None else cap, copy=True)
        # Küme işlemleri: dış sorguyla sarmala
        return exp.select("*").from_(exp.Subquery(this=root.copy(), alias=exp.TableAlias(this=exp.to_identifier("q")))).limit(cap)
