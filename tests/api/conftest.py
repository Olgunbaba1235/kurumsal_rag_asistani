"""REST API entegrasyon testleri — gerçek PostgreSQL (PGlite/WASM) + pgvector üzerinde.

* `tests/pglite` altındaki Node sunucusu bellek içi bir PostgreSQL başlatır
  (Docker gerekmez). Kurulum: ``npm install --prefix tests/pglite``
* Alembic migration'ı (tablolar, HNSW indeksi, RLS politikaları, rag_app rolü)
  gerçekten çalıştırılır.
* Uygulama superuser ile bağlansa da her transaction'da ``SET LOCAL ROLE rag_app``
  yapılır (DB_SESSION_ROLE) → RLS testlerde de üretimdeki gibi uygulanır.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import ApiSettings
from app.main import create_app
from app.services.container import ServiceContainer
from app.services.providers import Providers
from src.embeddings import HashingEmbedder
from src.hybrid_retriever import LexicalReranker
from src.llm import LLMError

PGLITE_DIR = Path(__file__).resolve().parents[1] / "pglite"
ROOT = Path(__file__).resolve().parents[2]
ADMIN_KEY = "test-admin-key-0123456789"
_LABEL_RE = re.compile(r'<parça etiket="\[([^\]]+)\]"')


class ScriptedLLM:
    """Test LLM'i: bağlamdaki ilk etiketi atıflayan deterministik yanıt üretir."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.answer_fn = lambda labels: f"İlgili kural bağlamda yer almaktadır [{labels[0]}]."
        # Şema adı → (prompt → dict) işleyicisi; tanımlı değilse LLMError (sezgisel yedeklere düşülür)
        self.json_handlers: dict[str, object] = {}
        self.json_calls: list[tuple[str, str]] = []

    def stream(self, messages, *, system=None, temperature=None):
        self.calls.append(list(messages))
        labels = _LABEL_RE.findall(messages[-1][1])
        yield self.answer_fn(labels)

    def generate_json(self, prompt, schema, *, system=None, fast=True):
        self.json_calls.append((schema.__name__, prompt))
        handler = self.json_handlers.get(schema.__name__)
        if handler is None:
            raise LLMError("testte yapısal çıktı yok")
        return schema.model_validate(handler(prompt))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    node = shutil.which("node")
    if node is None or not (PGLITE_DIR / "node_modules").exists():
        pytest.skip("Entegrasyon testleri için Node.js ve `npm install --prefix tests/pglite` gerekli.")
    port = _free_port()
    proc = subprocess.Popen(
        [node, "server.mjs", str(port)],
        cwd=PGLITE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 60
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("READY"):
            break
        if proc.poll() is not None:
            pytest.fail(f"PGlite sunucusu başlatılamadı: {line}{proc.stdout.read()}")
    else:
        proc.kill()
        pytest.fail("PGlite sunucusu 60 sn içinde hazır olmadı.")
    try:
        yield f"postgresql+asyncpg://postgres:postgres@127.0.0.1:{port}/postgres?ssl=disable"
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def settings(pg_url: str) -> ApiSettings:
    return ApiSettings(
        ENVIRONMENT="test",
        DATABASE_URL=pg_url,
        DB_POOL_SIZE=1,  # PGlite tek backend: tek bağlantı
        DB_MAX_OVERFLOW=0,
        DB_SESSION_ROLE="rag_app",  # superuser bağlantısında RLS'e tabi role geç
        DB_DISPOSE_TIMEOUT=0.5,
        EMBEDDING_PROVIDER="hashing",
        LLM_PROVIDER="none",
        REDIS_URL=None,
        RATE_LIMIT_PER_MINUTE=0,
        ADMIN_API_KEY=ADMIN_KEY,
        QUERY_REWRITING=False,
        _env_file=None,
    )


@pytest.fixture(scope="session")
def llm() -> ScriptedLLM:
    return ScriptedLLM()


def _run_migrations(sync_conn) -> None:  # noqa: ANN001
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["connection"] = sync_conn
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def client(settings: ApiSettings, llm: ScriptedLLM) -> Iterator[TestClient]:
    def factory(s: ApiSettings) -> ServiceContainer:
        providers = Providers(
            embedder=HashingEmbedder(dimension=s.embedding_dimension), llm=llm, reranker=LexicalReranker()
        )
        return ServiceContainer.build(s, providers=providers, use_redis=False)

    app = create_app(settings, container_factory=factory)
    with TestClient(app) as c:
        container: ServiceContainer = app.state.container

        async def migrate() -> None:
            async with container.db.engine.begin() as conn:
                await conn.run_sync(_run_migrations)

        c.portal.call(migrate)
        yield c


@pytest.fixture(scope="session")
def container(client: TestClient) -> ServiceContainer:
    return client.app.state.container


def run(client: TestClient, fn, *args):  # noqa: ANN001, ANN201
    """Uygulamanın event loop'unda bir coroutine fonksiyonu çalıştırır."""
    return client.portal.call(fn, *args)


def admin_headers() -> dict[str, str]:
    return {"X-Admin-Key": ADMIN_KEY}


def create_tenant(client: TestClient, slug: str, name: str, departments: list[str] | None = None) -> dict:
    body = {"name": name, "slug": slug}
    if departments:
        body["departments"] = departments
    r = client.post("/api/v1/admin/tenants", json=body, headers=admin_headers())
    assert r.status_code == 201, r.text
    return r.json()


def upload(client: TestClient, api_key: str, filename: str, content: bytes, department: str, wait: bool = True):
    return client.post(
        "/api/v1/documents/upload",
        params={"wait": str(wait).lower()},
        files={"file": (filename, content, "application/octet-stream")},
        data={"department": department},
        headers={"X-API-Key": api_key},
    )


SAMPLE_DIR = ROOT / "data" / "sample_docs"
SECRET_B = "GIZLI-B-777"
B_HR_DOC = (
    "# Rakip A.Ş. İzin Yönetmeliği\n\n"
    f"Rakip A.Ş. çalışanlarına yılda 30 iş günü yıllık izin verilir. Kurum içi kod: {SECRET_B}.\n\n"
    "Uzaktan çalışma haftada dört gün mümkündür ve otopark ücreti aylık 900 TL'dir."
).encode()


@pytest.fixture(scope="session")
def tenant_a(client: TestClient) -> dict:
    """Örnek Holding: İK yönetmeliği + Finans prosedürü yüklü."""
    t = create_tenant(client, "ornek-holding", "Örnek Holding A.Ş.")
    for name, dept in [
        ("IK_Izin_ve_Calisma_Yonetmeligi.md", "İnsan Kaynakları"),
        ("Finans_Harcama_Proseduru.md", "Finans"),
    ]:
        r = upload(client, t["api_key"], name, (SAMPLE_DIR / name).read_bytes(), dept)
        assert r.status_code == 201, r.text
        t.setdefault("docs", {})[name] = r.json()["document"]
    return t


@pytest.fixture(scope="session")
def tenant_b(client: TestClient) -> dict:
    """Rakip A.Ş.: farklı (çelişen) bir İK belgesi + gizli bir kod içerir."""
    t = create_tenant(client, "rakip-as", "Rakip A.Ş.")
    r = upload(client, t["api_key"], "Rakip_IK.md", B_HR_DOC, "İnsan Kaynakları")
    assert r.status_code == 201, r.text
    t["docs"] = {"Rakip_IK.md": r.json()["document"]}
    return t
