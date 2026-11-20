"""Docker/Postgres olmadan yerel geliştirme sunucusu: PGlite + tek süreçli uvicorn (--factory).

Kullanım:
    npm install --prefix tests/pglite        # bir kez
    python scripts/dev_pglite.py             # http://localhost:8000/docs
    python scripts/dev_pglite.py --port 8001 --no-seed

Ne yapar?
* ``tests/pglite/server.mjs`` ile bellek içi PostgreSQL + pgvector başlatır (alt süreç).
* uvicorn'u ``--factory`` modunda *aynı süreçte* çalıştırır (reload yok): PGlite tek backend
  olduğu için ikinci bir bağlantı (ör. ayrı ``alembic`` süreci) hazır-ifade çakışmasına yol açar;
  bu yüzden migration, uygulamanın kendi engine'i üzerinden açılışta çalıştırılır.
* ``--seed`` (varsayılan): "dev-sirket" tenant'ı, örnek dokümanlar ve demo satış verisi oluşturur;
  API anahtarını terminale yazar (Streamlit kenar çubuğuna yapıştırın).

Uyarı: veri bellektedir, süreç kapanınca silinir. Yalnızca geliştirme içindir.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PGLITE_DIR = ROOT / "tests" / "pglite"
SAMPLE_DIR = ROOT / "data" / "sample_docs"
SAMPLE_DEPARTMENTS = {
    "IK_Izin_ve_Calisma_Yonetmeligi.md": "İnsan Kaynakları",
    "Finans_Harcama_Proseduru.md": "Finans",
}
logger = logging.getLogger("dev_pglite")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_pglite(timeout: float = 60.0) -> tuple[subprocess.Popen, int]:
    node = shutil.which("node")
    if node is None or not (PGLITE_DIR / "node_modules").exists():
        sys.exit("Node.js ve `npm install --prefix tests/pglite` gerekli.")
    port = _free_port()
    proc = subprocess.Popen(
        [node, "server.mjs", str(port)], cwd=PGLITE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("READY"):
            return proc, port
        if proc.poll() is not None:
            sys.exit(f"PGlite başlatılamadı: {line}{proc.stdout.read()}")
    proc.kill()
    sys.exit(f"PGlite {timeout:.0f} sn içinde hazır olmadı.")


def _run_migrations(sync_conn) -> None:  # noqa: ANN001
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["connection"] = sync_conn
    command.upgrade(cfg, "head")


async def _seed(container) -> None:  # noqa: ANN001
    from app.schemas.tenant import TenantCreate
    from app.services.demo_data import seed_demo_sales

    tenant, api_key = await container.tenants.create(TenantCreate(name="Dev Şirket A.Ş.", slug="dev-sirket"))
    ctx = await container.tenants.authenticate(api_key)
    for name, dept in SAMPLE_DEPARTMENTS.items():
        path = SAMPLE_DIR / name
        if not path.exists():
            continue
        data = path.read_bytes()
        doc, _ = await container.documents.register_upload(ctx, dept, name, data)
        await container.documents.process(ctx, doc.id, data)
    await seed_demo_sales(container.db, ctx)
    banner = f"  Tenant: {tenant.slug}   API anahtarı (X-API-Key): {api_key}  "
    print("\n" + "=" * len(banner) + f"\n{banner}\n" + "=" * len(banner) + "\n", flush=True)


def create_dev_app():  # noqa: ANN201
    """uvicorn ``--factory`` girişi: normal uygulama + açılışta migration (ve isteğe bağlı seed)."""
    from app.main import create_app

    app = create_app()
    original = app.router.lifespan_context
    seed = os.environ.get("DEV_PGLITE_SEED") == "1"

    @asynccontextmanager
    async def lifespan(a):  # noqa: ANN001, ANN202
        async with original(a):
            container = a.state.container
            async with container.db.engine.begin() as conn:
                await conn.run_sync(_run_migrations)
            logger.info("Migration'lar uygulandı (alembic upgrade head).")
            if seed:
                await _seed(container)
            yield

    app.router.lifespan_context = lifespan
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-seed", action="store_true", help="Örnek tenant/doküman/veri oluşturma")
    args = ap.parse_args()

    proc, pg_port = start_pglite()
    env = {
        "DATABASE_URL": f"postgresql+asyncpg://postgres:postgres@127.0.0.1:{pg_port}/postgres?ssl=disable",
        "DB_POOL_SIZE": "1",
        "DB_MAX_OVERFLOW": "0",
        "DB_SESSION_ROLE": "rag_app",  # superuser bağlantısında RLS'e tabi role geç
        "DB_DISPOSE_TIMEOUT": "0.5",
        "ADMIN_API_KEY": "dev-admin-key-0123456789",
        "REDIS_URL": "",
        "RATE_LIMIT_PER_MINUTE": "0",
        "QUERY_REWRITING": "false",
        "DEV_PGLITE_SEED": "0" if args.no_seed else "1",
    }
    if not os.environ.get("GOOGLE_API_KEY"):  # anahtar yoksa çevrimdışı sağlayıcılar
        env.update(EMBEDDING_PROVIDER="hashing", LLM_PROVIDER="none")
    os.environ.update({k: v for k, v in env.items() if k not in os.environ or k.startswith(("DB_", "DATABASE"))})
    sys.path.insert(0, str(ROOT))

    print(f"PGlite hazır (127.0.0.1:{pg_port}). API: http://{args.host}:{args.port}/docs", flush=True)
    try:
        import uvicorn

        uvicorn.run("scripts.dev_pglite:create_dev_app", factory=True, host=args.host, port=args.port, reload=False)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
