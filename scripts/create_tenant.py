"""Komut satırından tenant oluşturur ve API anahtarını bir kez yazdırır.

    python scripts/create_tenant.py "Örnek Holding A.Ş." ornek-holding
    python scripts/create_tenant.py "Beta Ltd." beta --departments "Genel" "Finans"

Docker içinde:
    docker compose exec api python scripts/create_tenant.py "Örnek Holding" ornek-holding
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_api_settings  # noqa: E402
from app.core.db import Database  # noqa: E402
from app.schemas.tenant import TenantCreate  # noqa: E402
from app.services.tenant_service import TenantConflictError, TenantService  # noqa: E402
from src.config import DEFAULT_DEPARTMENTS  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Yeni tenant oluştur")
    parser.add_argument("name")
    parser.add_argument("slug")
    parser.add_argument("--departments", nargs="+", default=list(DEFAULT_DEPARTMENTS))
    args = parser.parse_args()

    db = Database(get_api_settings())
    try:
        tenant, api_key = await TenantService(db).create(
            TenantCreate(name=args.name, slug=args.slug, departments=args.departments)
        )
    except TenantConflictError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    finally:
        await db.dispose()

    print(f"Tenant oluşturuldu: {tenant.name} ({tenant.slug}) id={tenant.id}")
    print(f"Departmanlar     : {', '.join(tenant.departments)}")
    print(f"API anahtarı     : {api_key}")
    print("Bu anahtar yalnızca şimdi gösterilir; güvenli bir yerde saklayın.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
