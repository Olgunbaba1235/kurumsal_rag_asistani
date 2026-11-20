"""B2B toptan satış / üretim firması için gerçekçi, tekrarlanabilir demo verisi.

* ~30 ürün (6 kategori), ~40 müşteri (4 segment, 7 bölge), 2025-01 → 2026-10
  arası siparişler (mevsimsellik + büyüme trendi), 1-5 kalemli siparişler.
* Segmente göre iskonto ve sipariş hacmi; bazı ürünler kritik stok altında;
  eski siparişler çoğunlukla teslim edilmiş, yeniler hazırlanıyor/sevk edildi.
* Tohum (seed) tenant kimliğinden türetilir → aynı tenant için aynı veri,
  farklı tenant'lar için farklı veri (izolasyon testleri için).
* Tüm kayıtlar tenant kapsamlı oturumda yazılır (RLS WITH CHECK).
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import func, insert, select

from app.core.db import Database
from app.models.sql_models import Customer, Order, OrderItem, Product
from app.services.tenant_service import TenantContext

_D2 = Decimal("0.01")

PRODUCTS: list[tuple[str, str, str, float]] = [
    # (kategori, ad, birim, liste fiyatı TRY, KDV hariç)
    ("Bağlantı Elemanları", "M8x40 Galvaniz Cıvata (100'lü)", "paket", 185.0),
    ("Bağlantı Elemanları", "M10 Paslanmaz Somun (100'lü)", "paket", 240.0),
    ("Bağlantı Elemanları", "Çelik Dübel 10x80 (50'li)", "paket", 310.0),
    ("Bağlantı Elemanları", "Kimyasal Ankraj 300 ml", "adet", 420.0),
    ("Bağlantı Elemanları", "Rondela M12 (500'lü)", "paket", 155.0),
    ("Hidrolik & Pnömatik", "Hidrolik Hortum 1/2\" 2SN", "m", 135.0),
    ("Hidrolik & Pnömatik", "Pnömatik Silindir 50x100", "adet", 2850.0),
    ("Hidrolik & Pnömatik", "Hidrolik Pompa 16 cc", "adet", 9800.0),
    ("Hidrolik & Pnömatik", "Hızlı Kaplin 1/4\"", "adet", 165.0),
    ("Hidrolik & Pnömatik", "Şartlandırıcı 1/2\"", "adet", 1450.0),
    ("Elektrik Malzemeleri", "NYM Kablo 3x2,5 mm²", "m", 38.0),
    ("Elektrik Malzemeleri", "Otomatik Sigorta 16A C Tipi", "adet", 95.0),
    ("Elektrik Malzemeleri", "Kontaktör 25A 220V", "adet", 780.0),
    ("Elektrik Malzemeleri", "LED Projektör 100W", "adet", 1150.0),
    ("Elektrik Malzemeleri", "Pano Tipi Ölçü Aleti", "adet", 2300.0),
    ("Ambalaj", "Streç Film 17 mic (17 kg)", "rulo", 1650.0),
    ("Ambalaj", "Koli 40x30x30 Çift Oluklu", "adet", 21.0),
    ("Ambalaj", "Koli Bandı 45 mm (36'lı)", "koli", 690.0),
    ("Ambalaj", "Palet Çemberi 12 mm", "rulo", 980.0),
    ("Kaynak & Sarf", "Gazaltı Kaynak Teli 1,0 mm (15 kg)", "makara", 1350.0),
    ("Kaynak & Sarf", "Kesme Diski 115 mm (25'li)", "kutu", 475.0),
    ("Kaynak & Sarf", "Taşlama Diski 180 mm (10'lu)", "kutu", 520.0),
    ("Kaynak & Sarf", "Rutil Elektrod 3,25 mm (5 kg)", "paket", 610.0),
    ("Kaynak & Sarf", "Argon Gazı Tüp Dolumu", "adet", 1850.0),
    ("İş Güvenliği", "Nitril Kaplı İş Eldiveni (12'li)", "paket", 260.0),
    ("İş Güvenliği", "Baret EN 397", "adet", 185.0),
    ("İş Güvenliği", "Çelik Burunlu İş Ayakkabısı S3", "çift", 1250.0),
    ("İş Güvenliği", "Reflektörlü Yelek", "adet", 95.0),
    ("İş Güvenliği", "Toz Maskesi FFP2 (20'li)", "kutu", 340.0),
    ("İş Güvenliği", "Emniyet Kemeri Tam Vücut", "adet", 2150.0),
]

CITIES: list[tuple[str, str]] = [
    ("İstanbul", "Marmara"), ("Kocaeli", "Marmara"), ("Bursa", "Marmara"), ("Ankara", "İç Anadolu"),
    ("Konya", "İç Anadolu"), ("Kayseri", "İç Anadolu"), ("İzmir", "Ege"), ("Manisa", "Ege"),
    ("Denizli", "Ege"), ("Antalya", "Akdeniz"), ("Adana", "Akdeniz"), ("Mersin", "Akdeniz"),
    ("Gaziantep", "Güneydoğu Anadolu"), ("Samsun", "Karadeniz"), ("Trabzon", "Karadeniz"), ("Erzurum", "Doğu Anadolu"),
]
_NAME_PREFIX = ["Anadolu", "Ege", "Marmara", "Karadeniz", "Başkent", "Toros", "Yıldız", "Kuzey", "Delta", "Atlas",
                "Güney", "Kartal", "Çınar", "Doruk", "Akdeniz", "Meridyen", "Zirve", "Pınar", "Tunç", "Sistem"]
_NAME_SECTOR = ["Yapı Market", "Endüstri", "Makina", "Elektrik", "Ambalaj", "Hırdavat", "İnşaat", "Otomotiv",
                "Mühendislik", "Tedarik"]
# segment: (ağırlık, iskonto aralığı, miktar çarpanı, kredi limiti aralığı, vade seçenekleri)
_SEGMENTS = {
    "Distribütör": (0.2, (0.10, 0.18), 6.0, (1_500_000, 4_000_000), (60, 90)),
    "Bayi": (0.35, (0.05, 0.12), 3.0, (400_000, 1_200_000), (45, 60)),
    "Kurumsal": (0.3, (0.03, 0.08), 2.0, (250_000, 900_000), (30, 45, 60)),
    "Perakende Zinciri": (0.15, (0.08, 0.15), 4.0, (800_000, 2_500_000), (60, 75)),
}
_CHANNELS = {"saha_satis": 0.4, "b2b_portal": 0.35, "telefon": 0.15, "ihale": 0.10}
# Aylık mevsimsellik (inşaat/sanayi: ilkbahar-yaz yüksek, kış düşük)
_SEASON = {1: 0.75, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.15, 7: 1.05, 8: 0.95, 9: 1.1, 10: 1.15, 11: 1.0, 12: 0.85}


@dataclass(slots=True)
class SeedResult:
    created: bool
    customers: int
    products: int
    orders: int
    order_items: int


def _money(x: float | Decimal) -> Decimal:
    return Decimal(str(x)).quantize(_D2, rounding=ROUND_HALF_UP)


def _weighted(rng: random.Random, options: dict[str, float]) -> str:
    return rng.choices(list(options), weights=list(options.values()))[0]


async def _counts(db: Database, tenant: TenantContext) -> SeedResult:
    async with db.tenant_session(tenant.id) as s:
        counts = []
        for model in (Customer, Product, Order, OrderItem):
            counts.append(int(await s.scalar(select(func.count()).select_from(model)) or 0))
    return SeedResult(False, *counts)


async def seed_demo_sales(
    db: Database,
    tenant: TenantContext,
    *,
    n_customers: int = 40,
    n_orders: int = 450,
    start: dt.date = dt.date(2025, 1, 1),
    end: dt.date = dt.date(2026, 10, 31),
) -> SeedResult:
    existing = await _counts(db, tenant)
    if existing.customers:
        return existing  # idempotent: veri varsa dokunma

    rng = random.Random(tenant.id.int)

    async with db.tenant_session(tenant.id) as s, s.begin():
        # ---- ürünler
        product_rows = []
        for i, (category, name, unit, price) in enumerate(PRODUCTS, start=1):
            price *= rng.uniform(0.92, 1.08)
            reorder = rng.choice([20, 50, 100, 200, 500])
            stock = int(reorder * rng.uniform(0.3, 6.0))  # bazıları kritik seviyenin altında
            product_rows.append(
                {
                    "tenant_id": tenant.id,
                    "sku": f"{category[:3].upper().replace('İ', 'I').replace('Ğ', 'G')}-{i:04d}",
                    "name": name,
                    "category": category,
                    "unit": unit,
                    "unit_price": _money(price),
                    "unit_cost": _money(price * rng.uniform(0.55, 0.78)),
                    "stock_qty": stock,
                    "reorder_level": reorder,
                    "is_active": rng.random() > 0.05,
                }
            )
        products = (await s.execute(insert(Product).returning(Product.id, Product.unit_price, sort_by_parameter_order=True), product_rows)).all()

        # ---- müşteriler
        names: set[str] = set()
        while len(names) < n_customers:
            suffix = rng.choice(["A.Ş.", "Ltd. Şti.", "San. ve Tic. A.Ş."])
            names.add(f"{rng.choice(_NAME_PREFIX)} {rng.choice(_NAME_SECTOR)} {suffix}")
        customer_rows, segments = [], []
        for i, name in enumerate(sorted(names), start=1):
            segment = rng.choices(list(_SEGMENTS), weights=[v[0] for v in _SEGMENTS.values()])[0]
            _, _, _, (lo, hi), terms = _SEGMENTS[segment]
            city, region = rng.choice(CITIES)
            segments.append(segment)
            customer_rows.append(
                {
                    "tenant_id": tenant.id,
                    "code": f"C{i:05d}",
                    "name": name,
                    "segment": segment,
                    "city": city,
                    "region": region,
                    "credit_limit": _money(round(rng.uniform(lo, hi), -4)),
                    "payment_terms_days": rng.choice(terms),
                    "since": start - dt.timedelta(days=rng.randint(30, 3000)),
                    "is_active": rng.random() > 0.07,
                }
            )
        customers = (await s.execute(insert(Customer).returning(Customer.id, sort_by_parameter_order=True), customer_rows)).scalars().all()
        # Pareto: müşterilerin küçük bir kısmı cironun büyük kısmını yapar
        customer_weights = [rng.paretovariate(1.3) for _ in customers]

        # ---- siparişler ve kalemler
        span_days = (end - start).days
        dates = []
        while len(dates) < n_orders:
            d = start + dt.timedelta(days=rng.randint(0, span_days))
            growth = 1.0 + 0.35 * (d - start).days / span_days  # yıllık ~%20 büyüme
            if d.weekday() < 5 and rng.random() < _SEASON[d.month] * growth / 1.6:
                dates.append(d)
        dates.sort()

        order_rows, item_plan = [], []
        for i, order_date in enumerate(dates, start=1):
            ci = rng.choices(range(len(customers)), weights=customer_weights)[0]
            segment = segments[ci]
            _, (dlo, dhi), qty_mult, _, _ = _SEGMENTS[segment]
            discount = _money(rng.uniform(dlo, dhi) * 100)
            age = (end - order_date).days
            if rng.random() < 0.05:
                status = "iptal"
            elif age > 21:
                status = "teslim_edildi"
            elif age > 7:
                status = rng.choice(["sevk_edildi", "teslim_edildi"])
            else:
                status = rng.choice(["hazirlaniyor", "sevk_edildi"])
            shipped = None if status in ("hazirlaniyor", "iptal") else order_date + dt.timedelta(days=rng.randint(1, 6))

            lines = []
            for pi in rng.sample(range(len(products)), k=rng.randint(1, 5)):
                pid, list_price = products[pi]
                qty = max(1, int(rng.lognormvariate(1.6, 0.8) * qty_mult))
                unit_price = _money(Decimal(list_price) * (1 - discount / 100))
                lines.append((pid, qty, unit_price, _money(unit_price * qty)))
            total = sum((ln[3] for ln in lines), Decimal("0"))
            order_rows.append(
                {
                    "tenant_id": tenant.id,
                    "order_no": f"SIP-{order_date:%Y%m}-{i:05d}",
                    "customer_id": customers[ci],
                    "order_date": order_date,
                    "shipped_date": shipped,
                    "status": status,
                    "channel": _weighted(rng, _CHANNELS),
                    "discount_rate": discount,
                    "total_amount": total,
                }
            )
            item_plan.append(lines)

        order_ids = (await s.execute(insert(Order).returning(Order.id, sort_by_parameter_order=True), order_rows)).scalars().all()
        item_rows = [
            {
                "tenant_id": tenant.id,
                "order_id": oid,
                "product_id": pid,
                "quantity": qty,
                "unit_price": up,
                "line_total": lt,
            }
            for oid, lines in zip(order_ids, item_plan, strict=True)
            for pid, qty, up, lt in lines
        ]
        await s.execute(insert(OrderItem), item_rows)

    result = await _counts(db, tenant)
    result.created = True
    return result
