"""Demo için tablo içeren örnek hizmet sözleşmesi PDF'ini üretir.

Çalıştırma (bir kez; çıktı repoya eklenir):
    pip install reportlab
    python scripts/make_sample_contract.py

Türkçe karakterler için DejaVuSans (Linux) veya Arial (Windows) yazı tipi gömülür. İçerik tamamen
kurgusaldır; gerçek bir şirket veya sözleşmeyle ilgisi yoktur.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_docs" / "Hukuk_Hizmet_Sozlesmesi.pdf"
FONTS = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


def _register_fonts() -> None:
    for regular, bold in FONTS:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("TR", regular))
            pdfmetrics.registerFont(TTFont("TR-Bold", bold))
            return
    sys.exit("Türkçe glif içeren bir TTF yazı tipi bulunamadı (DejaVuSans veya Arial).")


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "TR", 9.5),
        ("FONT", (0, 0), (-1, 0), "TR-Bold", 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAFF")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#8A8FB0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build() -> Path:
    _register_fonts()
    h1 = ParagraphStyle("h1", fontName="TR-Bold", fontSize=15, leading=19, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName="TR-Bold", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
    p = ParagraphStyle("p", fontName="TR", fontSize=10, leading=14, spaceAfter=5)
    w = A4[0] - 4 * cm

    story = [
        Paragraph("YAZILIM BAKIM VE DESTEK HİZMET SÖZLEŞMESİ", h1),
        Paragraph("Sözleşme No: HS-2026-014 · Taraflar: Örnek Holding A.Ş. (Müşteri) ile Anka Bilişim Ltd. Şti. "
                  "(Hizmet Sağlayıcı). Bu belge demo amaçlı kurgusal bir örnektir.", p),
        Paragraph("Madde 1 – Konu ve Kapsam", h2),
        Paragraph("Hizmet Sağlayıcı, Müşteri'nin kurumsal kaynak planlama yazılımı için bakım, hata giderme, "
                  "güvenlik güncellemesi ve kullanıcı desteği hizmetlerini bu sözleşmede belirtilen hizmet "
                  "seviyelerine uygun olarak sunar.", p),
        Paragraph("Madde 2 – Sözleşme Süresi ve Yenileme", h2),
        Paragraph("Sözleşme 1 Ocak 2026 tarihinde yürürlüğe girer ve 12 ay süreyle geçerlidir. Taraflardan biri "
                  "süre bitiminden en az 60 gün önce yazılı olarak fesih bildiriminde bulunmazsa sözleşme aynı "
                  "koşullarla 1 yıl daha uzar.", p),
        Paragraph("Madde 3 – Hizmet Seviyeleri (SLA)", h2),
        Paragraph("Arıza kayıtları önceliklerine göre aşağıdaki sürelerde yanıtlanır ve çözülür:", p),
        _table([
            ["Öncelik", "Tanım", "İlk yanıt süresi", "Çözüm süresi"],
            ["P1 – Kritik", "Sistem tamamen kullanılamıyor", "30 dakika", "4 saat"],
            ["P2 – Yüksek", "Ana modüllerden biri çalışmıyor", "2 saat", "1 iş günü"],
            ["P3 – Orta", "Geçici çözümü olan hata", "1 iş günü", "5 iş günü"],
            ["P4 – Düşük", "Bilgi talebi / iyileştirme", "2 iş günü", "Sonraki sürüm"],
        ], [w * 0.18, w * 0.38, w * 0.22, w * 0.22]),
        Spacer(1, 6),
        Paragraph("Destek hattı hafta içi 08:30–18:30 arasında hizmet verir; P1 kayıtları için 7/24 nöbetçi "
                  "mühendis bulunur.", p),
        PageBreak(),
        Paragraph("Madde 4 – Ücret ve Ödeme Koşulları", h2),
        _table([
            ["Kalem", "Tutar (KDV hariç)", "Ödeme dönemi"],
            ["Yıllık bakım ve destek bedeli", "480.000 TL", "Üç aylık eşit taksit"],
            ["Ek geliştirme (adam/gün)", "9.500 TL", "Aylık, iş bitiminde"],
            ["Yerinde destek ziyareti", "6.000 TL / ziyaret", "Aylık"],
        ], [w * 0.42, w * 0.28, w * 0.30]),
        Spacer(1, 6),
        Paragraph("Faturalar düzenlendikleri tarihten itibaren 30 gün içinde ödenir. Geciken ödemeler için aylık "
                  "%2 gecikme faizi uygulanır.", p),
        Paragraph("Madde 5 – Hizmet Kredisi ve Cezai Şart", h2),
        Paragraph("Bir takvim ayında P1 çözüm süresinin aşıldığı her olay için Müşteri'ye o ayın bakım bedelinin "
                  "%5'i oranında hizmet kredisi tanınır. Aylık toplam hizmet kredisi, aylık bakım bedelinin %20'sini "
                  "aşamaz.", p),
        Paragraph("Madde 6 – Gizlilik ve Kişisel Veriler", h2),
        Paragraph("Hizmet Sağlayıcı, hizmet sırasında eriştiği ticari sırları ve kişisel verileri yalnızca bu "
                  "sözleşmenin ifası için kullanır ve 6698 sayılı KVKK'ya uygun olarak korur. Gizlilik yükümlülüğü "
                  "sözleşmenin sona ermesinden sonra 3 yıl devam eder.", p),
        Paragraph("Madde 7 – Fesih", h2),
        Paragraph("Taraflardan biri esaslı bir yükümlülüğünü ihlal eder ve yazılı ihtardan itibaren 15 gün içinde "
                  "ihlali gidermezse diğer taraf sözleşmeyi derhal feshedebilir.", p),
        Paragraph("Madde 8 – Uyuşmazlıkların Çözümü", h2),
        Paragraph("Bu sözleşmeden doğan uyuşmazlıklarda İstanbul (Çağlayan) Mahkemeleri ve İcra Daireleri "
                  "yetkilidir.", p),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title="Yazılım Bakım ve Destek Hizmet Sözleşmesi (örnek)",
                      author="Kurumsal RAG Asistanı demo").build(story)
    return OUT


if __name__ == "__main__":
    print(build())
