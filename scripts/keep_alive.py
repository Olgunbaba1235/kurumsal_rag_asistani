"""Streamlit Community Cloud demosunu uyanık tutar (GitHub Actions'tan zamanlanmış çalışır).

Community Cloud, bir süre ziyaret almayan uygulamayı uyku moduna alır; müşteri linke tıkladığında
"This app has gone to sleep" ekranı ve uzun bir açılış bekler. Bu betik gerçek bir tarayıcıyla
uygulamayı açar (yalın HTTP isteği oturum sayılmaz), uyku ekranı varsa "uyandır" butonuna basar ve
uygulama arayüzü yüklenene kadar bekler.

Yerelde: pip install playwright && playwright install chromium && python scripts/keep_alive.py
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://kurumsal-rag-pro.streamlit.app/")
WAKE_TEXTS = ("Yes, get this app back up", "get this app back up")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
        for text in WAKE_TEXTS:
            button = page.get_by_role("button", name=text)
            if button.count():
                print("Uygulama uykudaydı; uyandırılıyor…")
                button.first.click()
                break
        try:
            # Uygulama Community Cloud'da bir iframe içinde çalışır
            frame = page.frame_locator("iframe[title='streamlitApp']")
            frame.get_by_text("Kurumsal bilgiye saniyeler içinde").wait_for(timeout=180_000)
            print("Uygulama ayakta:", APP_URL)
            return 0
        except PlaywrightTimeout:
            print("Uygulama zamanında yüklenmedi:", APP_URL, file=sys.stderr)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
