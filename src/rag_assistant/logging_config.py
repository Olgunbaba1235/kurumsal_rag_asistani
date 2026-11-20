"""Merkezi loglama kurulumu."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Gürültülü üçüncü parti logları kıs
    for noisy in ("httpx", "chromadb", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
