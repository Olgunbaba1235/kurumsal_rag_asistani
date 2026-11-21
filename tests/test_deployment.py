"""Streamlit Community Cloud dağıtım güvenceleri.

Bulut ortamı uygulamayı farklı bir çalışma dizininden başlatabilir, yalnızca `requirements.txt`'i kurar ve
repodaki dosyaları görür. Bu testler o koşulları yerelde taklit eder.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"

# import adı → PyPI dağıtım adı
DIST = {
    "streamlit": "streamlit", "plotly": "plotly", "numpy": "numpy", "rank_bm25": "rank-bm25",
    "langchain_text_splitters": "langchain-text-splitters", "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings", "pdfplumber": "pdfplumber", "pypdf": "pypdf",
    "docx": "python-docx", "openpyxl": "openpyxl", "google": "google-genai",
}
# Vitrin uygulamasının yüklemediği (yalnızca API / eski arayüz) modüller
NOT_IN_SHOWCASE = {"vector_store.py"}
SERVER_ONLY = ("fastapi", "uvicorn", "asyncpg", "sqlalchemy", "alembic", "redis", "chromadb", "pgvector")


def _requirements() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[ ;]", line, maxsplit=1)[0].lower())
    return names


def _third_party_imports(paths: list[Path]) -> set[str]:
    found = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    local = {"src", "ui", "app", "rag_assistant", "__future__"}
    return {m for m in found if m not in sys.stdlib_module_names and m not in local}


def test_requirements_cover_every_showcase_import():
    files = [APP, *sorted((ROOT / "ui").glob("showcase_*.py"))]
    files += [p for p in sorted((ROOT / "src").glob("*.py")) if p.name not in NOT_IN_SHOWCASE]
    missing = {m for m in _third_party_imports(files) if DIST.get(m, m).lower() not in _requirements()}
    assert not missing, f"requirements.txt içinde eksik: {sorted(missing)}"


def test_cloud_requirements_stay_lean():
    reqs = _requirements()
    assert not [p for p in SERVER_ONLY if p in reqs], "sunucu paketleri requirements-dev.txt'e ait"
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements-api.txt" in dev and "-r requirements.txt" in dev


def test_secrets_and_local_caches_are_git_ignored():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for rule in (".env", ".streamlit/secrets.toml", "*.sqlite3", "*.db", "data/chroma/*", "data/uploads/*"):
        assert rule in rules, rule


def test_bundled_data_files_exist():
    from src.config import Settings

    sample_dir = Settings(_env_file=None).sample_docs_dir
    assert sample_dir.is_absolute() and sample_dir.is_relative_to(ROOT)
    assert sorted(p.suffix for p in sample_dir.iterdir()) == [".md", ".md", ".pdf"]


def test_app_runs_from_any_working_directory(tmp_path, monkeypatch):
    """Bulut uygulamayı repo kökü dışından başlatsa bile veri dosyaları bulunur, demo çalışır."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        import streamlit.dataframe_util as du

        monkeypatch.setattr(du, "convert_anything_to_arrow_bytes", lambda *a, **k: b"")
    monkeypatch.chdir(tmp_path)
    at = st_testing.AppTest.from_file(str(APP), default_timeout=90)
    at.session_state["stream_delay"] = 0.0
    at.session_state["demo_delay"] = 0.0
    at.run()
    assert not at.exception, at.exception
    assert os.getcwd() == str(tmp_path)
    ws = next(iter(at.session_state["workspaces"].values()))
    assert len(ws.documents) == 3
    next(b for b in at.button if b.label.startswith("▶")).click().run()
    assert not at.exception, at.exception
    assert len(next(iter(at.session_state["chats"].values()))) == 12
