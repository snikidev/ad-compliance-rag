from __future__ import annotations

from pathlib import Path

import pytest

from ad_compliance.config import Settings


@pytest.fixture
def guidance_pdf() -> Path:
    from ad_compliance.chunking import default_pdf_path

    return default_pdf_path()


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Point at the real bundled PDF without copying 600KB into every tmp dir
    # unless a test asks for a local data_dir of its own.
    return Settings(
        _env_file=None,
        lancedb_uri=tmp_path / ".lancedb",
        data_dir=data_dir,
        embed_provider="fastembed",
    )
