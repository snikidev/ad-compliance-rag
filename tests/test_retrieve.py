from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ad_compliance.chunking import default_pdf_path
from ad_compliance.cli import app
from ad_compliance.config import Settings
from ad_compliance.ingest import ingest_guidance
from ad_compliance.retrieve import EmptyIndexError, retrieve


def test_retrieve_before_ingest_raises(tmp_settings: Settings) -> None:
    with pytest.raises(EmptyIndexError):
        retrieve("cartoon pirates", settings=tmp_settings, mock_embeddings=True)


def test_hybrid_retrieve_finds_keyword_section(tmp_settings: Settings) -> None:
    ingest_guidance(default_pdf_path(), settings=tmp_settings, mock_embeddings=True)
    hits = retrieve(
        "loot boxes and skins in video game style ads",
        settings=tmp_settings,
        mock_embeddings=True,
        query_type="hybrid",
    )
    assert hits
    blob = " ".join(hit.text.lower() + " " + hit.section_title.lower() for hit in hits)
    assert "loot" in blob or "video game" in blob or "skins" in blob
    assert all(hit.page_start >= 1 for hit in hits)
    assert any(hit.section_title for hit in hits)


def test_fts_retrieve_finds_santa(tmp_settings: Settings) -> None:
    ingest_guidance(default_pdf_path(), settings=tmp_settings, mock_embeddings=True)
    hits = retrieve(
        "Santa Claus and the Easter Bunny",
        settings=tmp_settings,
        mock_embeddings=True,
        query_type="fts",
    )
    assert hits
    blob = " ".join(hit.text for hit in hits)
    assert "Santa" in blob or "Easter" in blob


def test_debug_retrieve_cli(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_guidance(default_pdf_path(), settings=tmp_settings, mock_embeddings=True)
    monkeypatch.setenv("AD_COMPLIANCE_LANCEDB_URI", str(tmp_settings.lancedb_uri))
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["debug-retrieve", "--mock-embeddings", "Santa Claus cartoon"],
    )
    assert result.exit_code == 0, result.stdout + str(result.exception)
    assert "hit" in result.stdout.lower()
