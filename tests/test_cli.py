from typer.testing import CliRunner

from ad_compliance import __version__
from ad_compliance.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "under-18s" in result.stdout


def test_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "LanceDB" in result.stdout


def test_eval_help() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "synthetic" in result.stdout.lower() or "retrieval" in result.stdout.lower()


def test_check_help() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "verdict" in result.stdout.lower() or "guidance" in result.stdout.lower()
