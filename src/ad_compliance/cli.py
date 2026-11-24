"""Command-line entry point. Subcommands are added as the project grows."""

from __future__ import annotations

from pathlib import Path

import typer

from ad_compliance import __version__

app = typer.Typer(
    name="ad-compliance",
    help="Check gambling ad copy against UK CAP/BCAP under-18s advertising guidance.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def _main() -> None:
    """Check gambling ad copy against UK CAP/BCAP under-18s advertising guidance."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def ingest(
    pdf: Path | None = typer.Option(
        None,
        help="Path to the CAP/BCAP guidance PDF. Defaults to the bundled copy in data/.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    rebuild: bool = typer.Option(
        False,
        help="Drop the LanceDB table before ingesting.",
    ),
    mock_embeddings: bool = typer.Option(
        False,
        help="Use deterministic dummy embeddings (for tests; not for real retrieval).",
    ),
) -> None:
    """Chunk the guidance PDF and upsert it into the local LanceDB store."""
    from ad_compliance.ingest import ingest_guidance

    result = ingest_guidance(pdf, rebuild=rebuild, mock_embeddings=mock_embeddings)
    typer.echo(
        f"Ingested {result.chunks} chunks from {result.pdf_path.name} "
        f"into {result.lancedb_uri}/{result.table} ({result.rows} rows)."
    )
