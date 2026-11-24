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


@app.command("debug-retrieve")
def debug_retrieve(
    query: str = typer.Argument(..., help="Ad copy or a free-text retrieval query."),
    top_k: int | None = typer.Option(None, help="Override AD_COMPLIANCE_TOP_K."),
    query_type: str = typer.Option(
        "hybrid",
        help="hybrid (default), vector, or fts.",
    ),
    mock_embeddings: bool = typer.Option(
        False,
        help="Use dummy embeddings (only useful if the index was also ingested that way).",
    ),
) -> None:
    """Show the guidance passages a query would retrieve, with scores and citations."""
    from ad_compliance.config import get_settings
    from ad_compliance.retrieve import format_debug_report, retrieve

    settings = get_settings()
    if top_k is not None:
        settings.top_k = top_k
    hits = retrieve(
        query,
        settings=settings,
        mock_embeddings=mock_embeddings,
        top_k=top_k,
        query_type=query_type,
    )
    typer.echo(format_debug_report(query, hits))


@app.command()
def check(
    ad_text: str | None = typer.Argument(
        None,
        help="The advertisement copy to review. Omit and use --file to read a document.",
    ),
    file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Read ad copy from a text file.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the ComplianceReport as JSON.",
    ),
    mock_embeddings: bool = typer.Option(
        False,
        help="Use dummy embeddings (only if the index was ingested that way).",
    ),
) -> None:
    """Retrieve relevant guidance and produce a structured compliance verdict."""
    from ad_compliance.config import get_settings
    from ad_compliance.verdict import check_ad, format_report

    text = ad_text
    if file is not None:
        text = file.read_text(encoding="utf-8")
    if not text or not text.strip():
        raise typer.BadParameter("Provide ad copy as an argument or via --file.")

    report = check_ad(text, settings=get_settings(), mock_embeddings=mock_embeddings)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(format_report(report))
