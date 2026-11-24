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


@app.command("eval")
def evaluate(
    dataset: Path | None = typer.Option(
        None,
        help="Path to a snippets JSON file. Defaults to the bundled evals/snippets.json.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    run_verdicts: bool = typer.Option(
        False,
        "--verdicts/--retrieval-only",
        help="Run the LLM judge as well as retrieval. Requires a configured provider key.",
    ),
    mock_embeddings: bool = typer.Option(
        False,
        help="Use dummy embeddings (FTS still runs). Useful in tests.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
) -> None:
    """Score retrieval (and optionally verdicts) against the synthetic ad set."""
    from ad_compliance.config import get_settings
    from ad_compliance.evaluate import format_eval_result, run_eval

    result = run_eval(
        settings=get_settings(),
        dataset_path=dataset,
        mock_embeddings=mock_embeddings,
        run_verdicts=run_verdicts,
    )
    if json_output:
        import json

        typer.echo(json.dumps(result.as_dict(), indent=2))
    else:
        typer.echo(format_eval_result(result))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("ad_compliance.api:app", host=host, port=port, reload=False)


@app.command()
def demo(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8501),
) -> None:
    """Run the Streamlit demo."""
    import sys
    from pathlib import Path

    from streamlit.web import cli as stcli

    script = Path(__file__).with_name("demo.py")
    sys.argv = [
        "streamlit",
        "run",
        str(script),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    stcli.main()


@app.command("mcp")
def mcp_command() -> None:
    """Run the FastMCP server on stdio (for Claude Code, Cursor, or OpenCode)."""
    from ad_compliance.mcp_server import mcp

    mcp.run()
