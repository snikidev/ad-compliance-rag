"""Command-line entry point. Subcommands are added as the project grows."""

from __future__ import annotations

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
