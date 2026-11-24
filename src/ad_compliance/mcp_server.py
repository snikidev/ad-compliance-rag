"""FastMCP server exposing a check_ad tool."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ad_compliance.verdict import check_ad as run_check

mcp = FastMCP(
    "ad-compliance",
    instructions=(
        "Review UK gambling advertising copy against CAP/BCAP guidance on "
        "protecting under-18s. This is a technical demonstration, not legal advice. "
        "The guidance is from CAP/ASA, not the Gambling Commission."
    ),
)


@mcp.tool
def check_ad(ad_text: str) -> dict[str, Any]:
    """Review gambling ad copy against UK CAP/BCAP under-18s advertising guidance.

    Args:
        ad_text: The advertisement copy, script, or post to review.

    Returns:
        A structured compliance report with an overall verdict, per-rule
        decisions, citations, and suggested rewrites. Not legal advice.
    """
    report = run_check(ad_text)
    return report.model_dump()
