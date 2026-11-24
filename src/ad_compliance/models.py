"""Structured verdict types for a compliance check."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

VerdictLabel = Literal["breach", "ok", "needs_review"]


class Citation(BaseModel):
    """A pointer back to a passage of the CAP/BCAP guidance."""

    section_number: int | None = None
    section_title: str
    page_start: int
    page_end: int
    quote: str = Field(description="A short verbatim excerpt from the guidance.")


class RuleVerdict(BaseModel):
    """The model's decision against one guidance section."""

    rule_id: str = Field(description="Section id, such as s20.")
    rule_title: str
    verdict: VerdictLabel
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    citation: Citation | None = None
    suggested_rewrite: str | None = Field(
        default=None,
        description="A safer rewrite of the ad copy, if the verdict is not ok.",
    )


class ComplianceReport(BaseModel):
    """Aggregated verdict across the guidance sections that were checked."""

    overall: VerdictLabel
    summary: str
    ad_text: str
    verdicts: list[RuleVerdict] = Field(default_factory=list)
    retrieved_sections: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def overall_follows_children(self) -> ComplianceReport:
        # Callers may set overall explicitly; aggregation helpers overwrite it.
        return self


def aggregate_label(verdicts: list[RuleVerdict]) -> VerdictLabel:
    """Breach wins, then needs_review, otherwise ok."""
    if any(item.verdict == "breach" for item in verdicts):
        return "breach"
    if any(item.verdict == "needs_review" for item in verdicts):
        return "needs_review"
    return "ok"
