"""LlamaIndex workflow: retrieve guidance, check each rule, aggregate a report."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from ad_compliance.config import Settings, get_settings
from ad_compliance.models import (
    Citation,
    ComplianceReport,
    RuleVerdict,
    aggregate_label,
)
from ad_compliance.providers import build_llm
from ad_compliance.retrieve import RetrievedChunk, retrieve

RuleChecker = Callable[[str, str, list[RetrievedChunk]], RuleVerdict]


class CheckInput(StartEvent):
    ad_text: str


class RetrievedEvent(Event):
    ad_text: str
    hits: list[RetrievedChunk]


class VerdictsEvent(Event):
    ad_text: str
    hits: list[RetrievedChunk]
    verdicts: list[RuleVerdict]


class ComplianceWorkflow(Workflow):
    """retrieve → per-rule check → aggregate."""

    def __init__(
        self,
        *,
        settings: Settings,
        embed_model: BaseEmbedding | None = None,
        llm: LLM | None = None,
        checker: RuleChecker | None = None,
        mock_embeddings: bool = False,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.settings = settings
        self.embed_model = embed_model
        self.llm = llm
        self.checker = checker
        self.mock_embeddings = mock_embeddings

    @step
    async def retrieve_passages(self, ev: CheckInput) -> RetrievedEvent:
        hits = retrieve(
            ev.ad_text,
            settings=self.settings,
            embed_model=self.embed_model,
            mock_embeddings=self.mock_embeddings,
        )
        return RetrievedEvent(ad_text=ev.ad_text, hits=hits)

    @step
    async def check_rules(self, ev: RetrievedEvent) -> VerdictsEvent:
        grouped = _group_by_section(ev.hits)
        verdicts: list[RuleVerdict] = []
        checker = self.checker or (
            lambda ad, sid, hits: check_section(ad, sid, hits, llm=self._require_llm())
        )
        for section_id, hits in grouped.items():
            verdicts.append(checker(ev.ad_text, section_id, hits))
        return VerdictsEvent(ad_text=ev.ad_text, hits=ev.hits, verdicts=verdicts)

    @step
    async def aggregate(self, ev: VerdictsEvent) -> StopEvent:
        report = build_report(ev.ad_text, ev.hits, ev.verdicts)
        return StopEvent(result=report)

    def _require_llm(self) -> LLM:
        if self.llm is None:
            self.llm = build_llm(self.settings)
        return self.llm


def check_ad(
    ad_text: str,
    *,
    settings: Settings | None = None,
    embed_model: BaseEmbedding | None = None,
    llm: LLM | None = None,
    checker: RuleChecker | None = None,
    mock_embeddings: bool = False,
) -> ComplianceReport:
    """Run the compliance workflow and return a structured report."""
    if not ad_text or not ad_text.strip():
        raise ValueError("ad_text must not be empty")
    settings = settings or get_settings()
    workflow = ComplianceWorkflow(
        settings=settings,
        embed_model=embed_model,
        llm=llm,
        checker=checker,
        mock_embeddings=mock_embeddings,
    )
    return asyncio.run(_run_workflow(workflow, ad_text.strip()))


def build_report(
    ad_text: str,
    hits: list[RetrievedChunk],
    verdicts: list[RuleVerdict],
) -> ComplianceReport:
    if not hits:
        return ComplianceReport(
            overall="needs_review",
            summary=(
                "No guidance passages were retrieved, so the copy was not assessed. "
                "Run `ad-compliance ingest` and try again."
            ),
            ad_text=ad_text,
            verdicts=[],
            retrieved_sections=[],
        )
    overall = aggregate_label(verdicts) if verdicts else "needs_review"
    sections = []
    for hit in hits:
        label = hit.citation_label()
        if label not in sections:
            sections.append(label)
    summary = _summarise(overall, verdicts)
    return ComplianceReport(
        overall=overall,
        summary=summary,
        ad_text=ad_text,
        verdicts=verdicts,
        retrieved_sections=sections,
    )


def check_section(
    ad_text: str,
    section_id: str,
    hits: list[RetrievedChunk],
    *,
    llm: LLM,
) -> RuleVerdict:
    """Ask the LLM to judge the ad against one guidance section."""
    title = hits[0].section_title if hits else section_id
    number = hits[0].section_number if hits else None
    context = "\n\n".join(_format_hit(hit) for hit in hits)
    prompt = _RULE_PROMPT.format(
        ad_text=ad_text.strip(),
        section_id=section_id,
        section_title=title,
        context=context,
    )
    verdict = _predict_verdict(llm, prompt)
    verdict.rule_id = section_id
    verdict.rule_title = title or verdict.rule_title
    if verdict.citation is None and hits:
        verdict.citation = _fallback_citation(hits[0])
    else:
        verdict.citation = _align_citation(verdict.citation, hits)
    if (
        number is not None
        and verdict.citation is not None
        and verdict.citation.section_number is None
    ):
        verdict.citation.section_number = number
    if verdict.verdict == "ok":
        verdict.suggested_rewrite = None
    return verdict


def format_report(report: ComplianceReport) -> str:
    """Human-readable rendering used by the CLI."""
    lines = [
        f"Overall: {report.overall.upper()}",
        report.summary,
        "",
    ]
    if not report.verdicts:
        return "\n".join(lines).strip() + "\n"
    for item in report.verdicts:
        lines.append(
            f"- [{item.verdict}] {item.rule_id} {item.rule_title} "
            f"(confidence {item.confidence:.2f})"
        )
        lines.append(f"  {item.rationale}")
        if item.citation is not None:
            pages = (
                f"p. {item.citation.page_start}"
                if item.citation.page_start == item.citation.page_end
                else f"pp. {item.citation.page_start}–{item.citation.page_end}"
            )
            lines.append(f"  Citation: {item.citation.section_title} ({pages})")
            lines.append(f"  Quote: {item.citation.quote}")
        if item.suggested_rewrite:
            lines.append(f"  Suggested rewrite: {item.suggested_rewrite}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _group_by_section(hits: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for hit in hits:
        key = hit.section_id or hit.chunk_id
        grouped[key].append(hit)
    return dict(grouped)


def _format_hit(hit: RetrievedChunk) -> str:
    return f"{hit.citation_label()}\n{hit.text}"


def _summarise(overall: str, verdicts: list[RuleVerdict]) -> str:
    if not verdicts:
        return "No per-rule verdicts were produced."
    breaches = [item.rule_id for item in verdicts if item.verdict == "breach"]
    reviews = [item.rule_id for item in verdicts if item.verdict == "needs_review"]
    if overall == "breach":
        return "Likely issues against: " + ", ".join(breaches) + "."
    if overall == "needs_review":
        return "Borderline or incomplete against: " + ", ".join(reviews) + "."
    return f"No issues identified across {len(verdicts)} retrieved guidance section(s)."


def _predict_verdict(llm: LLM, prompt: str) -> RuleVerdict:
    if hasattr(llm, "structured_predict"):
        try:
            result = llm.structured_predict(RuleVerdict, prompt)  # type: ignore[arg-type]
            if isinstance(result, RuleVerdict):
                return result
        except Exception:
            pass
    response = llm.complete(
        prompt + "\n\nRespond with a single JSON object that matches the schema. No markdown."
    )
    text = getattr(response, "text", None) or str(response)
    return RuleVerdict.model_validate(_extract_json(text))


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _fallback_citation(hit: RetrievedChunk) -> Citation:
    quote = " ".join(hit.text.split())
    if len(quote) > 240:
        quote = quote[:239].rstrip() + "…"
    return Citation(
        section_number=hit.section_number,
        section_title=hit.section_title,
        page_start=hit.page_start,
        page_end=hit.page_end,
        quote=quote,
    )


def _align_citation(citation: Citation | None, hits: list[RetrievedChunk]) -> Citation | None:
    """Keep a faithful model quote; otherwise use an excerpt from a retrieved chunk."""
    if citation is None:
        return _fallback_citation(hits[0]) if hits else None
    corpus = " ".join(hit.text for hit in hits)
    if citation.quote and _quote_in_text(citation.quote, corpus):
        # Fill in missing location from the first hit that contains the quote.
        for hit in hits:
            if _quote_in_text(citation.quote, hit.text):
                return Citation(
                    section_number=hit.section_number,
                    section_title=hit.section_title or citation.section_title,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    quote=_trim_quote(citation.quote),
                )
    # Unfaithful or empty quote: fall back to a real excerpt.
    return _fallback_citation(hits[0])


def quote_is_faithful(quote: str, source: str) -> bool:
    return _quote_in_text(quote, source)


def _quote_in_text(quote: str, source: str) -> bool:
    def norm(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    needle, haystack = norm(quote), norm(source)
    if not needle or len(needle) < 12:
        return False
    if needle in haystack:
        return True
    # Allow a slightly shortened quote.
    return bool(len(needle) > 40 and needle[:40] in haystack)


def _trim_quote(quote: str, limit: int = 240) -> str:
    compact = " ".join(quote.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


async def _run_workflow(workflow: ComplianceWorkflow, ad_text: str) -> ComplianceReport:
    handler = workflow.run(start_event=CheckInput(ad_text=ad_text))
    report = await handler
    if not isinstance(report, ComplianceReport):
        raise TypeError(f"Workflow returned {type(report)!r}, expected ComplianceReport")
    return report


_RULE_PROMPT = """You are reviewing a UK gambling advertisement against Advertising Guidance
from the Committee of Advertising Practice (CAP) and the Broadcast Committee of
Advertising Practice (BCAP) on protecting under-18s. This is not legal advice and
the guidance is not a Gambling Commission licence condition.

Judge the ad ONLY against the guidance section provided. Use one of:
- breach: the copy clearly conflicts with this section
- ok: the copy is clearly consistent with this section
- needs_review: borderline, or you lack facts that this section says matter
  (for example, audience targeting or a personality's following)

The citation.quote MUST be a short verbatim excerpt from the guidance text below,
not from the ad, and not paraphrased.

If the verdict is breach or needs_review, suggest a rewrite of the ad copy.
If the verdict is ok, leave suggested_rewrite null.

Section id: {section_id}
Section title: {section_title}

AD COPY:
\"\"\"{ad_text}\"\"\"

GUIDANCE:
\"\"\"{context}\"\"\"
"""
