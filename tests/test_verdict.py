from __future__ import annotations

from ad_compliance.chunking import default_pdf_path
from ad_compliance.config import Settings
from ad_compliance.ingest import ingest_guidance
from ad_compliance.models import Citation, RuleVerdict, aggregate_label
from ad_compliance.retrieve import RetrievedChunk
from ad_compliance.verdict import (
    build_report,
    check_ad,
    format_report,
    quote_is_faithful,
)


def test_aggregate_label_precedence() -> None:
    ok = RuleVerdict(rule_id="s1", rule_title="t", verdict="ok", rationale="x", confidence=0.9)
    review = RuleVerdict(
        rule_id="s2", rule_title="t", verdict="needs_review", rationale="x", confidence=0.4
    )
    breach = RuleVerdict(
        rule_id="s3", rule_title="t", verdict="breach", rationale="x", confidence=0.8
    )
    assert aggregate_label([ok]) == "ok"
    assert aggregate_label([ok, review]) == "needs_review"
    assert aggregate_label([ok, review, breach]) == "breach"


def test_unfaithful_quote_is_detected() -> None:
    source = "Cartoons and Santa Claus are high-risk content for under-18s."
    assert quote_is_faithful("Santa Claus are high-risk", source)
    assert not quote_is_faithful("the Gambling Commission requires a licence", source)


def test_empty_retrieval_is_needs_review() -> None:
    report = build_report("an adult horse-racing bet, typography only", [], [])
    assert report.overall == "needs_review"
    assert report.verdicts == []


def test_workflow_with_scripted_checker(tmp_settings: Settings) -> None:
    ingest_guidance(default_pdf_path(), settings=tmp_settings, mock_embeddings=True)

    def checker(ad_text: str, section_id: str, hits: list[RetrievedChunk]) -> RuleVerdict:
        quote = " ".join(hits[0].text.split())[:180]
        return RuleVerdict(
            rule_id=section_id,
            rule_title=hits[0].section_title,
            verdict="breach",
            rationale="Scripted test: Santa-themed animation is high risk.",
            confidence=0.91,
            citation=Citation(
                section_number=hits[0].section_number,
                section_title=hits[0].section_title,
                page_start=hits[0].page_start,
                page_end=hits[0].page_end,
                quote=quote,
            ),
            suggested_rewrite="Drop the cartoon Santa and show only typographic branding.",
        )

    report = check_ad(
        "Spin with Santa and his cuddly elves in our new cartoon slots!",
        settings=tmp_settings,
        mock_embeddings=True,
        checker=checker,
    )
    assert report.overall == "breach"
    assert report.verdicts
    assert all(item.verdict == "breach" for item in report.verdicts)
    rendered = format_report(report)
    assert "BREACH" in rendered
    assert "Santa" in rendered or "rewrite" in rendered.lower()


def test_build_llm_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from ad_compliance.config import Settings as S
    from ad_compliance.providers import build_llm

    settings = S(_env_file=None, llm_provider="openai")
    try:
        build_llm(settings)
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
