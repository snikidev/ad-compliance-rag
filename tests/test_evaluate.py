from __future__ import annotations

from collections import Counter

from ad_compliance.evaluate import (
    CaseScore,
    default_dataset_path,
    faithfulness_rate,
    load_dataset,
    retrieval_hit,
    retrieval_hit_rate,
    run_eval,
    verdict_accuracy,
)
from ad_compliance.models import Citation, ComplianceReport, RuleVerdict
from ad_compliance.retrieve import RetrievedChunk


def test_dataset_is_labelled_and_synthetic() -> None:
    cases = load_dataset(default_dataset_path())
    assert 30 <= len(cases) <= 50
    counts = Counter(case.label for case in cases)
    assert counts["breach"] >= 10
    assert counts["ok"] >= 10
    assert counts["needs_review"] >= 10
    assert all(case.expected_sections for case in cases)
    assert all(case.text.strip() for case in cases)
    assert all(case.id.startswith(("b", "o", "r")) for case in cases)


def test_retrieval_and_verdict_metrics() -> None:
    scores = [
        CaseScore(
            case_id="a",
            label="breach",
            predicted="breach",
            retrieval_hit=True,
            expected_sections=(20,),
            retrieved_sections=(20, 23),
            faithful_citations=1,
            total_citations=1,
        ),
        CaseScore(
            case_id="b",
            label="ok",
            predicted="needs_review",
            retrieval_hit=False,
            expected_sections=(28,),
            retrieved_sections=(16,),
            faithful_citations=0,
            total_citations=1,
        ),
    ]
    assert retrieval_hit((20, 23), (16, 20)) is True
    assert retrieval_hit((20,), (16, 18)) is False
    assert retrieval_hit_rate(scores) == 0.5
    assert verdict_accuracy(scores) == 0.5
    assert faithfulness_rate(scores) == 0.5
    assert verdict_accuracy([]) is None


def test_run_eval_with_stubs() -> None:
    cases = load_dataset(default_dataset_path())
    first = cases[0]

    def retriever(text: str) -> list[RetrievedChunk]:
        # Always return the first labelled section so hit-rate is 1.0.
        match = next(case for case in cases if case.text == text)
        section = match.expected_sections[0]
        return [
            RetrievedChunk(
                chunk_id="x",
                text="Cartoons and Santa Claus are high-risk content for under-18s. " * 2,
                score=1.0,
                section_id=f"s{section:02d}",
                section_title="stub",
                section_number=section,
                page_start=20,
                page_end=20,
                code_refs=(),
            )
        ]

    def judge(text: str) -> ComplianceReport:
        match = next(case for case in cases if case.text == text)
        quote = "Cartoons and Santa Claus are high-risk content for under-18s."
        return ComplianceReport(
            overall=match.label,
            summary="stub",
            ad_text=text,
            verdicts=[
                RuleVerdict(
                    rule_id="s20",
                    rule_title="stub",
                    verdict=match.label,
                    rationale="stub",
                    confidence=0.8,
                    citation=Citation(
                        section_number=20,
                        section_title="stub",
                        page_start=20,
                        page_end=20,
                        quote=quote,
                    ),
                )
            ],
        )

    result = run_eval(
        ingest_if_needed=False,
        retriever=retriever,
        judge=judge,
        run_verdicts=True,
    )
    assert result.cases == len(cases)
    assert result.retrieval_hit_rate == 1.0
    assert result.verdict_accuracy == 1.0
    assert result.faithfulness == 1.0
    assert not result.skipped_verdicts
    assert first.id in {score.case_id for score in result.per_case}


def test_retrieval_eval_smoke(tmp_settings) -> None:
    """FTS still runs with mock embeddings; this is a smoke test, not a published score."""
    from ad_compliance.config import Settings

    assert isinstance(tmp_settings, Settings)
    result = run_eval(
        settings=tmp_settings,
        mock_embeddings=True,
        ingest_if_needed=True,
        run_verdicts=False,
    )
    assert result.cases >= 30
    assert result.skipped_verdicts
    assert result.verdict_accuracy is None
    assert result.retrieval_hit_rate is not None
    assert result.retrieval_hit_rate >= 0.25
