"""Evaluation over the synthetic ad-snippet dataset."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_compliance.chunking import default_pdf_path
from ad_compliance.config import Settings, get_settings
from ad_compliance.ingest import ingest_guidance
from ad_compliance.models import ComplianceReport, VerdictLabel
from ad_compliance.retrieve import RetrievedChunk, retrieve
from ad_compliance.verdict import check_ad, quote_is_faithful

DATASET_NAME = "snippets.json"


@dataclass(frozen=True)
class EvalCase:
    """One labelled, synthetic advertisement."""

    id: str
    text: str
    label: VerdictLabel
    expected_sections: tuple[int, ...]
    channel: str = ""
    notes: str = ""


@dataclass
class CaseScore:
    case_id: str
    label: VerdictLabel
    predicted: VerdictLabel | None
    retrieval_hit: bool
    expected_sections: tuple[int, ...]
    retrieved_sections: tuple[int, ...]
    faithful_citations: int = 0
    total_citations: int = 0
    error: str | None = None


@dataclass
class EvalResult:
    cases: int
    retrieval_hit_rate: float | None
    verdict_accuracy: float | None
    faithfulness: float | None
    per_case: list[CaseScore] = field(default_factory=list)
    skipped_verdicts: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "verdict_accuracy": self.verdict_accuracy,
            "faithfulness": self.faithfulness,
            "skipped_verdicts": self.skipped_verdicts,
            "notes": self.notes,
            "per_case": [case.__dict__ for case in self.per_case],
        }


def default_dataset_path() -> Path:
    candidates = [
        Path.cwd() / "evals" / DATASET_NAME,
        Path(__file__).resolve().parents[2] / "evals" / DATASET_NAME,
        Path(__file__).resolve().parent / "evals" / DATASET_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {DATASET_NAME} in {candidates}")


def load_dataset(path: Path | None = None) -> list[EvalCase]:
    data = json.loads((path or default_dataset_path()).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    cases: list[EvalCase] = []
    for item in items:
        cases.append(
            EvalCase(
                id=item["id"],
                text=item["text"],
                label=item["label"],
                expected_sections=tuple(item.get("expected_sections") or ()),
                channel=item.get("channel") or "",
                notes=item.get("notes") or "",
            )
        )
    return cases


def retrieval_hit(expected: Sequence[int], retrieved: Sequence[int | None]) -> bool:
    """A case hits if any labelled section number appears in the retrieved set."""
    wanted = {number for number in expected if number}
    if not wanted:
        return True
    got = {number for number in retrieved if number}
    return bool(wanted & got)


def retrieval_hit_rate(scores: Sequence[CaseScore]) -> float:
    if not scores:
        return 0.0
    return sum(1 for score in scores if score.retrieval_hit) / len(scores)


def verdict_accuracy(scores: Sequence[CaseScore]) -> float | None:
    judged = [score for score in scores if score.predicted is not None]
    if not judged:
        return None
    return sum(1 for score in judged if score.predicted == score.label) / len(judged)


def faithfulness_rate(scores: Sequence[CaseScore]) -> float | None:
    total = sum(score.total_citations for score in scores)
    if total == 0:
        return None
    return sum(score.faithful_citations for score in scores) / total


def run_eval(
    *,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    ingest_if_needed: bool = True,
    mock_embeddings: bool = False,
    run_verdicts: bool = False,
    retriever: Callable[[str], list[RetrievedChunk]] | None = None,
    judge: Callable[[str], ComplianceReport] | None = None,
) -> EvalResult:
    """Score retrieval (always) and, optionally, verdicts.

    Verdicts are skipped unless `run_verdicts` is true or a `judge` is supplied,
    so CI can measure retrieval with mock embeddings and no API key.
    """
    settings = settings or get_settings()
    cases = load_dataset(dataset_path)

    if retriever is None and ingest_if_needed:
        try:
            from ad_compliance.store import open_vector_store

            store = open_vector_store(settings)
            empty = store._table is None  # noqa: SLF001
        except Exception:
            empty = True
        if empty:
            ingest_guidance(
                default_pdf_path(settings.data_dir),
                settings=settings,
                mock_embeddings=mock_embeddings,
            )

    def _retrieve(text: str) -> list[RetrievedChunk]:
        if retriever is not None:
            return retriever(text)
        return retrieve(text, settings=settings, mock_embeddings=mock_embeddings)

    scores: list[CaseScore] = []
    notes: list[str] = []
    skipped = not run_verdicts and judge is None
    if skipped:
        notes.append("Verdicts were not run (no LLM judge). Retrieval metrics only.")

    for case in cases:
        try:
            hits = _retrieve(case.text)
        except Exception as exc:
            scores.append(
                CaseScore(
                    case_id=case.id,
                    label=case.label,
                    predicted=None,
                    retrieval_hit=False,
                    expected_sections=case.expected_sections,
                    retrieved_sections=(),
                    error=str(exc),
                )
            )
            continue

        retrieved_sections = tuple(
            hit.section_number for hit in hits if hit.section_number is not None
        )
        report: ComplianceReport | None = None
        predicted: VerdictLabel | None = None
        faithful = 0
        citations = 0
        error = None
        if not skipped:
            try:
                report = (
                    judge(case.text)
                    if judge is not None
                    else check_ad(
                        case.text,
                        settings=settings,
                        mock_embeddings=mock_embeddings,
                    )
                )
                predicted = report.overall
                faithful, citations = _citation_scores(report, hits)
            except Exception as exc:
                error = str(exc)

        scores.append(
            CaseScore(
                case_id=case.id,
                label=case.label,
                predicted=predicted,
                retrieval_hit=retrieval_hit(case.expected_sections, retrieved_sections),
                expected_sections=case.expected_sections,
                retrieved_sections=retrieved_sections,
                faithful_citations=faithful,
                total_citations=citations,
                error=error,
            )
        )

    return EvalResult(
        cases=len(cases),
        retrieval_hit_rate=retrieval_hit_rate(scores),
        verdict_accuracy=verdict_accuracy(scores),
        faithfulness=faithfulness_rate(scores),
        per_case=scores,
        skipped_verdicts=skipped,
        notes=notes,
    )


def format_eval_result(result: EvalResult) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        f"Cases: {result.cases}",
        f"Retrieval hit-rate: {pct(result.retrieval_hit_rate)}",
        f"Verdict accuracy: {pct(result.verdict_accuracy)}",
        f"Citation faithfulness: {pct(result.faithfulness)}",
    ]
    lines.extend(result.notes)
    misses = [score for score in result.per_case if not score.retrieval_hit]
    if misses:
        lines.append("Retrieval misses:")
        for score in misses:
            lines.append(
                f"  {score.case_id}: expected {list(score.expected_sections)} "
                f"got {list(score.retrieved_sections)}"
            )
    return "\n".join(lines) + "\n"


def _citation_scores(report: ComplianceReport, hits: list[RetrievedChunk]) -> tuple[int, int]:
    corpus = " ".join(hit.text for hit in hits)
    faithful = 0
    total = 0
    for item in report.verdicts:
        if item.citation is None or not item.citation.quote:
            continue
        total += 1
        if quote_is_faithful(item.citation.quote, corpus):
            faithful += 1
    return faithful, total
