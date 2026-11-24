"""FastAPI HTTP interface for the compliance checker."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ad_compliance import __version__
from ad_compliance.models import ComplianceReport
from ad_compliance.retrieve import RetrievedChunk, retrieve
from ad_compliance.verdict import check_ad

app = FastAPI(
    title="Ad Compliance Checker",
    description=(
        "Review gambling advertising copy against UK CAP/BCAP guidance on "
        "protecting under-18s. This is a technical demonstration, not legal advice."
    ),
    version=__version__,
)


class CheckRequest(BaseModel):
    ad_text: str = Field(min_length=1, description="The advertisement copy to review.")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    query_type: str = "hybrid"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/check", response_model=ComplianceReport)
def check(payload: CheckRequest) -> ComplianceReport:
    try:
        return check_ad(payload.ad_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrieve", response_model=list[RetrievedChunk])
def retrieve_endpoint(payload: RetrieveRequest) -> list[RetrievedChunk]:
    try:
        return retrieve(payload.query, top_k=payload.top_k, query_type=payload.query_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
