from __future__ import annotations

from fastapi.testclient import TestClient

from ad_compliance import __version__
from ad_compliance.api import app
from ad_compliance.models import ComplianceReport


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == __version__


def test_check_endpoint_uses_workflow(monkeypatch) -> None:
    from ad_compliance import api as api_module

    fake = ComplianceReport(
        overall="ok",
        summary="stub",
        ad_text="plain typographic racing odds, 18+",
        verdicts=[],
    )

    def _fake_check(ad_text: str, **kwargs):
        assert "racing" in ad_text
        return fake

    monkeypatch.setattr(api_module, "check_ad", _fake_check)
    client = TestClient(app)
    response = client.post("/check", json={"ad_text": "plain typographic racing odds, 18+"})
    assert response.status_code == 200
    assert response.json()["overall"] == "ok"


def test_check_rejects_empty() -> None:
    client = TestClient(app)
    response = client.post("/check", json={"ad_text": ""})
    assert response.status_code == 422
