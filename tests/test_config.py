from pathlib import Path

from ad_compliance.config import Settings


def test_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AD_COMPLIANCE_LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.embed_provider == "fastembed"
    assert settings.embed_model == "BAAI/bge-small-en-v1.5"
    assert settings.lancedb_uri == Path(".lancedb")
    assert settings.top_k == 8
    assert settings.reranker == "rrf"
    assert settings.openai_api_key is None


def test_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("AD_COMPLIANCE_TOP_K", "12")
    monkeypatch.setenv("AD_COMPLIANCE_RERANKER", "none")
    settings = Settings(_env_file=None)
    assert settings.top_k == 12
    assert settings.reranker == "none"


def test_unprefixed_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    settings = Settings(_env_file=None)
    assert settings.resolve_openai_key() == "sk-test-not-real"
