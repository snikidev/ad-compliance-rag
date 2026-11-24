"""Build embedding (and, later, LLM) clients from settings."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from ad_compliance.config import Settings

# bge-small-en-v1.5 and the unit-test mock both use this width.
DEFAULT_EMBED_DIM = 384


def build_embed_model(
    settings: Settings | None = None,
    *,
    mock: bool = False,
    embed_dim: int = DEFAULT_EMBED_DIM,
) -> BaseEmbedding:
    """Return the configured embedding model.

    `mock=True` produces a deterministic dummy vector and is what the unit tests
    use, so CI never downloads an ONNX model or talks to an API.
    """
    if mock:
        return MockEmbedding(embed_dim=embed_dim)

    settings = settings or Settings()
    provider = settings.embed_provider
    if provider == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        cache_dir = Path(".fastembed_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return FastEmbedEmbedding(model_name=settings.embed_model, cache_dir=str(cache_dir))

    if provider == "openai":
        raise RuntimeError(
            "OpenAI embeddings are configured in a later stage; "
            "set AD_COMPLIANCE_EMBED_PROVIDER=fastembed for local ingest."
        )
    if provider == "azure_openai":
        raise RuntimeError(
            "Azure OpenAI embeddings are configured in a later stage; "
            "set AD_COMPLIANCE_EMBED_PROVIDER=fastembed for local ingest."
        )
    if provider == "ollama":
        raise RuntimeError(
            "Ollama embeddings are configured in a later stage; "
            "set AD_COMPLIANCE_EMBED_PROVIDER=fastembed for local ingest."
        )
    raise ValueError(f"Unknown embed provider: {provider}")


def build_llm(settings: Settings | None = None, *, mock: bool = False):
    """Return a chat LLM. Implemented fully in the verdict-engine stage."""
    if mock:
        from llama_index.core.llms.mock import MockLLM

        return MockLLM()
    raise RuntimeError(
        "LLM providers (OpenAI, Azure OpenAI, Ollama) are wired up in the "
        "verdict-engine stage. Until then, keep AD_COMPLIANCE_RERANKER=rrf."
    )
