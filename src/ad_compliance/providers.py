"""Build embedding and LLM clients from settings."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import LLM
from llama_index.core.llms.mock import MockLLM

from ad_compliance.config import Settings, get_settings

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

    settings = settings or get_settings()
    provider = settings.embed_provider
    if provider == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        cache_dir = Path(".fastembed_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return FastEmbedEmbedding(model_name=settings.embed_model, cache_dir=str(cache_dir))

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = settings.resolve_openai_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAIEmbedding(model=settings.embed_model, api_key=api_key)

    if provider == "azure_openai":
        from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

        api_key = settings.resolve_azure_key()
        if not api_key or not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required.")
        return AzureOpenAIEmbedding(
            model=settings.embed_model,
            deployment_name=settings.azure_openai_embedding_deployment,
            api_key=api_key,
            azure_endpoint=settings.azure_openai_embedding_endpoint
            or settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    if provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unknown embed provider: {provider}")


def build_llm(settings: Settings | None = None, *, mock: bool = False) -> LLM:
    """Return the configured chat LLM."""
    if mock:
        return MockLLM()

    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = settings.resolve_openai_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(model=settings.llm_model, api_key=api_key, temperature=0.0)

    if provider == "azure_openai":
        from llama_index.llms.azure_openai import AzureOpenAI

        api_key = settings.resolve_azure_key()
        if not api_key or not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required.")
        return AzureOpenAI(
            model=settings.llm_model,
            engine=settings.azure_openai_chat_deployment,
            deployment_name=settings.azure_openai_chat_deployment,
            api_key=api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,
        )

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            request_timeout=120.0,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
