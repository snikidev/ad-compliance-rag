"""Runtime settings loaded from the environment and an optional .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["openai", "azure_openai", "ollama"]
EmbedProvider = Literal["fastembed", "openai", "azure_openai", "ollama"]
RerankerName = Literal["rrf", "llm", "none"]


class Settings(BaseSettings):
    """Application settings.

    Provider API keys are also read from the usual un-prefixed names
    (`OPENAI_API_KEY`, `AZURE_OPENAI_*`) so a standard .env works.
    """

    model_config = SettingsConfigDict(
        env_prefix="AD_COMPLIANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: LlmProvider = "openai"
    embed_provider: EmbedProvider = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "gpt-4o-mini"

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "AD_COMPLIANCE_OPENAI_API_KEY"),
    )
    azure_openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_OPENAI_API_KEY", "AD_COMPLIANCE_AZURE_OPENAI_API_KEY"),
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_OPENAI_ENDPOINT", "AD_COMPLIANCE_AZURE_OPENAI_ENDPOINT"
        ),
    )
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_API_VERSION", "AD_COMPLIANCE_AZURE_OPENAI_API_VERSION"
        ),
    )
    azure_openai_chat_deployment: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AD_COMPLIANCE_AZURE_OPENAI_CHAT_DEPLOYMENT",
        ),
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AD_COMPLIANCE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        ),
    )
    azure_openai_embedding_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT",
            "AD_COMPLIANCE_AZURE_OPENAI_EMBEDDING_ENDPOINT",
        ),
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"

    lancedb_uri: Path = Path(".lancedb")
    lancedb_table: str = "guidance"
    data_dir: Path = Path("data")

    top_k: int = 8
    rerank_top_n: int = 5
    reranker: RerankerName = "rrf"

    def resolve_openai_key(self) -> str | None:
        return self.openai_api_key

    def resolve_azure_key(self) -> str | None:
        return self.azure_openai_api_key


def get_settings() -> Settings:
    """Load settings from the environment. Callers may cache this if they wish."""
    return Settings()
