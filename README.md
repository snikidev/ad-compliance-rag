# Ad Compliance Checker

A Retrieval-Augmented Generation (RAG) showcase that reviews gambling
advertising copy against the UK CAP/BCAP guidance *Gambling and lotteries
advertising: protecting under-18s* (2022).

This is a technical demonstration, not legal advice. The source document is
Advertising Guidance from the Committee of Advertising Practice (CAP) and the
Broadcast Committee of Advertising Practice (BCAP), interpreted by the ASA. It
is not a Gambling Commission licence condition.

## Status

The project is being rebuilt in stages on `main`. Scaffolding (packaging, lint,
tests, CI) lands first; ingestion, retrieval, the verdict workflow, evals, and
interfaces follow.

## Quickstart

```bash
uv sync
cp .env.example .env
uv run pytest
uv run ad-compliance ingest   # local FastEmbed + LanceDB under .lancedb/
uv run ad-compliance version
```

`ingest` uses FastEmbed by default (no API key). Unit tests use mock embeddings so CI never downloads a model or talks to a provider.

Python 3.12+ is required. See `.env.example` for provider settings.

## Licence

MIT
