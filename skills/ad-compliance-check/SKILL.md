---
name: ad-compliance-check
description: >
  Review UK gambling advertising copy against CAP/BCAP guidance on protecting
  under-18s. Use when the user wants to check an ad, script, social post, or
  influencer brief for strong appeal to under-18s, retrieve the relevant
  guidance passages, or run the project's evals. Call the ad-compliance CLI
  or the check_ad MCP tool; do not invent verdicts from memory of the Codes.
---

# Ad Compliance Checker

This skill reviews **gambling and lottery marketing copy** against the 2022
CAP/BCAP Advertising Guidance *Gambling and lotteries advertising: protecting
under-18s*. It is a technical demonstration, **not legal advice**. The source
document is Advertising Guidance from CAP/BCAP, applied by the ASA. It is
**not** a Gambling Commission licence condition and does not bind the ASA
Councils.

Prefer the tools below over recalling the Codes from training data. Always
show citations (section, page, quote) from the retrieved guidance.

## When to use

- The user pastes ad copy, a script, a social post, or an influencer brief
  and asks whether it is likely to appeal strongly to under-18s.
- The user wants the specific CAP/BCAP passages that apply to a creative idea
  (cartoons, Santa, loot boxes, footballers, under-25s, music, targeting).
- The user wants to ingest the guidance PDF, inspect retrieval, or run evals.

Do not use this skill for general legal advice, licence applications, or
non-UK advertising regimes.

## Prerequisites

From a checkout of this repository:

```bash
uv sync
uv run ad-compliance ingest
```

Ingest uses **FastEmbed** locally and does not need an API key. It writes a
LanceDB index to `.lancedb/`. The index is **not** shipped pre-built; the
first run has to build it (the embedding model is downloaded once).

`check` (the verdict) needs an LLM provider. Copy `.env.example` to `.env`
and set one of OpenAI, Azure OpenAI, or Ollama.

You can also run the CLI from git without a checkout:

```bash
uvx --from git+https://github.com/snikidev/llamaindex-python-rag ad-compliance --help
```

`uvx` still needs a working directory in which to write `.lancedb/` and to
find or download the guidance PDF (bundled in the package).

## Call the CLI

```bash
# Build or refresh the index
uv run ad-compliance ingest

# Structured verdict (breach | ok | needs_review) with citations
uv run ad-compliance check "PASTE AD COPY HERE"
uv run ad-compliance check --file ./ad.txt --json

# Inspect what would be retrieved, with scores and passages
uv run ad-compliance debug-retrieve "cartoon Santa slots ad"

# Retrieval metrics, no LLM
uv run ad-compliance eval --retrieval-only

# Full judge (needs a provider key)
uv run ad-compliance eval --verdicts
```

Interpret `needs_review` as "the guidance says this depends on facts we do
not have" (for example, a personality's follower demographics), not as a
pass.

## Call the MCP server

The same check is exposed as a FastMCP tool named `check_ad`.

```bash
uv run ad-compliance mcp
```

Or, in a client config (Claude Code, Cursor, OpenCode, OpenChamber):

```json
{
  "mcpServers": {
    "ad-compliance": {
      "command": "uv",
      "args": ["run", "ad-compliance", "mcp"]
    }
  }
}
```

Tool: `check_ad`
Argument: `ad_text` (string) — the advertisement copy, script, or post.

The tool returns a structured report: `overall`, `summary`, per-rule
`verdicts` (each with `verdict`, `rationale`, `confidence`, `citation`, and
optional `suggested_rewrite`). Show those citations to the user. If the tool
errors because the index is missing, run `ad-compliance ingest` first. If it
errors because no LLM key is set, tell the user which provider variable to
export.

## How to present results

1. Lead with the overall verdict and a one-line summary.
2. List only the rules that are `breach` or `needs_review`, with the quote
   and page reference.
3. Include any suggested rewrite, clearly labelled as a creative suggestion,
   not a legal clearance.
4. Repeat the disclaimer if the user might treat the output as a formal
   opinion.

## Limits

- The index covers this one 2022 guidance note, not the full CAP or BCAP
  Codes and not later ASA rulings.
- Targeting and personality-appeal questions often cannot be closed without
  audience data.
- Do not claim the tool speaks for the ASA, CAP, BCAP, or the Gambling
  Commission.
