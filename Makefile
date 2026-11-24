.PHONY: lint test ci install eval ingest

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

ci: lint test

ingest:
	uv run ad-compliance ingest

eval:
	uv run ad-compliance eval
