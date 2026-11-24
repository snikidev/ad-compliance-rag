# Optional demo image. The project runs without Docker; see the README quickstart.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY evals ./evals

RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# A first-run ingest happens at container start if .lancedb is empty.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["demo"]
