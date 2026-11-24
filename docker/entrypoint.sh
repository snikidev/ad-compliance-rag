#!/bin/sh
set -eu
if [ ! -d /app/.lancedb ] || [ -z "$(ls -A /app/.lancedb 2>/dev/null || true)" ]; then
  echo "Building the local LanceDB index (FastEmbed, first run only)…"
  ad-compliance ingest
fi
case "${1:-demo}" in
  demo)
    exec streamlit run src/ad_compliance/demo.py --server.address 0.0.0.0 --server.port 8501
    ;;
  api)
    exec uvicorn ad_compliance.api:app --host 0.0.0.0 --port 8000
    ;;
  *)
    exec "$@"
    ;;
esac
