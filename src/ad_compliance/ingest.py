"""Ingest the CAP/BCAP guidance PDF into the embedded LanceDB store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding

from ad_compliance.chunking import DOC_ID, chunk_pdf, default_pdf_path
from ad_compliance.config import Settings, get_settings
from ad_compliance.providers import build_embed_model
from ad_compliance.store import (
    nodes_from_chunks,
    open_vector_store,
    table_row_count,
    upsert_nodes,
)


@dataclass(frozen=True)
class IngestResult:
    """Summary of a completed ingest run."""

    pdf_path: Path
    doc_id: str
    chunks: int
    rows: int
    lancedb_uri: Path
    table: str


def ingest_guidance(
    pdf_path: Path | None = None,
    *,
    settings: Settings | None = None,
    embed_model: BaseEmbedding | None = None,
    mock_embeddings: bool = False,
    rebuild: bool = False,
) -> IngestResult:
    """Chunk, embed, and upsert the guidance PDF.

    Re-running this function replaces the previous rows for the same document
    and keeps content-addressed ids, so it is safe to call repeatedly.
    """
    settings = settings or get_settings()
    path = Path(pdf_path) if pdf_path is not None else default_pdf_path(settings.data_dir)
    chunks = chunk_pdf(path)
    nodes = nodes_from_chunks(chunks)

    model = embed_model or build_embed_model(settings, mock=mock_embeddings)
    texts = [node.get_content() for node in nodes]
    vectors = model.get_text_embedding_batch(texts)
    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector

    store = open_vector_store(settings, rebuild=rebuild)
    upsert_nodes(store, nodes, doc_id=DOC_ID)
    return IngestResult(
        pdf_path=path,
        doc_id=DOC_ID,
        chunks=len(nodes),
        rows=table_row_count(store),
        lancedb_uri=Path(settings.lancedb_uri),
        table=settings.lancedb_table,
    )
