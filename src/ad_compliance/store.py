"""LanceDB helpers: embedded on-disk table, opened in append mode by default."""

from __future__ import annotations

from pathlib import Path

import lancedb
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from ad_compliance.chunking import GuidanceChunk
from ad_compliance.config import Settings


def nodes_from_chunks(chunks: list[GuidanceChunk]) -> list[TextNode]:
    """Convert guidance chunks into LlamaIndex nodes with stable ids."""
    nodes: list[TextNode] = []
    for chunk in chunks:
        metadata = {
            "doc_id": chunk.doc_id,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "section_number": chunk.section_number,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "chunk_count": chunk.chunk_count,
            "code_refs": ", ".join(chunk.code_refs),
        }
        node = TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata),
            excluded_llm_metadata_keys=[],
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=chunk.doc_id)
        nodes.append(node)
    return nodes


def open_vector_store(
    settings: Settings,
    *,
    rebuild: bool = False,
) -> LanceDBVectorStore:
    """Open (or create) the embedded LanceDB table.

    `rebuild=True` drops the table first. Otherwise an existing table is opened
    in append mode so ingest can upsert without wiping other documents.
    """
    uri = Path(settings.lancedb_uri)
    uri.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(uri))
    table_name = settings.lancedb_table
    if rebuild and table_name in _table_names(db):
        db.drop_table(table_name)

    mode = "create"
    store = LanceDBVectorStore(
        uri=str(uri),
        table_name=table_name,
        mode=mode,
        query_type="hybrid",
        connection=db,
    )
    return store


def upsert_nodes(
    store: LanceDBVectorStore,
    nodes: list[BaseNode],
    *,
    doc_id: str,
) -> int:
    """Replace every node for `doc_id` with `nodes` (idempotent by document).

    Node ids are content-addressed, so a no-op re-ingest keeps the same ids.
    """
    if not nodes:
        return 0

    if store._table is not None:  # noqa: SLF001
        _delete_doc(store, doc_id)

    # Fresh tables are created by add() when store.table is still None.
    # Re-adding after delete also marks the FTS index stale.
    ids = store.add(nodes)
    _ensure_fts(store)
    return len(ids)


def _table_names(db: lancedb.DBConnection) -> list[str]:
    listing = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    return list(listing)


def _delete_doc(store: LanceDBVectorStore, doc_id: str) -> None:
    """Delete by doc_id using Lance SQL string literals (double quotes are identifiers)."""
    table = store._table  # noqa: SLF001
    if table is None:
        return
    safe = doc_id.replace("'", "''")
    table.delete(f"{store.doc_id_key} = '{safe}'")


def _ensure_fts(store: LanceDBVectorStore) -> None:
    table = store._table  # noqa: SLF001
    if table is None:
        return
    try:
        from lancedb.index import FTS

        table.create_index(store.text_key, config=FTS(), replace=True)
        store._fts_index_ready = True  # noqa: SLF001 - flag used by the hybrid query path
    except Exception:
        # FTS is best-effort here; the query path also creates the index on first use.
        store._fts_index_ready = False  # noqa: SLF001


def table_row_count(store: LanceDBVectorStore) -> int:
    table = store._table  # noqa: SLF001
    if table is None:
        return 0
    return int(table.count_rows())


def node_ids_in_store(store: LanceDBVectorStore) -> list[str]:
    table = store._table  # noqa: SLF001
    if table is None:
        return []
    return list(table.to_pandas()["id"].astype(str))
