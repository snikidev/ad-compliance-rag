from __future__ import annotations

from ad_compliance.chunking import DOC_ID, default_pdf_path
from ad_compliance.config import Settings
from ad_compliance.ingest import ingest_guidance
from ad_compliance.store import node_ids_in_store, open_vector_store, table_row_count


def test_ingest_is_idempotent(tmp_settings: Settings) -> None:
    pdf = default_pdf_path()
    first = ingest_guidance(pdf, settings=tmp_settings, mock_embeddings=True)
    assert first.chunks == first.rows
    assert first.chunks > 30
    assert first.doc_id == DOC_ID

    store = open_vector_store(tmp_settings)
    ids_first = sorted(node_ids_in_store(store))

    second = ingest_guidance(pdf, settings=tmp_settings, mock_embeddings=True)
    assert second.rows == first.rows
    assert table_row_count(open_vector_store(tmp_settings)) == first.rows
    assert sorted(node_ids_in_store(open_vector_store(tmp_settings))) == ids_first


def test_rebuild_recreates_table(tmp_settings: Settings) -> None:
    pdf = default_pdf_path()
    first = ingest_guidance(pdf, settings=tmp_settings, mock_embeddings=True)
    again = ingest_guidance(pdf, settings=tmp_settings, mock_embeddings=True, rebuild=True)
    assert again.rows == first.rows
