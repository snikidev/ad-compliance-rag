from __future__ import annotations

from pathlib import Path

from ad_compliance.chunking import DOC_ID, chunk_pdf, default_pdf_path


def test_default_pdf_is_present() -> None:
    path = default_pdf_path()
    assert path.is_file()
    assert path.suffix.lower() == ".pdf"


def test_all_numbered_sections_are_extracted(guidance_pdf: Path) -> None:
    chunks = chunk_pdf(guidance_pdf)
    numbers = {chunk.section_number for chunk in chunks if chunk.section_number is not None}
    assert numbers == set(range(1, 31))
    assert len(chunks) >= 30
    assert all(chunk.doc_id == DOC_ID for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_wrapped_section_titles(guidance_pdf: Path) -> None:
    chunks = chunk_pdf(guidance_pdf)
    titles = {chunk.section_number: chunk.section_title for chunk in chunks if chunk.section_number}
    assert "appeal to under-18s" in titles[8]
    assert "strong" in titles[15].lower()
    assert "profile" in titles[17].lower()
    assert titles[19].endswith("characters")


def test_metadata_and_size_bounds(guidance_pdf: Path) -> None:
    chunks = chunk_pdf(guidance_pdf)
    assert all(chunk.page_start >= 1 for chunk in chunks)
    assert all(chunk.page_end >= chunk.page_start for chunk in chunks)
    assert all(len(chunk.text) <= 1600 for chunk in chunks)
    assert any("Section 16." in chunk.text for chunk in chunks)
    # At least a few chunks should have picked up CAP/BCAP Code citations.
    assert any(chunk.code_refs for chunk in chunks)


def test_ids_are_stable(guidance_pdf: Path) -> None:
    first = [chunk.chunk_id for chunk in chunk_pdf(guidance_pdf)]
    second = [chunk.chunk_id for chunk in chunk_pdf(guidance_pdf)]
    assert first == second
