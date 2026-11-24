"""Hybrid retrieval over the ingested CAP/BCAP guidance."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from ad_compliance.config import Settings, get_settings
from ad_compliance.providers import build_embed_model
from ad_compliance.store import open_vector_store


class EmptyIndexError(RuntimeError):
    """Raised when retrieval is attempted before the guidance has been ingested."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked guidance passage, ready to show or to feed a verdict step."""

    chunk_id: str
    text: str
    score: float | None
    section_id: str
    section_title: str
    section_number: int | None
    page_start: int
    page_end: int
    code_refs: tuple[str, ...]

    def citation_label(self) -> str:
        if self.section_number is not None:
            heading = f"s{self.section_number}. {self.section_title}"
        else:
            heading = self.section_title
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}–{self.page_end}"
        )
        return f"{heading} ({pages})"


def retrieve(
    query: str,
    *,
    settings: Settings | None = None,
    embed_model: BaseEmbedding | None = None,
    mock_embeddings: bool = False,
    top_k: int | None = None,
    query_type: str | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid (FTS + vector) search, with RRF fusion inside LanceDB.

    Set `query_type` to ``vector`` or ``fts`` when comparing modes. The default
    comes from settings and is ``hybrid``. Optional LLM rerank is applied after
    retrieval when ``settings.reranker == "llm"``.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    settings = settings or get_settings()
    model = embed_model or build_embed_model(settings, mock=mock_embeddings)
    store = _open_populated_store(settings)
    store.query_type = query_type or "hybrid"

    fetch_k = top_k or settings.top_k
    index = VectorStoreIndex.from_vector_store(store, embed_model=model)
    retriever = index.as_retriever(
        similarity_top_k=fetch_k,
        vector_store_query_mode=_query_mode(store.query_type),
    )
    try:
        nodes = retriever.retrieve(query)
    except Exception as exc:
        # LlamaIndex's LanceDB adapter raises a Warning (yes, raise Warning)
        # when the result set is empty.
        if "empty" in str(exc).lower():
            return []
        raise

    nodes = _maybe_llm_rerank(nodes, query, settings)
    limit = settings.rerank_top_n if settings.reranker != "none" else fetch_k
    return [_to_chunk(node) for node in nodes[:limit]]


def format_debug_report(query: str, hits: list[RetrievedChunk]) -> str:
    """Pretty-print retrieval hits for the debug-retrieve command."""
    if not hits:
        return f"No passages matched {query!r}. Did you run `ad-compliance ingest`?"
    lines = [f"{len(hits)} hit(s) for {query!r}:", ""]
    for rank, hit in enumerate(hits, start=1):
        score = "n/a" if hit.score is None else f"{hit.score:.4f}"
        quote = _shorten(hit.text, 280)
        lines.append(f"{rank}. {hit.citation_label()}  score={score}")
        lines.append(f"   id: {hit.chunk_id}")
        lines.append(f"   {quote}")
        if hit.code_refs:
            lines.append(f"   code refs: {', '.join(hit.code_refs)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _open_populated_store(settings: Settings) -> LanceDBVectorStore:
    store = open_vector_store(settings)
    if store._table is None:  # noqa: SLF001
        raise EmptyIndexError(
            "No LanceDB table found. Run `ad-compliance ingest` first "
            f"(looked in {settings.lancedb_uri}/{settings.lancedb_table})."
        )
    return store


def _query_mode(query_type: str) -> VectorStoreQueryMode:
    mapping = {
        "hybrid": VectorStoreQueryMode.HYBRID,
        "fts": VectorStoreQueryMode.TEXT_SEARCH,
        "vector": VectorStoreQueryMode.DEFAULT,
    }
    return mapping.get(query_type, VectorStoreQueryMode.HYBRID)


def _maybe_llm_rerank(
    nodes: list[NodeWithScore],
    query: str,
    settings: Settings,
) -> list[NodeWithScore]:
    if settings.reranker != "llm" or not nodes:
        return nodes
    from llama_index.core.postprocessor import LLMRerank

    from ad_compliance.providers import build_llm

    rerank = LLMRerank(top_n=settings.rerank_top_n, llm=build_llm(settings))
    return list(rerank.postprocess_nodes(nodes, query_str=query))


def _to_chunk(node: NodeWithScore) -> RetrievedChunk:
    meta = dict(node.node.metadata or {})
    section_number = meta.get("section_number")
    if section_number == "" or section_number is None:
        number: int | None = None
    else:
        number = int(section_number)
    refs_raw = meta.get("code_refs") or ""
    refs = tuple(part.strip() for part in str(refs_raw).split(",") if part.strip())
    return RetrievedChunk(
        chunk_id=node.node.node_id,
        text=node.node.get_content(),
        score=node.score,
        section_id=str(meta.get("section_id") or ""),
        section_title=str(meta.get("section_title") or ""),
        section_number=number,
        page_start=int(meta.get("page_start") or 0),
        page_end=int(meta.get("page_end") or 0),
        code_refs=refs,
    )


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
