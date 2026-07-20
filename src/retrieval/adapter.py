"""Adapter from B3 VectorRetriever hits to C's frozen RetrieverProtocol."""

from copy import deepcopy
import time
from typing import Any, Callable, Dict, List, Optional

from src.models.schemas import RetrievalHit

from .retriever import VectorRetriever


class RetrieverAdapterError(Exception):
    """Base error for C protocol conversion failures."""


class MissingKnowledgeBaseError(RetrieverAdapterError):
    """Raised instead of falling back to unsafe all-KB retrieval."""

    def __init__(self) -> None:
        super().__init__("kb_id is required for isolated retrieval")


ContextFactory = Callable[..., Any]


class VectorRetrieverAdapter:
    """Synchronous structural implementation of C's RetrieverProtocol.

    C's contract is ``retrieve(query, top_k=5, **kwargs) -> RAGContext``.
    The C models are imported only while constructing a real context, allowing
    this branch to remain compatible until C's frozen contract is merged.
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        context_factory: Optional[ContextFactory] = None,
    ) -> None:
        self._retriever = retriever
        self._context_factory = context_factory

    @property
    def retriever_name(self) -> str:
        """Return the wrapped retriever name required by C."""

        return self._retriever.retriever_name

    def search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        """Forward B strategy searches while retaining C's adapter contract.

        C's service injects its ``RetrieverProtocol`` implementation directly
        into ``RAGContext``.  B strategies consume the isolated ``search``
        boundary, so the adapter must expose both surfaces without performing
        a second embedding or retrieval operation.
        """

        return self._retriever.search(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
        )

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> Any:
        """Extract C parameters, execute one search, and build RAGContext."""

        kb_id = kwargs.get("kb_id")
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise MissingKnowledgeBaseError()
        filters = kwargs.get("filters")
        started = time.perf_counter()
        hits = self._retriever.search(
            query,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if self._context_factory is not None:
            return self._context_factory(
                query=query,
                hits=deepcopy(hits),
                retrieval_method=self.retriever_name,
                retrieval_latency_ms=latency_ms,
                total_candidates=len(hits),
                metadata={"kb_id": kb_id, "filters": deepcopy(filters or {})},
            )
        return self._build_c_context(query, kb_id, filters, hits, latency_ms)

    def _build_c_context(
        self,
        query: str,
        kb_id: str,
        filters: Optional[Dict[str, Any]],
        hits: List[RetrievalHit],
        latency_ms: float,
    ) -> Any:
        try:
            from src.models.rag import RAGChunk, RAGContext, RAGSource
        except ImportError as exc:
            raise RetrieverAdapterError(
                "C RAGContext/RAGChunk/RAGSource contracts are not available on "
                "this branch; merge C's frozen contract or inject context_factory"
            ) from exc

        chunks = []
        for hit in hits:
            metadata = deepcopy(hit.chunk.metadata)
            metadata.update(deepcopy(hit.metadata))
            metadata.update(
                {
                    "rank": hit.rank,
                    "retriever": hit.retriever,
                    "retrieval_hit_metadata": deepcopy(hit.metadata),
                }
            )
            chunks.append(
                RAGChunk(
                    chunk_id=hit.chunk.id,
                    content=hit.chunk.text,
                    embedding=deepcopy(hit.chunk.embedding),
                    source=RAGSource(
                        document_id=hit.chunk.document_id,
                        chunk_index=hit.chunk.index,
                        source_path=metadata.get("source"),
                        title=metadata.get("section") or metadata.get("filename"),
                        page=metadata.get("page"),
                        score=hit.score,
                        metadata=metadata,
                    ),
                )
            )
        return RAGContext(
            chunks=chunks,
            query=query,
            retrieval_method=self.retriever_name,
            retrieval_latency_ms=latency_ms,
            total_candidates=len(hits),
            metadata={"kb_id": kb_id, "filters": deepcopy(filters or {})},
        )


__all__ = [
    "MissingKnowledgeBaseError",
    "RetrieverAdapterError",
    "VectorRetrieverAdapter",
]
