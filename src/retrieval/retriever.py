"""Standard isolated vector retriever plus legacy compatibility entry points."""

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from src.models.schemas import RetrievalHit

from .embeddings import BaseEmbedder
from .vector_index import BaseVectorIndex


class VectorRetriever:
    """Embed text and search one explicitly selected knowledge base."""

    retriever_name: str = "vector"

    def __init__(self, embedder: BaseEmbedder, index: BaseVectorIndex):
        self.embedder = embedder
        self.index = index

    def search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalHit]:
        """Return isolated shared hits with one-based continuous ranks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None:
            if not isinstance(filters, Mapping):
                raise TypeError("filters must be a mapping or None")
            if filters.get("kb_id", kb_id) != kb_id:
                raise ValueError("filters cannot override kb_id")

        query_vector = self.embedder.embed(query)
        hits = self.index.search(
            query_vector,
            kb_id=kb_id,
            top_k=top_k,
            filters=filters,
        )
        result = []
        for rank, hit in enumerate(hits, start=1):
            metadata = deepcopy(hit.chunk.metadata)
            metadata.update(deepcopy(hit.metadata))
            metadata.setdefault("kb_id", hit.chunk.kb_id)
            metadata.setdefault("document_id", hit.chunk.document_id)
            metadata.setdefault("chunk_id", hit.chunk.id)
            metadata.setdefault("filename", "")
            metadata.setdefault("page", 1)
            metadata.setdefault("section", "")
            result.append(
                RetrievalHit(
                    chunk=deepcopy(hit.chunk),
                    score=hit.score,
                    rank=rank,
                    retriever=self.retriever_name,
                    metadata=metadata,
                )
            )
        return result


class Retriever(VectorRetriever):
    """Legacy retriever retained for existing callers.

    ``retrieve`` is the explicit legacy all-KB management path. New code must
    call :meth:`VectorRetriever.search` with a required ``kb_id``.
    """

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        query_vector = self.embedder.embed_query(query)
        return self.index.search_all(query_vector, top_k=top_k)

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 5,
        kb_id: Optional[str] = None,
    ) -> List[RetrievalHit]:
        if kb_id is None:
            return self.retrieve(query, top_k=top_k)
        return self.search(query, kb_id=kb_id, top_k=top_k)


__all__ = ["Retriever", "VectorRetriever"]
