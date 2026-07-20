"""
RAG Base — Owner: C (主责), 全组评审
Abstract base class defining the RAG pipeline interface.
All strategies MUST subclass this.

Also defines Protocol interfaces for external modules (retrieval, graph, LLM generation)
so that C's orchestration code never directly couples to B or D implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.models.rag import RAGContext, RAGQuery, RAGResponse


# ── Protocol Interfaces for external dependencies ──────────────────────────
# These protocols define the shape C *expects* from modules owned by B and D.
# C never imports concrete implementations — only these protocols.


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval components (Owner: B).

    Any vector / hybrid / keyword retriever must satisfy this interface.
    """

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> RAGContext:
        """Retrieve relevant chunks for a query.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of chunks to return.
            **kwargs: Additional retrieval parameters (filters, etc.).

        Returns:
            A RAGContext containing retrieved chunks.
        """
        ...

    @property
    def retriever_name(self) -> str:
        """Human-readable name for logging and trace."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking / compression components (Owner: B)."""

    def rerank(
        self, query: str, chunks: List[Any], top_k: int = 5
    ) -> List[Any]:
        """Rerank chunks by relevance to the query.

        Args:
            query: The search query.
            chunks: Candidate chunks (implementation-specific type).
            top_k: Number of top chunks to keep after reranking.

        Returns:
            Reranked list of chunks.
        """
        ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol for LLM generation (may be backed by LangChain, direct API, etc.)."""

    async def generate(
        self, prompt: str, context: str, **kwargs: Any
    ) -> str:
        """Generate a response using the LLM.

        Args:
            prompt: System / instruction prompt.
            context: Retrieved context to ground the answer.
            **kwargs: Model parameters (temperature, max_tokens, etc.).

        Returns:
            Generated answer text.
        """
        ...

    async def generate_with_tokens(
        self, prompt: str, context: str, **kwargs: Any
    ) -> tuple[str, Dict[str, int]]:
        """Like generate(), but also returns token usage.

        Returns:
            Tuple of (answer_text, {"prompt": N, "completion": M, "total": T}).
        """
        ...


@runtime_checkable
class GraphRetrieverProtocol(Protocol):
    """Protocol for knowledge-graph retrieval (Owner: D).

    C's GraphRAG and Agentic strategies depend on this — not on concrete graph classes.
    """

    def graph_search(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> RAGContext:
        """Search the knowledge graph for entities / relationships matching the query.

        Returns:
            RAGContext with chunks derived from graph nodes / community reports.
        """
        ...

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single entity by its graph ID."""
        ...

    def get_community_report(self, community_id: str) -> Optional[str]:
        """Retrieve a pre-computed community summary report."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding models (Owner: B)."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, same length as texts.
        """
        ...

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Returns:
            Embedding vector for the query.
        """
        ...


# ── Abstract Strategy Base ────────────────────────────────────────────────


class RAGStrategyBase(ABC):
    """Abstract base class for ALL RAG strategies.

    Every strategy — Naive, Advanced, Modular, GraphRAG, Agentic — MUST subclass this
    and implement the three core methods: retrieve, generate, run.

    Lifecycle:
        1. run(query)        ← public entry point (calls retrieve → generate)
        2.   retrieve(query) ← gather context chunks
        3.   generate(ctx)   ← produce final answer from context

    Subclasses may override run() for custom orchestration (e.g. multi-step agent loops).
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self._config = config
        self._name = self.__class__.__name__

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def strategy_name(self) -> str:
        """Human-readable strategy name used by registry and trace."""
        return self._name

    @property
    def strategy_type(self) -> "strategy_type":  # noqa: F821
        """Return the StrategyType enum value for this strategy."""
        raise NotImplementedError("Subclasses must define strategy_type")

    # ── Core pipeline methods ───────────────────────────────────────────

    @abstractmethod
    async def retrieve(self, query: RAGQuery) -> RAGContext:
        """Retrieve relevant context chunks for the given query.

        This is the RETRIEVAL phase of the RAG pipeline.
        Strategies may use vector search, graph traversal, hybrid, or agent tools.

        Args:
            query: Unified query object.

        Returns:
            Populated RAGContext with chunks.
        """
        ...

    @abstractmethod
    async def generate(self, context: RAGContext, query: RAGQuery) -> RAGResponse:
        """Generate an answer grounded in the retrieved context.

        This is the AUGMENTATION + GENERATION phase.

        Args:
            context: Retrieved context chunks.
            query: Original query (carries metadata, conversation_id, etc.).

        Returns:
            RAGResponse with answer, citations, confidence, etc.
        """
        ...

    @abstractmethod
    async def run(self, query: RAGQuery) -> RAGResponse:
        """Execute the full RAG pipeline: retrieve → augment → generate.

        This is the primary public entry point. Strategies may override to add
        intermediate steps (reranking, rewriting, multi-hop, tool calls, etc.).

        Args:
            query: Unified query object.

        Returns:
            Complete RAGResponse.
        """
        ...

    # ── Optional hooks ──────────────────────────────────────────────────

    async def pre_retrieve(self, query: RAGQuery) -> RAGQuery:
        """Hook called before retrieval. Override to implement query rewriting, etc."""
        return query

    async def post_retrieve(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        """Hook called after retrieval. Override to implement reranking, filtering, etc."""
        return context

    async def on_error(self, query: RAGQuery, error: Exception) -> RAGResponse:
        """Hook called when the pipeline encounters an unrecoverable error.

        Override to implement graceful degradation or fallback responses.
        By default, returns an error response.
        """
        return RAGResponse(
            query_id=query.query_id,
            answer="",
            strategy=query.strategy,
            error=f"{type(error).__name__}: {error}",
        )
