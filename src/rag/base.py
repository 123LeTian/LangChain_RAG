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

from src.models.rag import (
    RAGContext,
    RAGMode,
    RAGQuery,
    RAGRequest,
    RAGResponse,
    RAGResult,
    StrategyType,
    TraceStage,
)


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
    def strategy_type(self) -> StrategyType:
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


# ═══════════════════════════════════════════════════════════════════════════
# C2 — RAGStrategy (Unified High-Level Interface)
# ═══════════════════════════════════════════════════════════════════════════


class RAGStrategy(ABC):
    """Unified high-level interface that every RAG paradigm MUST implement.

    This is the PRIMARY contract for all five RAG modes:
      Naive / Advanced / Modular / Graph / Agentic

    Design rules (enforced by convention — the strategy MUST NOT):
      1. Read HTTP requests or access any transport layer
      2. Directly operate databases or file systems
      3. Create LLM instances (use ``context.llm``)
      4. Create Retriever instances (use ``context.retriever``)

    All external dependencies are injected through :class:`RAGContext`:
      - ``context.llm``            — LLM generator
      - ``context.retriever``      — document retriever
      - ``context.tools``          — agent tools (for Agentic mode)
      - ``context.trace_recorder`` — callable for recording trace events
      - ``context.config``         — strategy-level configuration overrides

    Lifecycle:
        service = RAGService(...)
        context = service.build_context(request)   # injects deps
        strategy = registry.get(request.mode)
        result = await strategy.run(request, context)

    Subclassing::

        class MyNaiveRAG(RAGStrategy):
            mode: RAGMode = RAGMode.NAIVE

            async def run(self, request, context) -> RAGResult:
                # 1. Retrieve
                chunks_ctx = context.retriever.retrieve(request.query)
                context.chunks = chunks_ctx.chunks

                # 2. Record trace
                if context.trace_recorder:
                    context.trace_recorder(TraceStage.RETRIEVE, request.query,
                                           f"{len(context.chunks)} chunks", 45.2)

                # 3. Generate
                answer = await context.llm.generate(
                    prompt="Answer based on context.",
                    context=context.combined_text,
                )

                # 4. Return result
                return RAGResult(answer=answer, hits=context.chunks)
    """

    # ── Class-level mode identifier ──────────────────────────────────────

    # Subclasses MUST override this with a concrete RAGMode value.
    # Using a sentinel so that hasattr() works and the IDE sees the type.
    mode: RAGMode = RAGMode.NAIVE  # type: ignore[assignment]
    """Which RAG paradigm this strategy implements.

    Subclasses MUST override with a specific RAGMode.
    Example: ``mode = RAGMode.NAIVE``
    """

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def strategy_name(self) -> str:
        """Human-readable strategy name for logging and trace."""
        return self.__class__.__name__

    @property
    def strategy_mode(self) -> RAGMode:
        """Return the RAGMode for this strategy."""
        return self.mode

    # ── Core pipeline ────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, request: RAGRequest, context: RAGContext) -> RAGResult:
        """Execute the full RAG pipeline and return a unified result.

        This is the ONE method every strategy must implement.
        The strategy reads the user's intent from *request* and all
        capabilities (retriever, LLM, tools, tracer) from *context*.

        Args:
            request: The user's query + mode + options from the API layer.
            context:  Retrieved data AND injected dependencies (llm, retriever,
                      tools, trace_recorder, config).  The strategy reads
                      dependencies from here and writes results (chunks,
                      metadata) back.

        Returns:
            RAGResult with answer, citations, hits, trace, usage, warnings.

        Implementation pattern::

            async def run(self, request, context):
                # 1. Retrieve (via context.retriever — never create your own)
                result = context.retriever.retrieve(request.query, top_k=5)
                context.chunks = result.chunks

                # 2. Optionally record a trace event
                self._record(context, TraceStage.RETRIEVE, request.query,
                             f"{len(context.chunks)} chunks", latency)

                # 3. Generate (via context.llm — never create your own)
                answer = await context.llm.generate(
                    prompt="You are a helpful assistant.",
                    context=context.combined_text,
                )

                # 4. Return
                return RAGResult(answer=answer, hits=context.chunks)
        """
        ...

    # ── Optional hooks ───────────────────────────────────────────────────

    async def pre_run(self, request: RAGRequest, context: RAGContext) -> None:
        """Hook called BEFORE the main pipeline.

        Override to validate inputs, normalize the query, or set defaults.
        Default implementation is a no-op.
        """
        return

    async def post_run(
        self, request: RAGRequest, context: RAGContext, result: RAGResult
    ) -> RAGResult:
        """Hook called AFTER a successful pipeline run.

        Override to add metadata, filter citations, or attach extra warnings.
        Default implementation returns the result unchanged.
        """
        return result

    async def on_error(
        self, request: RAGRequest, context: RAGContext, error: Exception
    ) -> RAGResult:
        """Hook called when the pipeline encounters an unrecoverable error.

        Override to implement graceful degradation.  The default
        implementation returns a RAGResult with an empty answer and
        the error message in warnings.

        Args:
            request: The original request.
            context:  The context as it was at the time of failure.
            error:    The exception that terminated the pipeline.

        Returns:
            A RAGResult describing the failure (never raise).
        """
        return RAGResult(
            answer="",
            warnings=[f"{type(error).__name__}: {error}"],
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _record(
        self,
        context: RAGContext,
        stage: TraceStage,
        input_summary: str,
        output_summary: str,
        duration_ms: float = 0.0,
    ) -> None:
        """Convenience: record a trace event via context.trace_recorder.

        Does nothing if no trace_recorder is configured.
        """
        recorder = context.trace_recorder
        if recorder is None:
            return
        try:
            recorder(stage, input_summary, output_summary, duration_ms)
        except Exception:
            pass  # Trace failure must never break the pipeline
