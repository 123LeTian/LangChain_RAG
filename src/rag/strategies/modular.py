"""
Modular RAG Strategy — Owner: C
Composable RAG pipeline built from swappable ingestion / retrieval / generation modules.

The Modular RAG paradigm lets users compose custom pipelines by assembling modules:
  - QueryRewriteModule: improve the query before retrieval
  - RetrieveModule: vector / hybrid / graph retrieval
  - RerankModule: re-score and filter chunks
  - CompressModule: compress context to fit the LLM window
  - GenerateModule: produce the final answer

A builder pattern (ModularPipelineBuilder) constructs pipelines.
A pipeline is a sequence of Module objects executed in order over RAGContext.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from src.models.rag import (
    RAGChunk,
    RAGContext,
    RAGQuery,
    RAGResponse,
    RAGSource,
    StrategyType,
)
from src.rag.base import (
    GeneratorProtocol,
    GraphRetrieverProtocol,
    RAGStrategyBase,
    RerankerProtocol,
    RetrieverProtocol,
)
from src.rag.registry import StrategyRegistry

logger = logging.getLogger(__name__)


# ── Module Interface ──────────────────────────────────────────────────────


class Module(ABC):
    """Abstract base for a single pipeline module.

    Each module has a name and a run() method that transforms a RAGContext.
    Modules are composable: the output of one becomes the input of the next.
    """

    name: str = "base_module"

    @abstractmethod
    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        """Transform the context. Returns the modified context."""
        ...


# ── Concrete Modules ──────────────────────────────────────────────────────


class QueryRewriteModule(Module):
    """Rewrite/expand the user query for better retrieval recall.

    Uses the LLM to produce a more search-friendly version of the query.
    The original query is preserved in context.metadata.
    """

    name: str = "query_rewrite"

    def __init__(self, llm: GeneratorProtocol) -> None:
        self._llm = llm

    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        t0 = time.perf_counter()
        try:
            prompt = (
                "Rewrite the following user query into a more specific, "
                "search-optimized form. Expand acronyms, add relevant keywords, "
                "and make it more suitable for semantic search. "
                "Output ONLY the rewritten query, nothing else.\n\n"
                f"Original: {query.text}\nRewritten:"
            )
            rewritten = await self._llm.generate(prompt=prompt, context="")
            rewritten = rewritten.strip()
        except Exception as exc:
            logger.warning("Query rewrite failed: %s. Using original query.", exc)
            rewritten = query.text

        context.metadata["original_query"] = query.text
        context.metadata["rewritten_query"] = rewritten
        context.metadata["rewrite_latency_ms"] = (time.perf_counter() - t0) * 1000

        return context


class RetrieveModule(Module):
    """Retrieve chunks using the configured retriever.

    Supports vector, hybrid, and graph retrieval modes.
    """

    name: str = "retrieve"

    def __init__(
        self,
        retrieval: RetrieverProtocol,
        graph: Optional[GraphRetrieverProtocol] = None,
        mode: str = "hybrid",  # vector | hybrid | graph | all
        top_k: int = 10,
    ) -> None:
        self._retrieval = retrieval
        self._graph = graph
        self._mode = mode
        self._top_k = top_k

    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        t0 = time.perf_counter()

        # Use rewritten query if available
        search_query = context.metadata.get("rewritten_query", query.text)

        all_chunks: List[RAGChunk] = []

        if self._mode in ("vector", "hybrid", "all"):
            try:
                result = self._retrieval.retrieve(
                    query=search_query, top_k=self._top_k
                )
                all_chunks.extend(result.chunks)
            except Exception as exc:
                logger.error("Vector retrieval failed in module: %s", exc)

        if self._mode in ("graph", "all") and self._graph:
            try:
                graph_result = self._graph.graph_search(
                    query=search_query, top_k=self._top_k
                )
                all_chunks.extend(graph_result.chunks)
            except Exception as exc:
                logger.error("Graph retrieval failed in module: %s", exc)

        context.chunks = all_chunks
        context.total_candidates = len(all_chunks)
        context.retrieval_latency_ms = (time.perf_counter() - t0) * 1000
        context.metadata["retrieval_mode"] = self._mode
        context.metadata["retrieval_top_k"] = self._top_k

        return context


class RerankModule(Module):
    """Rerank chunks by relevance to improve generation quality."""

    name: str = "rerank"

    def __init__(
        self,
        reranker: RerankerProtocol,
        top_k: int = 5,
    ) -> None:
        self._reranker = reranker
        self._top_k = top_k

    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        if not context.chunks:
            return context

        t0 = time.perf_counter()
        try:
            reranked = self._reranker.rerank(
                query=query.text,
                chunks=context.chunks,
                top_k=min(self._top_k, len(context.chunks)),
            )
            context.chunks = reranked  # type: ignore[assignment]
            context.metadata["reranked"] = True
            context.metadata["rerank_latency_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            logger.warning("Reranking failed: %s. Keeping original order.", exc)
            context.metadata["reranked"] = False

        return context


class CompressModule(Module):
    """Compress/truncate context to fit within the LLM's token window.

    Preserves the top-ranked chunks up to max_tokens characters.
    """

    name: str = "compress"

    def __init__(self, max_tokens: int = 4000, chars_per_token: float = 3.5) -> None:
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        max_chars = int(self._max_tokens * self._chars_per_token)
        total = 0
        kept: List[RAGChunk] = []

        for chunk in context.chunks:
            estimated = len(chunk.content)
            if total + estimated <= max_chars:
                kept.append(chunk)
                total += estimated
            else:
                # Truncate the last chunk to fit
                remaining = max_chars - total
                if remaining > 200:  # Only keep if we can fit a meaningful amount
                    truncated = RAGChunk(
                        content=chunk.content[:remaining] + "...",
                        source=chunk.source,
                    )
                    kept.append(truncated)
                break

        context.metadata["compressed"] = True
        context.metadata["original_chunk_count"] = len(context.chunks)
        context.metadata["kept_chunk_count"] = len(kept)
        context.metadata["total_chars"] = sum(len(c.content) for c in kept)
        context.chunks = kept

        return context


class GenerateModule(Module):
    """Generate the final answer using the LLM with the retrieved context."""

    name: str = "generate"

    def __init__(self, llm: GeneratorProtocol, system_prompt: Optional[str] = None) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or (
            "You are a helpful assistant. Answer the user's question based on "
            "the provided context. If the context is insufficient, say so. "
            "Cite specific sources when possible."
        )

    async def run(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        t0 = time.perf_counter()

        context_text = context.combined_text
        try:
            answer = await self._llm.generate(
                prompt=self._system_prompt,
                context=context_text,
            )
        except Exception as exc:
            logger.error("LLM generation failed in module: %s", exc)
            answer = f"Error generating answer: {exc}"

        context.metadata["generated_answer"] = answer
        context.metadata["generation_latency_ms"] = (time.perf_counter() - t0) * 1000
        context.metadata["context_chars"] = len(context_text)

        return context


# ── Pipeline Builder ──────────────────────────────────────────────────────


class ModularPipeline:
    """A composed sequence of modules executed in order over RAGContext."""

    def __init__(self, modules: Optional[List[Module]] = None) -> None:
        self._modules: List[Module] = list(modules) if modules else []

    @property
    def modules(self) -> List[Module]:
        return list(self._modules)

    async def execute(self, context: RAGContext, query: RAGQuery) -> RAGContext:
        """Run all modules sequentially, passing context through each.

        Each module receives the output of the previous module.
        If a module fails, execution continues with the next module
        (graceful degradation).

        Args:
            context: Initial RAGContext (may be empty).
            query: The original query.

        Returns:
            The final RAGContext after all modules have run.
        """
        for module in self._modules:
            try:
                context = await module.run(context, query)
            except Exception as exc:
                logger.error(
                    "Module '%s' failed: %s. Continuing pipeline.", module.name, exc
                )
                context.metadata[f"module_error_{module.name}"] = str(exc)
        return context


class ModularPipelineBuilder:
    """Builder for constructing ModularPipeline instances.

    Usage:
        pipeline = (
            ModularPipelineBuilder()
            .add(QueryRewriteModule(llm))
            .add(RetrieveModule(retrieval, mode="hybrid"))
            .add(RerankModule(reranker))
            .add(CompressModule(max_tokens=4000))
            .add(GenerateModule(llm))
            .build()
        )
        context = await pipeline.execute(RAGContext(query=q.text), q)
    """

    def __init__(self) -> None:
        self._modules: List[Module] = []

    def add(self, module: Module) -> ModularPipelineBuilder:
        """Append a module to the pipeline. Returns self for chaining."""
        if not isinstance(module, Module):
            raise TypeError(f"Expected Module, got {type(module).__name__}")
        self._modules.append(module)
        return self

    def insert(self, index: int, module: Module) -> ModularPipelineBuilder:
        """Insert a module at a specific position."""
        self._modules.insert(index, module)
        return self

    def remove(self, name: str) -> ModularPipelineBuilder:
        """Remove all modules with the given name."""
        self._modules = [m for m in self._modules if m.name != name]
        return self

    def clear(self) -> ModularPipelineBuilder:
        """Remove all modules."""
        self._modules.clear()
        return self

    def build(self) -> ModularPipeline:
        """Build and return the ModularPipeline."""
        return ModularPipeline(list(self._modules))


# ── Modular RAG Strategy ─────────────────────────────────────────────────


class ModularRAGStrategy(RAGStrategyBase):
    """RAG strategy that uses a composable module pipeline.

    This is the concrete strategy for the Modular RAG paradigm.
    Users can customize the pipeline via the builder or inject a
    pre-built pipeline.

    Auto-registers with the StrategyRegistry.
    """

    strategy_type: StrategyType = StrategyType.MODULAR

    def __init__(
        self,
        retrieval: Optional[RetrieverProtocol] = None,
        llm: Optional[GeneratorProtocol] = None,
        graph: Optional[GraphRetrieverProtocol] = None,
        reranker: Optional[RerankerProtocol] = None,
        pipeline: Optional[ModularPipeline] = None,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config)
        self._retrieval = retrieval
        self._llm = llm
        self._graph = graph
        self._reranker = reranker
        self._pipeline = pipeline

    # ── Dependency injection ───────────────────────────────────────────

    def set_retriever(self, retrieval: RetrieverProtocol) -> None:
        self._retrieval = retrieval
        self._pipeline = None  # Invalidate cached pipeline

    def set_generator(self, llm: GeneratorProtocol) -> None:
        self._llm = llm
        self._pipeline = None

    def set_graph_retriever(self, graph: GraphRetrieverProtocol) -> None:
        self._graph = graph
        self._pipeline = None

    # ── Pipeline ───────────────────────────────────────────────────────

    def _get_pipeline(self) -> ModularPipeline:
        """Return the pipeline, building a default one if none was set."""
        if self._pipeline is not None:
            return self._pipeline
        return self._build_default_pipeline()

    def _build_default_pipeline(self) -> ModularPipeline:
        """Build the default modular pipeline."""
        builder = ModularPipelineBuilder()

        # Add rewrite if LLM is available
        if self._llm:
            builder.add(QueryRewriteModule(self._llm))

        # Add retrieval
        if self._retrieval:
            builder.add(RetrieveModule(self._retrieval, graph=self._graph, mode="hybrid"))
        elif self._graph:
            builder.add(RetrieveModule(
                self._retrieval,  # type: ignore[arg-type] — allowed to be None in this path
                graph=self._graph,
                mode="graph",
            ))

        # Add rerank if available
        if self._reranker:
            builder.add(RerankModule(self._reranker, top_k=5))

        # Add compression
        builder.add(CompressModule(max_tokens=4000))

        # Add generation
        if self._llm:
            builder.add(GenerateModule(self._llm))

        return builder.build()

    def set_pipeline(self, pipeline: ModularPipeline) -> None:
        """Inject a custom pipeline (overrides the default)."""
        self._pipeline = pipeline

    # ── RAGStrategyBase implementation ─────────────────────────────────

    async def retrieve(self, query: RAGQuery) -> RAGContext:
        """Run only the retrieval portion of the pipeline."""
        context = RAGContext(query=query.text)
        pipeline = self._get_pipeline()

        # Find and execute all modules up to (and including) retrieval/rerank
        for module in pipeline.modules:
            if module.name in ("query_rewrite", "retrieve", "rerank", "compress"):
                context = await module.run(context, query)
            if module.name == "generate":
                break  # Stop before generation

        return context

    async def generate(self, context: RAGContext, query: RAGQuery) -> RAGResponse:
        """Run only the generation portion."""
        if not self._llm:
            return RAGResponse(
                query_id=query.query_id,
                answer="[No LLM available for generation]",
                strategy=StrategyType.MODULAR,
                context=context,
            )

        t0 = time.perf_counter()
        answer = await self._llm.generate(
            prompt="You are a helpful assistant. Answer based on the provided context.",
            context=context.combined_text,
        )
        latency = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            strategy=StrategyType.MODULAR,
            context=context,
            latency_ms=latency,
        )

    async def run(self, query: RAGQuery) -> RAGResponse:
        """Execute the full modular pipeline."""
        t0 = time.perf_counter()

        context = RAGContext(query=query.text)
        pipeline = self._get_pipeline()
        context = await pipeline.execute(context, query)

        # Extract answer from context metadata (set by GenerateModule)
        answer = context.metadata.get("generated_answer", "")
        latency = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            query_id=query.query_id,
            answer=answer,
            strategy=StrategyType.MODULAR,
            context=context,
            latency_ms=latency,
            metadata={
                "pipeline_modules": [m.name for m in pipeline.modules],
                "pipeline_latencies": {
                    k: v
                    for k, v in context.metadata.items()
                    if k.endswith("_latency_ms")
                },
            },
        )

    async def stream(self, query: RAGQuery) -> AsyncIterator[Any]:
        """Streaming execution — yields events for each module."""
        # Simple implementation: yield module start/end events
        context = RAGContext(query=query.text)
        pipeline = self._get_pipeline()

        for module in pipeline.modules:
            # Yield module-start event
            yield {"type": "module_start", "module": module.name}

            context = await module.run(context, query)

            yield {"type": "module_end", "module": module.name}

        answer = context.metadata.get("generated_answer", "")
        yield {"type": "done", "answer": answer}


# ── Auto-register ─────────────────────────────────────────────────────────

# Register with the global registry (safe to call multiple times — overwrites)
try:
    registry = StrategyRegistry()
    registry.register(
        StrategyType.MODULAR,
        ModularRAGStrategy,
        metadata={
            "description": "Composable RAG pipeline built from swappable modules",
            "paradigm": "Modular",
        },
    )
except Exception:
    pass  # Registration will be handled by auto-discovery
