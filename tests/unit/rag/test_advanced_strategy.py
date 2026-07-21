import asyncio
from copy import deepcopy

import pytest

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.rag.naive_support import REFUSAL_ANSWER
from src.rag.strategies.advanced import (
    AdvancedRAGStrategy,
    AdvancedRAGValidationError,
)
from src.retrieval.compressor import ContextCompressor
from tests.rag_fakes import (
    FakeContext,
    FakeContractFactory,
    FakeRAGMode,
    FakeRAGResult,
    FakeRequest,
    FakeTraceStage,
)


def run(coro):
    return asyncio.run(coro)


def make_hit(
    chunk_id,
    score,
    *,
    kb_id="kb-b",
    rank=1,
    text=None,
):
    text = text or f"grounded text for {chunk_id}"
    metadata = {
        "kb_id": kb_id,
        "document_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "filename": f"{chunk_id}.md",
        "source": f"{chunk_id}.md",
        "page": rank,
        "section": "Advanced",
    }
    return RetrievalHit(
        chunk=ChunkRecord(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            kb_id=kb_id,
            text=text,
            index=rank - 1,
            metadata=deepcopy(metadata),
        ),
        score=score,
        rank=rank,
        retriever="vector",
        metadata=deepcopy(metadata),
    )


class MappingRetriever:
    retriever_name = "vector"

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        value = self.mapping[kwargs["query"]]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


class FakeRewriter:
    def __init__(self, values=None, error=None):
        self.values = ["alternative"] if values is None else values
        self.error = error
        self.calls = []

    async def rewrite(self, query, *, max_queries):
        self.calls.append((query, max_queries))
        if self.error:
            raise self.error
        return list(self.values)


class FakeReranker:
    def __init__(self, error=None, empty=False):
        self.error = error
        self.empty = empty
        self.calls = []

    def rerank(self, query, hits, top_k=5):
        self.calls.append((query, [hit.chunk.id for hit in hits], top_k))
        if self.error:
            raise self.error
        if self.empty:
            return []
        return list(reversed(hits))[:top_k]


class FakeCompressor:
    def __init__(self, keep=1, text_limit=12, error=None, empty=False):
        self.keep = keep
        self.text_limit = text_limit
        self.error = error
        self.empty = empty
        self.calls = []

    def compress(self, hits):
        self.calls.append([hit.chunk.id for hit in hits])
        if self.error:
            raise self.error
        if self.empty:
            return []
        result = deepcopy(hits[: self.keep])
        for hit in result:
            hit.chunk.text = hit.chunk.text[: self.text_limit]
        return result


class FakeGenerator:
    def __init__(self, answer="advanced answer", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    async def generate_with_tokens(self, prompt, context, **kwargs):
        self.calls.append({"prompt": prompt, "context": context, **kwargs})
        if self.error:
            raise self.error
        return self.answer, {"prompt": 20, "completion": 5, "total": 25}


def request(options=None, **kwargs):
    return FakeRequest(
        query=kwargs.pop("query", "original"),
        kb_id=kwargs.pop("kb_id", "kb-b"),
        mode=kwargs.pop("mode", FakeRAGMode.ADVANCED),
        options={} if options is None else options,
        **kwargs,
    )


def strategy(**kwargs):
    return AdvancedRAGStrategy(
        contract_factory=FakeContractFactory(),
        **kwargs,
    )


def test_full_advanced_pipeline_fuses_reranks_compresses_and_cites_final_context():
    vector = MappingRetriever(
        {
            "original": [make_hit("a", 0.7), make_hit("b", 0.6, rank=2)],
            "alternative": [
                make_hit("b", 0.9),
                make_hit("c", 0.8, rank=2),
            ],
        }
    )
    hybrid = MappingRetriever(
        {
            "original": [make_hit("a", 0.75)],
            "alternative": [make_hit("c", 0.85)],
        }
    )
    rewriter = FakeRewriter(["alternative", "", "alternative"])
    reranker = FakeReranker()
    compressor = FakeCompressor(keep=1, text_limit=10)
    generator = FakeGenerator()
    context = FakeContext(vector, generator)

    result = run(
        strategy(
            query_rewriter=rewriter,
            hybrid_retriever=hybrid,
            reranker=reranker,
            compressor=compressor,
        ).run(request({"top_k": 3, "rerank_top_k": 2}), context)
    )

    assert isinstance(result, FakeRAGResult)
    assert rewriter.calls == [("original", 3)]
    assert [call["query"] for call in vector.calls] == [
        "original",
        "alternative",
    ]
    assert [call["query"] for call in hybrid.calls] == [
        "original",
        "alternative",
    ]
    assert {call["kb_id"] for call in vector.calls + hybrid.calls} == {"kb-b"}
    assert reranker.calls == [("original", ["b", "c", "a"], 2)]
    assert compressor.calls == [["a", "c"]]
    assert [hit.chunk_id for hit in result.hits] == ["a"]
    assert result.hits[0].content == "grounded t"
    assert result.hits[0].source.metadata["rank"] == 1
    assert result.hits[0].source.metadata["matched_queries"] == ["original"]
    assert [(c.document_id, c.chunk_id) for c in result.citations] == [
        ("doc-a", "a")
    ]
    assert result.citations[0].text_snippet == "grounded t"
    assert "grounded t" in generator.calls[0]["context"]
    assert "grounded text for c" not in generator.calls[0]["context"]
    assert result.usage == {"prompt": 20, "completion": 5, "total": 25}
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.REWRITE,
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.RERANK,
        FakeTraceStage.COMPRESS,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert "before_top=b,c,a" in result.trace[2].input_summary
    assert "after_top=a,c" in result.trace[2].output_summary
    assert "before_chars=" in result.trace[3].input_summary
    assert "after_chars=10" in result.trace[3].output_summary
    assert all(event.duration_ms >= 0 for event in result.trace)


def test_all_advanced_switches_off_behaves_like_naive_and_emits_no_module_traces():
    vector = MappingRetriever({"original": [make_hit("a", 0.8)]})
    rewriter = FakeRewriter()
    hybrid = MappingRetriever({"original": [make_hit("b", 0.9)]})
    reranker = FakeReranker()
    compressor = FakeCompressor()
    generator = FakeGenerator()
    options = {
        "rewrite_enabled": False,
        "multi_query_enabled": False,
        "hybrid_enabled": False,
        "rerank_enabled": False,
        "compression_enabled": False,
    }

    result = run(
        strategy(
            query_rewriter=rewriter,
            hybrid_retriever=hybrid,
            reranker=reranker,
            compressor=compressor,
        ).run(request(options), FakeContext(vector, generator))
    )

    assert rewriter.calls == []
    assert hybrid.calls == []
    assert reranker.calls == []
    assert compressor.calls == []
    assert [hit.chunk_id for hit in result.hits] == ["a"]
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert result.warnings == []


def test_rewrite_only_uses_first_alternative_and_original_remains_in_trace_plan():
    vector = MappingRetriever(
        {
            "original": [make_hit("original-hit", 0.8)],
            "alternative": [make_hit("rewrite-hit", 0.9)],
        }
    )
    options = {
        "multi_query_enabled": False,
        "hybrid_enabled": False,
        "rerank_enabled": False,
        "compression_enabled": False,
    }

    result = run(
        strategy(query_rewriter=FakeRewriter(["alternative"])).run(
            request(options),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [call["query"] for call in vector.calls] == ["alternative"]
    assert [hit.chunk_id for hit in result.hits] == ["rewrite-hit"]
    assert "original" in result.trace[0].output_summary
    assert "alternative" in result.trace[0].output_summary


def test_rewrite_exception_degrades_to_original_query_with_warning():
    vector = MappingRetriever({"original": [make_hit("a", 0.8)]})
    result = run(
        strategy(
            query_rewriter=FakeRewriter(error=TimeoutError("rewrite timeout"))
        ).run(
            request(
                {
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [call["query"] for call in vector.calls] == ["original"]
    assert any("rewrite degraded" in warning for warning in result.warnings)
    assert result.trace[0].stage == FakeTraceStage.REWRITE


def test_partial_query_failure_and_hybrid_failure_both_degrade():
    vector = MappingRetriever(
        {
            "original": RuntimeError("base original failed"),
            "alternative": [make_hit("a", 0.8)],
        }
    )
    hybrid = MappingRetriever(
        {
            "original": RuntimeError("hybrid original failed"),
            "alternative": RuntimeError("hybrid alternative failed"),
        }
    )
    result = run(
        strategy(
            query_rewriter=FakeRewriter(["alternative"]),
            hybrid_retriever=hybrid,
        ).run(
            request({"rerank_enabled": False, "compression_enabled": False}),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["a"]
    assert any("vector query retrieval failed" in w for w in result.warnings)
    assert any("hybrid retrieval degraded" in w for w in result.warnings)


def test_all_retrieval_failures_return_structured_error_without_generator():
    vector = MappingRetriever({"original": RuntimeError("vector offline")})
    generator = FakeGenerator()
    result = run(
        strategy().run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(vector, generator),
        )
    )

    assert generator.calls == []
    assert result.citations == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.ERROR,
    ]
    assert any("retrieve failed" in warning for warning in result.warnings)


def test_reranker_failure_uses_pre_rerank_order_and_records_degradation():
    vector = MappingRetriever(
        {"original": [make_hit("a", 0.9), make_hit("b", 0.8, rank=2)]}
    )
    result = run(
        strategy(reranker=FakeReranker(error=RuntimeError("rerank failed"))).run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["a", "b"]
    assert any("rerank degraded" in warning for warning in result.warnings)
    assert "fallback=pre_rerank" in result.trace[1].output_summary


def test_reranker_changes_order_and_strategy_renumbers_from_one():
    vector = MappingRetriever(
        {"original": [make_hit("a", 0.9), make_hit("b", 0.8, rank=2)]}
    )
    result = run(
        strategy(reranker=FakeReranker()).run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["b", "a"]
    assert [hit.source.metadata["rank"] for hit in result.hits] == [1, 2]
    assert result.hits[0].source.metadata["pre_reranker_rank"] == 2
    assert result.hits[1].source.metadata["pre_reranker_rank"] == 1


def test_compression_failure_preserves_uncompressed_hits_and_metadata():
    vector = MappingRetriever({"original": [make_hit("a", 0.9)]})
    result = run(
        strategy(
            compressor=FakeCompressor(error=RuntimeError("compress failed"))
        ).run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                }
            ),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert result.hits[0].source.document_id == "doc-a"
    assert result.hits[0].source.source_path == "a.md"
    assert result.hits[0].source.page == 1
    assert any("compression degraded" in warning for warning in result.warnings)


def test_existing_context_compressor_enforces_budget_and_preserves_source_metadata():
    vector = MappingRetriever(
        {
            "original": [
                make_hit("a", 0.9, text="A" * 100),
                make_hit("b", 0.8, rank=2, text="B" * 20),
            ]
        }
    )
    result = run(
        strategy(compressor=ContextCompressor(max_tokens=30, chars_per_token=2)).run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                }
            ),
            FakeContext(vector, FakeGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["a"]
    assert result.hits[0].content == "A" * 60 + "..."
    assert result.hits[0].source.document_id == "doc-a"
    assert result.hits[0].source.source_path == "a.md"
    assert result.hits[0].source.page == 1
    assert result.hits[0].source.metadata["rank"] == 1


def test_empty_compression_refuses_without_calling_generator_or_creating_citations():
    generator = FakeGenerator()
    result = run(
        strategy(compressor=FakeCompressor(empty=True)).run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                }
            ),
            FakeContext(
                MappingRetriever({"original": [make_hit("a", 0.9)]}),
                generator,
            ),
        )
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.hits == []
    assert generator.calls == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.COMPRESS,
        FakeTraceStage.COMPLETE,
    ]


@pytest.mark.parametrize(
    ("hits", "threshold", "warning_text"),
    [
        ([], 0.0, "no hits"),
        ([make_hit("low", 0.1)], 0.9, "below score_threshold"),
    ],
)
def test_no_hits_or_all_low_scores_refuse_without_generator(
    hits,
    threshold,
    warning_text,
):
    generator = FakeGenerator()
    options = {
        "score_threshold": threshold,
        "rewrite_enabled": False,
        "multi_query_enabled": False,
        "hybrid_enabled": False,
        "rerank_enabled": False,
        "compression_enabled": False,
    }

    result = run(
        strategy().run(
            request(options),
            FakeContext(MappingRetriever({"original": hits}), generator),
        )
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.hits == []
    assert generator.calls == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.COMPLETE,
    ]
    assert any(warning_text in warning for warning in result.warnings)


def test_generator_failure_records_generate_then_error():
    result = run(
        strategy().run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(
                MappingRetriever({"original": [make_hit("a", 0.9)]}),
                FakeGenerator(error=TimeoutError("generator timeout")),
            ),
        )
    )

    assert result.citations == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.ERROR,
    ]


def test_cross_kb_hit_is_rejected_and_never_cited():
    result = run(
        strategy().run(
            request(
                {
                    "rewrite_enabled": False,
                    "multi_query_enabled": False,
                    "hybrid_enabled": False,
                    "rerank_enabled": False,
                    "compression_enabled": False,
                }
            ),
            FakeContext(
                MappingRetriever(
                    {"original": [make_hit("foreign", 0.9, kb_id="kb-a")]}
                ),
                FakeGenerator(),
            ),
        )
    )

    assert result.citations == []
    assert result.hits == []
    assert any("another kb" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("rag_request", "message"),
    [
        (request(query=" "), "request.query"),
        (request(kb_id=""), "request.kb_id"),
        (request(mode=FakeRAGMode.NAIVE), "mode must be advanced"),
        (request({"top_k": 0}), "top_k"),
        (request({"max_queries": 0}), "max_queries"),
        (request({"rewrite_enabled": "yes"}), "rewrite_enabled"),
        (
            request({"filters": {"kb_id": "kb-a"}}),
            "filters cannot override",
        ),
    ],
)
def test_invalid_advanced_requests_are_explicit(rag_request, message):
    with pytest.raises(AdvancedRAGValidationError, match=message):
        run(
            strategy().run(
                rag_request,
                FakeContext(MappingRetriever({}), FakeGenerator()),
            )
        )


def test_request_options_override_strategy_and_strategy_overrides_context():
    vector = MappingRetriever({"original": [make_hit("a", 0.9)]})
    context = FakeContext(
        vector,
        FakeGenerator(),
        config={"top_k": 4},
    )
    options = {
        "top_k": 2,
        "rewrite_enabled": False,
        "multi_query_enabled": False,
        "hybrid_enabled": False,
        "rerank_enabled": False,
        "compression_enabled": False,
    }

    run(strategy(config={"top_k": 3}).run(request(options), context))

    assert vector.calls[0]["top_k"] == 2
