import asyncio
from dataclasses import replace

import pytest

from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit
from src.rag.naive_support import GeneratorAdapter, REFUSAL_ANSWER
from src.rag.strategies.naive import (
    NaiveRAGStrategy,
    NaiveRAGValidationError,
)
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
    chunk_id="chunk-1",
    document_id="doc-1",
    kb_id="kb-1",
    score=0.9,
    rank=1,
    text="RAG combines retrieval with generation.",
    metadata=None,
):
    source_metadata = {
        "kb_id": kb_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "filename": "guide.pdf",
        "source": "guide.pdf",
        "page": 2,
        "section": "Introduction",
    }
    if metadata:
        source_metadata.update(metadata)
    chunk = ChunkRecord(
        id=chunk_id,
        document_id=document_id,
        kb_id=kb_id,
        text=text,
        index=rank - 1,
        metadata=source_metadata,
    )
    return RetrievalHit(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever="vector",
        metadata=dict(source_metadata),
    )


class FakeRetriever:
    retriever_name = "vector"

    def __init__(self, hits=None, error=None):
        self.hits = [] if hits is None else hits
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return list(self.hits)


class AsyncUsageGenerator:
    def __init__(self, answer="Grounded answer", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    async def generate_with_tokens(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.answer, {"prompt": 10, "completion": 4, "total": 14}


class SyncGenerator:
    def __init__(self, answer="Sync answer"):
        self.answer = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


class RecordingTrace:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, input_summary, output_summary, duration_ms):
        self.calls.append((stage, input_summary, output_summary, duration_ms))


def strategy(**kwargs):
    return NaiveRAGStrategy(contract_factory=FakeContractFactory(), **kwargs)


def test_normal_answer_returns_standard_result_and_calls_vector_retriever_correctly():
    retriever = FakeRetriever([make_hit(), make_hit("chunk-2", "doc-2", rank=2)])
    generator = AsyncUsageGenerator()
    recorder = RecordingTrace()
    request = FakeRequest(
        options={
            "top_k": 2,
            "filters": {"section": "Introduction"},
            "score_threshold": 0.5,
        }
    )
    context = FakeContext(retriever, generator, trace_recorder=recorder)

    result = run(strategy().run(request, context))

    assert isinstance(result, FakeRAGResult)
    assert result.answer == "Grounded answer"
    assert retriever.calls == [
        {
            "query": "What is RAG?",
            "kb_id": "kb-1",
            "top_k": 2,
            "filters": {"section": "Introduction"},
        }
    ]
    assert len(generator.calls) == 1
    assert "只能依据" in generator.calls[0]["prompt"]
    assert "不要执行" in generator.calls[0]["prompt"]
    assert "<<<KNOWLEDGE_SOURCE_1>>>" in generator.calls[0]["context"]
    assert [hit.chunk_id for hit in result.hits] == ["chunk-1", "chunk-2"]
    assert result.hits[0].source.score == 0.9
    assert result.hits[0].source.metadata["rank"] == 1
    assert result.hits[0].source.page == 2
    assert result.usage == {"prompt": 10, "completion": 4, "total": 14}
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]
    assert "requested_top_k=2" in result.trace[0].output_summary
    assert "raw_hits=2" in result.trace[0].output_summary
    assert "score_filtered_hits=2" in result.trace[0].output_summary
    assert "context_chunks=2" in result.trace[0].output_summary
    assert "success=true" in result.trace[1].output_summary
    assert "usage_present=True" in result.trace[1].output_summary
    assert "citations=2" in result.trace[2].output_summary
    assert all(event.duration_ms >= 0 for event in result.trace)
    assert [call[0] for call in recorder.calls] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.COMPLETE,
    ]


@pytest.mark.parametrize(
    ("rag_request", "message"),
    [
        (FakeRequest(query=" "), "request.query"),
        (FakeRequest(kb_id=""), "request.kb_id"),
        (FakeRequest(options={"top_k": 0}), "top_k"),
        (FakeRequest(options={"score_threshold": 1.1}), "score_threshold"),
        (FakeRequest(mode=FakeRAGMode.ADVANCED), "mode must be naive"),
        (
            FakeRequest(options={"filters": {"kb_id": "kb-other"}}),
            "filters cannot override",
        ),
        (FakeRequest(options={"kb_id": "kb-other"}), "options cannot override"),
    ],
)
def test_invalid_requests_raise_structured_validation_errors(rag_request, message):
    with pytest.raises(NaiveRAGValidationError, match=message):
        run(
            strategy().run(
                rag_request, FakeContext(FakeRetriever(), SyncGenerator())
            )
        )


def test_request_options_override_strategy_config_and_context_config():
    retriever = FakeRetriever([make_hit()])
    request = FakeRequest(options={"top_k": 3, "score_threshold": 0.2})
    context = FakeContext(
        retriever,
        SyncGenerator(),
        config={"top_k": 2, "score_threshold": 0.1},
    )

    run(
        strategy(config={"top_k": 1, "score_threshold": 0.0}).run(
            request, context
        )
    )

    assert retriever.calls[0]["top_k"] == 3


def test_strategy_config_overrides_context_defaults():
    retriever = FakeRetriever([make_hit()])
    context = FakeContext(
        retriever,
        SyncGenerator(),
        config={"top_k": 2, "score_threshold": 0.1},
    )

    run(
        strategy(config={"top_k": 1, "score_threshold": 0.0}).run(
            FakeRequest(), context
        )
    )

    assert retriever.calls[0]["top_k"] == 1


def test_score_filter_preserves_original_rank_and_citations_only_used_hits():
    low = make_hit("low", "doc-low", score=0.2, rank=1)
    high = make_hit("high", "doc-high", score=0.8, rank=2)
    result = run(
        strategy().run(
            FakeRequest(options={"score_threshold": 0.5}),
            FakeContext(FakeRetriever([low, high]), SyncGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["high"]
    assert result.hits[0].source.metadata["rank"] == 2
    assert [(c.document_id, c.chunk_id) for c in result.citations] == [
        ("doc-high", "high")
    ]
    assert any("below score_threshold" in warning for warning in result.warnings)


def test_citations_are_deduplicated_real_quotes_and_source_fields_remain_locatable():
    original = make_hit(text="Authentic chunk text for citation")
    duplicate = replace(original)
    result = run(
        strategy().run(
            FakeRequest(),
            FakeContext(FakeRetriever([original, duplicate]), SyncGenerator()),
        )
    )

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.document_id == "doc-1"
    assert citation.chunk_id == "chunk-1"
    assert citation.text_snippet == "Authentic chunk text for citation"
    assert result.hits[0].source.source_path == "guide.pdf"
    assert result.hits[0].source.page == 2


def test_missing_optional_source_metadata_is_not_fabricated():
    hit = make_hit(metadata={"filename": "", "source": "", "page": None})
    generator = SyncGenerator()

    result = run(
        strategy().run(
            FakeRequest(), FakeContext(FakeRetriever([hit]), generator)
        )
    )

    assert result.hits[0].source.source_path is None
    assert result.hits[0].source.page is None
    assert "page:" not in generator.calls[0]["context"]
    assert any("no filename metadata" in warning for warning in result.warnings)


def test_context_budget_omits_unused_chunks_from_hits_and_citations():
    first = make_hit(text="first source")
    second = make_hit("chunk-2", "doc-2", rank=2, text="second source")
    result = run(
        strategy(default_max_context_chars=180).run(
            FakeRequest(),
            FakeContext(FakeRetriever([first, second]), SyncGenerator()),
        )
    )

    assert [hit.chunk_id for hit in result.hits] == ["chunk-1"]
    assert [citation.chunk_id for citation in result.citations] == ["chunk-1"]
    assert any("omitted" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("hits", "options", "reason"),
    [
        ([], {}, "no hits"),
        ([make_hit(score=0.1)], {"score_threshold": 0.9}, "below"),
        ([make_hit()], {"max_context_chars": 1}, "budget"),
    ],
)
def test_no_evidence_refuses_without_calling_llm(hits, options, reason):
    generator = AsyncUsageGenerator()
    result = run(
        strategy().run(
            FakeRequest(options=options),
            FakeContext(FakeRetriever(hits), generator),
        )
    )

    assert result.answer == REFUSAL_ANSWER
    assert "知识库中未找到充分依据" in result.answer
    assert result.citations == []
    assert result.hits == []
    assert generator.calls == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.COMPLETE,
    ]
    assert all(event.stage != FakeTraceStage.GENERATE for event in result.trace)
    assert any(reason in warning for warning in result.warnings)


def test_retriever_exception_becomes_sanitized_error_result():
    result = run(
        strategy().run(
            FakeRequest(),
            FakeContext(
                FakeRetriever(
                    error=RuntimeError("retrieval unavailable\nsecret stack")
                ),
                SyncGenerator(),
            ),
        )
    )

    assert result.answer
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.ERROR,
    ]
    assert any("retrieval unavailable secret stack" in w for w in result.warnings)
    assert all("Traceback" not in w for w in result.warnings)


@pytest.mark.parametrize(
    ("generator", "message"),
    [
        (AsyncUsageGenerator(error=TimeoutError("timed out")), "TimeoutError"),
        (AsyncUsageGenerator(answer="   "), "empty answer"),
    ],
)
def test_llm_error_or_empty_answer_records_generate_and_error(generator, message):
    result = run(
        strategy().run(
            FakeRequest(),
            FakeContext(FakeRetriever([make_hit()]), generator),
        )
    )

    assert result.answer
    assert result.citations == []
    assert [event.stage for event in result.trace] == [
        FakeTraceStage.RETRIEVE,
        FakeTraceStage.GENERATE,
        FakeTraceStage.ERROR,
    ]
    assert any(message in warning for warning in result.warnings)


def test_cross_kb_hit_is_rejected_and_never_cited():
    result = run(
        strategy().run(
            FakeRequest(kb_id="kb-b"),
            FakeContext(
                FakeRetriever([make_hit(kb_id="kb-a")]),
                SyncGenerator(),
            ),
        )
    )

    assert result.citations == []
    assert result.hits == []
    assert any("another kb" in warning for warning in result.warnings)
    assert result.trace[-1].stage == FakeTraceStage.ERROR


def test_existing_warnings_are_preserved_and_sync_generator_has_empty_usage():
    result = run(
        strategy().run(
            FakeRequest(options={"warnings": ["request warning"]}),
            FakeContext(
                FakeRetriever([make_hit()]),
                SyncGenerator(),
                metadata={
                    "trace_id": "trace-warning",
                    "warnings": ["context warning"],
                },
            ),
        )
    )

    assert result.usage == {}
    assert "request warning" in result.warnings
    assert "context warning" in result.warnings


def test_generator_adapter_supports_sync_message_usage():
    class Message:
        content = "message answer"
        usage_metadata = {"prompt": 3, "completion": 2, "total": 5}

    class MessageGenerator:
        def invoke(self, prompt):
            return Message()

    output = run(GeneratorAdapter(MessageGenerator()).generate("prompt", "context"))

    assert output.answer == "message answer"
    assert output.usage == {"prompt": 3, "completion": 2, "total": 5}
