import asyncio

from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphRetriever
from src.models.knowledge import ChunkRecord
from src.models.rag import RAGContext, RAGMode, RAGRequest, RAGResult, TraceStage
from src.rag.strategies.graph_rag import GraphRAGStrategy, REFUSAL_ANSWER


def run(coro):
    return asyncio.run(coro)


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "graph.md", "page": 1, "section": "demo"},
    )


def build_strategy():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    return GraphRAGStrategy(GraphRetriever(builder.repository))


def context(llm=None):
    return RAGContext(
        query="RAG",
        chunks=[],
        llm=llm,
        metadata={"trace_id": "trace-graph", "kb_id": "kb-1"},
    )


def test_graph_rag_strategy_local_returns_answer_citations_and_trace():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(),
        )
    )

    assert isinstance(result, RAGResult)
    assert "Graph local search" in result.answer
    assert result.citations
    assert result.citations[0].document_id == "doc-1"
    assert result.citations[0].chunk_id
    assert result.hits[0].source.metadata["graph_scope"] == "local"
    assert [event.stage for event in result.trace] == [
        TraceStage.GRAPH_SEARCH,
        TraceStage.GENERATE,
        TraceStage.COMPLETE,
    ]


def test_graph_rag_strategy_global_returns_report_answer_citations_and_trace():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="overall Retrieval theme",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "global"},
            ),
            context(),
        )
    )

    assert "Graph global search" in result.answer
    assert result.citations
    assert result.hits[0].source.metadata["community_id"]
    assert result.hits[0].source.metadata["report_id"]
    assert result.trace[0].stage == TraceStage.GRAPH_SEARCH
    assert "scope=global" in result.trace[0].input_summary


def test_graph_rag_strategy_auto_selects_global_for_summary_queries():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="summarize the overall Retrieval theme",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )

    assert result.hits[0].source.metadata["graph_scope"] == "global"


def test_graph_rag_strategy_auto_selects_local_by_default():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )

    assert result.hits[0].source.metadata["graph_scope"] == "local"


def test_graph_rag_strategy_no_match_refuses_with_warning():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="Nonexistent",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(),
        )
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.warnings == ["no matching graph entities found"]


def test_graph_rag_strategy_uses_injected_llm_when_available():
    class FakeLLM:
        async def generate_with_tokens(self, prompt, context):
            return "LLM graph answer", {"prompt": 3, "completion": 4, "total": 7}

    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(llm=FakeLLM()),
        )
    )

    assert result.answer == "LLM graph answer"
    assert result.usage == {"prompt": 3, "completion": 4, "total": 7}
