import asyncio

from src.api.api_models import RAGMode, RAGRequest, TraceStage
from src.api.real_rag_service import RealRAGService
from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphRetriever
from src.models.knowledge import ChunkRecord
from src.rag.strategies.graph_rag import GraphRAGStrategy


def run(coro):
    return asyncio.run(coro)


def make_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "moutai.md", "page": 1, "section": "demo"},
    )


def make_graph_service() -> RealRAGService:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    builder = GraphIndexBuilder()
    chunks = [
        make_chunk("chunk-1", "Moutai direct channel contains self-operated stores."),
        make_chunk("chunk-2", "Moutai wholesale agency channel contains distributors."),
        make_chunk("chunk-3", "Important matters relate to sales channels and governance."),
    ]
    builder.build_from_chunks(kb_id="kb-1", chunks=chunks)

    service = RealRAGService()
    service._initialized = True
    service.chunks = chunks
    service.graph_kb_id = "kb-1"
    service.graph_builder = builder
    service.graph_retriever = GraphRetriever(builder.repository)
    service.graph_strategy = GraphRAGStrategy(service.graph_retriever)
    service._create_llm = lambda *args, **kwargs: None
    return service


def test_graph_mode_enters_graph_search_and_skips_vector_pipeline():
    result = run(
        make_graph_service().query(
            RAGRequest(
                mode=RAGMode.GRAPH,
                query="贵州茅台的直销渠道和批发代理渠道分别包括哪些主体？",
                options={"top_k": 5},
            )
        )
    )

    stages = [event.stage for event in result.trace]
    assert result.mode == RAGMode.GRAPH
    assert TraceStage.GRAPH_SEARCH in stages
    assert TraceStage.RETRIEVE not in stages
    assert TraceStage.RERANK not in stages
    assert TraceStage.COMPRESS not in stages


def test_graph_mode_supports_local_and_global_scope_trace():
    service = make_graph_service()

    local = run(
        service.query(
            RAGRequest(
                mode=RAGMode.GRAPH,
                query="Moutai direct channel",
                options={"top_k": 5, "graph_scope": "local"},
            )
        )
    )
    global_result = run(
        service.query(
            RAGRequest(
                mode=RAGMode.GRAPH,
                query="overall Moutai themes",
                options={"top_k": 5, "graph_scope": "global"},
            )
        )
    )

    local_graph_event = next(
        event for event in local.trace if event.stage == TraceStage.GRAPH_SEARCH
    )
    global_graph_event = next(
        event for event in global_result.trace if event.stage == TraceStage.GRAPH_SEARCH
    )
    assert "scope=local" in local_graph_event.input_summary
    assert "scope=global" in global_graph_event.input_summary


def test_graph_mode_falls_back_to_template_when_llm_creation_fails():
    service = make_graph_service()
    service._create_llm = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("missing model config")
    )

    result = run(
        service.query(
            RAGRequest(
                mode=RAGMode.GRAPH,
                query="Moutai direct channel",
                options={"top_k": 5, "graph_scope": "local"},
            )
        )
    )

    assert result.hits
    assert "\u5c40\u90e8\u68c0\u7d22" in result.answer
    assert result.warnings == ["graph generation fallback: RuntimeError"]
