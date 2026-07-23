import asyncio

from src.api.api_models import RAGMode, RAGRequest, TraceStage
from src.api.real_rag_service import RealRAGService
from src.models.knowledge import ChunkRecord
from src.models.rag import RAGChunk, RAGContext, RAGSource
from src.models.schemas import RetrievalHit as SchemaRetrievalHit


def run(coro):
    return asyncio.run(coro)


def _schema_hit(chunk_id: str, score: float, text: str) -> SchemaRetrievalHit:
    return SchemaRetrievalHit(
        chunk=ChunkRecord(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            kb_id="kb-1",
            text=text,
            index=0,
            metadata={
                "document_id": f"doc-{chunk_id}",
                "chunk_id": chunk_id,
                "filename": f"{chunk_id}.md",
                "kb_id": "kb-1",
                "page": 1,
            },
        ),
        score=score,
        rank=1,
        retriever="fake",
        metadata={},
    )


class FakeRetriever:
    def retrieve(self, _query: str, _top_k: int):
        return [
            _schema_hit(
                "main-business",
                0.92,
                "Moutai main business is production and sales of Moutai liquor.",
            )
        ]


class FakeGraphRetriever:
    def __init__(self):
        self.calls = []

    def graph_search(self, query: str, top_k: int = 5, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return RAGContext(
            query=query,
            chunks=[
                RAGChunk(
                    chunk_id="graph-sales",
                    content="Graph context: direct channel contains i Moutai.",
                    source=RAGSource(
                        document_id="doc-graph",
                        source_path="graph.md",
                        page=2,
                        score=0.88,
                        metadata={"rank": 1, "graph_scope": "local"},
                    ),
                )
            ],
        )

    def get_entity(self, entity_id: str):
        return None

    def get_community_report(self, community_id: str):
        return None


class FakeLLM:
    def invoke(self, prompt: str):
        if "Summarize the following documents concisely." in prompt:
            return type(
                "Response",
                (),
                {"content": "Graph summary: direct channel contains i Moutai."},
            )()
        assert "Query:" in prompt
        assert "main business" in prompt or "Moutai" in prompt
        return type(
            "Response",
            (),
            {"content": "Moutai main business is production and sales of liquor."},
        )()


def test_agentic_mode_enters_agent_workflow_and_skips_plain_pipeline():
    service = RealRAGService()
    service._initialized = True
    service.retriever = FakeRetriever()
    service.graph_retriever = None
    service.graph_kb_id = "kb-1"
    service._create_llm = lambda *args, **kwargs: FakeLLM()

    result = run(
        service.query(
            RAGRequest(
                mode=RAGMode.AGENTIC,
                query="main business?",
                options={"top_k": 5, "max_steps": 6},
            )
        )
    )

    stages = [event.stage for event in result.trace]
    assert result.mode == RAGMode.AGENTIC
    assert TraceStage.INTENT in stages
    assert TraceStage.TOOL_CALL in stages
    assert TraceStage.GENERATE in stages
    assert TraceStage.COMPLETE in stages
    assert TraceStage.RETRIEVE not in stages
    assert TraceStage.RERANK not in stages
    assert TraceStage.COMPRESS not in stages
    assert result.hits[0].chunk_id == "main-business"
    assert result.citations[0].filename == "main-business.md"
    assert result.citations[0].page == 1
    assert result.usage["detail"]["pipeline_config"]["mode"] == "agentic"
    assert result.usage["detail"]["pipeline_config"]["max_steps"] == 6
    assert "max_steps=6" in result.trace[-1].input_summary


def test_agentic_returns_chinese_config_error_when_all_retrieval_tools_disabled():
    service = RealRAGService()

    result = run(
        service.query(
            RAGRequest(
                mode=RAGMode.AGENTIC,
                query="main business?",
                options={
                    "agent_vector_enabled": False,
                    "agent_graph_enabled": False,
                },
            )
        )
    )

    assert result.mode == RAGMode.AGENTIC
    assert result.answer.startswith("[配置错误]")
    assert "Vector Tool 或 Graph Tool" in result.answer
    assert result.usage["error"] == "invalid_config"


def test_agentic_vector_switch_disables_vector_tool_and_uses_graph_fallback():
    graph = FakeGraphRetriever()
    service = RealRAGService()
    service._initialized = True
    service.retriever = FakeRetriever()
    service.graph_retriever = graph
    service.graph_kb_id = "kb-1"
    service._create_llm = lambda *args, **kwargs: FakeLLM()

    result = run(
        service.query(
            RAGRequest(
                mode=RAGMode.AGENTIC,
                query="main business?",
                options={
                    "top_k": 3,
                    "agent_vector_enabled": False,
                    "agent_graph_enabled": True,
                },
            )
        )
    )

    tool_events = [
        event for event in result.trace if event.stage == TraceStage.TOOL_CALL
    ]
    assert [event.input_summary for event in tool_events] == [
        "tool=graph_search; query=main business?, top_k=3",
        "tool=document_summary; chunks=1",
    ]
    assert "selected_tools=['graph_search', 'document_summary']" in result.trace[0].output_summary
    assert "routed_tools=['vector_search']" in result.trace[0].output_summary
    assert result.citations[0].filename == "graph.md"
    assert result.citations[0].page == 2
    assert graph.calls[0]["top_k"] == 3
    assert result.usage["detail"]["pipeline_config"]["vector_tool"] is False
    assert result.usage["detail"]["pipeline_config"]["graph_tool"] is True
