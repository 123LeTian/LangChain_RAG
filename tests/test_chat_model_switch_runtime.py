from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.api_models import RAGMode, RAGRequest, RAGResult
from src.api.app import app
from src.api.dependencies import (
    get_chat_application_service,
    get_chat_message_service,
    get_chat_session_service,
)
from src.api.real_rag_service import RealRAGService
from src.chat.chat_application_service import ChatApplicationService
from src.chat.memory_service import MemoryService
from src.chat.message_service import MessageService
from src.chat.model_registry import ModelRegistry
from src.chat.rag_gateway import RAGGateway
from src.chat.session_service import SessionService
from src.chat_storage.sqlite_chat_store import SQLiteChatStore
from src.models.knowledge import ChunkRecord
from src.models.schemas import RetrievalHit as SchemaRetrievalHit


def test_real_rag_naive_forces_plain_pipeline():
    service = RealRAGService()

    config = service._parse_config(
        RAGRequest(
            query="What is the 2025 revenue?",
            mode=RAGMode.NAIVE,
            options={
                "rewrite_enabled": True,
                "retrieve_enabled": True,
                "rerank_enabled": True,
                "compress_enabled": True,
                "verify_enabled": True,
                "top_k": 5,
                "rerank_top_k": 5,
            },
        )
    )

    assert config == {
        "rewrite": False,
        "retrieve": True,
        "rerank": False,
        "compress": False,
        "verify": False,
        "top_k": 5,
        "rerank_top_k": 5,
        "score_threshold": 0.0,
    }


def test_real_rag_advanced_forces_enhanced_pipeline():
    service = RealRAGService()

    config = service._parse_config(
        RAGRequest(
            query="Compare 2025 and 2024 metrics.",
            mode=RAGMode.ADVANCED,
            options={
                "rewrite_enabled": False,
                "retrieve_enabled": False,
                "rerank_enabled": False,
                "compress_enabled": False,
                "verify_enabled": True,
                "top_k": 10,
                "rerank_top_k": 5,
                "score_threshold": 0.2,
            },
        )
    )

    assert config == {
        "rewrite": True,
        "retrieve": True,
        "rerank": True,
        "compress": True,
        "verify": True,
        "top_k": 10,
        "rerank_top_k": 5,
        "score_threshold": 0.2,
    }


@pytest.mark.asyncio
async def test_real_rag_modular_config_error_is_chinese():
    service = RealRAGService()

    result = await service.query(
        RAGRequest(
            query="贵州茅台的主要业务是什么？",
            mode=RAGMode.MODULAR,
            options={
                "retrieve_enabled": False,
                "rerank_enabled": True,
            },
        )
    )

    assert result.answer.startswith("[配置错误]")
    assert "模块“rerank（重排）”依赖模块“retrieve（检索）”" in result.answer
    assert "[Config Error]" not in result.answer


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
            },
        ),
        score=score,
        rank=1,
        retriever="fake",
        metadata={},
    )


@pytest.mark.asyncio
async def test_real_rag_advanced_uses_advanced_prompt_and_score_threshold(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.prompts: List[str] = []

        def invoke(self, prompt: str):
            self.prompts.append(prompt)
            return type("Response", (), {"content": "advanced answer"})()

    class FakeRetriever:
        def retrieve(self, _query: str, _top_k: int):
            return [
                _schema_hit("low", 0.1, "low score context"),
                _schema_hit("high", 0.3, "high score context"),
            ]

    class FakeReranker:
        def rerank(self, _query: str, hits: list[SchemaRetrievalHit], _top_k: int):
            return hits

    class FakeCompressor:
        def compress(self, hits: list[SchemaRetrievalHit]):
            return hits

    class FakeCitationBuilder:
        def build_citations(self, _hits: list[SchemaRetrievalHit]):
            return []

    async def fake_rewrite(self, query: str, max_queries: int = 3):
        return [query]

    from src.api.llm_query_rewriter import LLMQueryRewriter

    monkeypatch.setattr(LLMQueryRewriter, "rewrite", fake_rewrite)

    llm = FakeLLM()
    service = RealRAGService()
    service._initialized = True
    service.retriever = FakeRetriever()
    service.reranker = FakeReranker()
    service.compressor = FakeCompressor()
    service.citation_builder = FakeCitationBuilder()
    service.build_naive_prompt = lambda query, context: f"naive::{query}::{context}"
    service.build_advanced_prompt = lambda query, context: f"advanced::{query}::{context}"
    service._create_llm = lambda *args, **kwargs: llm

    result = await service.query(
        RAGRequest(
            query="贵州茅台酒核心产区为什么说不可复制？",
            mode=RAGMode.ADVANCED,
            options={"top_k": 5, "rerank_top_k": 5, "score_threshold": 0.2},
        )
    )

    assert llm.prompts == ["advanced::贵州茅台酒核心产区为什么说不可复制？::[Chunk 1] high score context"]
    assert [hit.chunk_id for hit in result.hits] == ["high"]
    assert result.usage["pipeline_config"]["score_threshold"] == 0.2
    assert "score_threshold=0.20" in result.trace[1].input_summary
    assert any("score_filtered=1" in event.output_summary for event in result.trace)


@pytest.mark.asyncio
async def test_real_rag_score_threshold_respects_ui_precision(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt: str):
            return type("Response", (), {"content": "advanced answer"})()

    class FakeRetriever:
        def retrieve(self, _query: str, _top_k: int):
            return [_schema_hit("precise", 0.399, "marketing network 66 countries")]

    class FakeReranker:
        def rerank(self, _query: str, hits: list[SchemaRetrievalHit], _top_k: int):
            for hit in hits:
                hit.score = 0.4996
            return hits

    class FakeCompressor:
        def compress(self, hits: list[SchemaRetrievalHit]):
            return hits

    class FakeCitationBuilder:
        def build_citations(self, _hits: list[SchemaRetrievalHit]):
            return []

    async def fake_rewrite(self, query: str, max_queries: int = 3):
        return [query]

    from src.api.llm_query_rewriter import LLMQueryRewriter

    monkeypatch.setattr(LLMQueryRewriter, "rewrite", fake_rewrite)

    service = RealRAGService()
    service._initialized = True
    service.retriever = FakeRetriever()
    service.reranker = FakeReranker()
    service.compressor = FakeCompressor()
    service.citation_builder = FakeCitationBuilder()
    service.build_naive_prompt = lambda query, context: f"naive::{query}::{context}"
    service.build_advanced_prompt = lambda query, context: f"advanced::{query}::{context}"
    service._create_llm = lambda *args, **kwargs: FakeLLM()

    result = await service.query(
        RAGRequest(
            query="贵州茅台的营销网络覆盖到什么范围？",
            mode=RAGMode.ADVANCED,
            options={"top_k": 5, "rerank_top_k": 5, "score_threshold": 0.5},
        )
    )

    assert result.answer == "advanced answer"
    assert [hit.chunk_id for hit in result.hits] == ["precise"]


class RecordingRAGService:
    def __init__(self):
        self.requests: List[RAGRequest] = []

    async def query(self, request: RAGRequest) -> RAGResult:
        self.requests.append(request)
        return RAGResult(
            answer="runtime answer",
            citations=[],
            hits=[],
            trace=[],
            usage={},
            mode=request.mode,
        )


def _read_sse(response) -> List[Dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def runtime_client(tmp_path):
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    memory_service = MemoryService(message_service)
    registry = ModelRegistry(custom_path=tmp_path / "custom_models.json")
    registry.create_model({
        "id": "deepseek-reasoner",
        "provider": "deepseek",
        "display_name": "DeepSeek Reasoner",
        "model_name": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "enabled": True,
    })
    registry.create_model({
        "id": "gpt-4o",
        "provider": "openai",
        "display_name": "GPT-4o",
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "enabled": False,
    })
    rag_service = RecordingRAGService()
    app_service = ChatApplicationService(
        session_service,
        message_service,
        memory_service,
        RAGGateway(rag_service),
        registry,
    )

    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as test_client:
        yield test_client, rag_service

    app.dependency_overrides.clear()


def test_chat_stream_request_model_reaches_rag_request_options(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={
            "question": "hello",
            "model_id": "deepseek-reasoner",
            "rag_mode": "naive",
            "temperature": 0.3,
        },
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    options = rag_service.requests[-1].options
    model_config = options["model"]
    assert model_config["id"] == "deepseek-reasoner"
    assert model_config["model_name"] == "deepseek-reasoner"
    assert options["temperature"] == 0.3
    assert events[-1]["type"] == "done"


def test_chat_stream_uses_session_model_when_request_omits_model(runtime_client):
    client, rag_service = runtime_client
    session = client.post(
        "/api/chat/sessions",
        json={"title": "chat", "model_id": "deepseek-reasoner"},
    ).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello", "rag_mode": "naive"},
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    assert rag_service.requests[-1].options["model"]["id"] == "deepseek-reasoner"


def test_chat_stream_uses_default_model_without_request_or_session_model(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello"},
    ) as response:
        assert response.status_code == 200
        events = _read_sse(response)

    assert not rag_service.requests
    assert events[-1]["content"] == "您好，我是本地默认模型，不会回答任何问题，只用于测试，请在模型配置页面添加您的模型"
    assert events[-1]["model_id"] == "mock-chat"


def test_chat_rag_request_uses_original_question_not_history_context(runtime_client):
    client, rag_service = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    for question in ["First question", "Second question"]:
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={
                "question": question,
                "model_id": "deepseek-reasoner",
                "rag_mode": "naive",
            },
        ) as response:
            assert response.status_code == 200
            _read_sse(response)

    request = rag_service.requests[-1]
    assert request.query == "Second question"
    assert "First question" not in request.query
    assert "最近历史" not in request.query


def test_missing_key_stream_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    RealRAGServiceDouble = type(
        "RealRAGService",
        (),
        {
            "__module__": "src.api.real_rag_service",
            "query": lambda self, _request: None,
        },
    )
    store = SQLiteChatStore(tmp_path / "chat.db")
    session_service = SessionService(store)
    message_service = MessageService(store)
    registry = ModelRegistry(custom_path=tmp_path / "custom_missing_key_models.json")
    registry.create_model({
        "id": "deepseek-chat",
        "provider": "deepseek",
        "display_name": "DeepSeek Chat",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "enabled": True,
    })
    app_service = ChatApplicationService(
        session_service,
        message_service,
        MemoryService(message_service),
        RAGGateway(RealRAGServiceDouble()),
        registry,
    )
    app.dependency_overrides[get_chat_session_service] = lambda: session_service
    app.dependency_overrides[get_chat_message_service] = lambda: message_service
    app.dependency_overrides[get_chat_application_service] = lambda: app_service

    with TestClient(app) as client:
        session = client.post("/api/chat/sessions", json={"title": "chat"}).json()
        with client.stream(
            "POST",
            f"/api/chat/sessions/{session['id']}/stream",
            json={"question": "hello", "model_id": "deepseek-chat"},
        ) as response:
            assert response.status_code == 200
            events = _read_sse(response)

    app.dependency_overrides.clear()
    assert events[-1]["type"] == "error"
    assert "DEEPSEEK_API_KEY" in events[-1]["message"]
    messages = message_service.list_messages(session["id"])
    assert [message.role for message in messages] == ["user"]


def test_disabled_model_stream_returns_validation_error(runtime_client):
    client, _ = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    response = client.post(
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello", "model_id": "gpt-4o"},
    )

    assert response.status_code == 422


def test_model_trace_contains_model_id_provider_but_not_secret(runtime_client):
    client, _ = runtime_client
    session = client.post("/api/chat/sessions", json={"title": "chat"}).json()

    with client.stream(
        "POST",
        f"/api/chat/sessions/{session['id']}/stream",
        json={"question": "hello", "model_id": "deepseek-reasoner"},
    ) as response:
        events = _read_sse(response)

    trace_events = [event for event in events if event["type"] == "trace"]
    trace_text = json.dumps(trace_events, ensure_ascii=False)
    assert "deepseek-reasoner" in trace_text
    assert "deepseek" in trace_text
    assert "DEEPSEEK_API_KEY" not in trace_text
    assert "secret-value" not in trace_text


@pytest.mark.asyncio
async def test_old_rag_request_without_model_config_uses_default_deepseek_path(monkeypatch):
    class FakeResponse:
        content = "runtime answer"

    calls: List[Dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def invoke(self, _prompt: str) -> FakeResponse:
            return FakeResponse()

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-real")
    service = RealRAGService()

    result = await service.query(
        RAGRequest(query="hello", mode=RAGMode.NAIVE, options={})
    )

    assert result.answer == "runtime answer"
    assert calls[-1]["model"] == "deepseek-chat"
    assert calls[-1]["base_url"] == "https://api.deepseek.com/v1"


@pytest.mark.asyncio
async def test_real_rag_request_model_config_drives_generation_and_verify(monkeypatch):
    class FakeResponse:
        content = "runtime answer"

        usage_metadata = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    calls: List[Dict[str, Any]] = []
    invoked_prompts: List[str] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def invoke(self, prompt: str) -> FakeResponse:
            invoked_prompts.append(prompt)
            return FakeResponse()

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("CUSTOM_RAG_API_KEY", "custom-secret")
    service = RealRAGService()
    service._initialized = True
    service.build_naive_prompt = lambda query, context: f"query={query}\ncontext={context}"

    result = await service.query(
        RAGRequest(
            query="Explain quarterly margin trends",
            mode=RAGMode.MODULAR,
            options={
                "model": {
                    "id": "custom-rag",
                    "provider": "openai",
                    "display_name": "Custom RAG",
                    "model_name": "custom-rag-model",
                    "base_url": "https://llm.example.test/v1",
                    "api_key_env": "CUSTOM_RAG_API_KEY",
                    "enabled": True,
                    "supports_stream": True,
                },
                "rewrite_enabled": False,
                "retrieve_enabled": False,
                "rerank_enabled": False,
                "compress_enabled": False,
                "verify_enabled": True,
                "temperature": 0.25,
            },
        )
    )

    assert result.answer == "runtime answer"
    assert calls == [
        {
            "model": "custom-rag-model",
            "api_key": "custom-secret",
            "temperature": 0.25,
            "max_tokens": 1000,
            "base_url": "https://llm.example.test/v1",
        }
    ]
    assert len(invoked_prompts) == 2
    assert "Explain quarterly margin trends" in invoked_prompts[0]
