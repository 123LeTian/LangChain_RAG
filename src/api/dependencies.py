"""
API Dependencies — Owner: A
FastAPI 依赖注入层。
所有业务服务（KnowledgeService / RAGService 等）通过此层注入，
路由层不直接实例化服务，便于测试 Mock 和替换实现。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Depends

from src.chat.chat_application_service import ChatApplicationService
from src.chat.message_service import MessageService
from src.chat.memory_service import MemoryService
from src.chat.model_registry import ModelRegistry
from src.chat.preset_service import PresetService
from src.chat.rag_gateway import RAGGateway
from src.chat.export_service import ExportService
from src.chat.search_service import SearchService
from src.chat.session_service import SessionService
from src.chat.token_service import TokenService
from src.chat.retry_policy import RetryPolicy
from src.chat.structured_logger import StructuredLogger
from src.chat_storage.sqlite_chat_store import SQLiteChatStore
from src.config.runtime_config import get_runtime_config
from src.evaluation.multi_mode_runner import (
    MultiModeEvaluationRunner,
    build_chunk_evaluation_samples,
    create_runner,
)
from src.evaluation.runner import DEFAULT_DATASET_PATH


# ============================================================================
# 知识库服务（桩）
# ============================================================================

class KnowledgeService:
    """知识库服务桩 — B 同学实现"""
    async def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        return []

    async def get_knowledge_base(self, kb_id: str) -> dict | None:
        return None

    async def create_knowledge_base(self, owner_id: str, name: str, description: str = "") -> dict:
        return {}


# ============================================================================
# RAG 服务 — 已接入真实 DeepSeek + 向量检索
# ============================================================================

class RAGService:
    """RAG 服务 — 委托给 RealRAGService (DeepSeek + HuggingFace 向量检索)"""
    def __init__(self):
        from src.api.real_rag_service import RealRAGService
        self._real = RealRAGService()

    async def query(self, request: Any) -> Any:
        return await self._real.query(request)

    async def query_stream(self, request: Any):
        async for chunk in self._real.query_stream(request):
            yield chunk

    async def evaluation_chunks(self) -> list[Any]:
        ensure_init = getattr(self._real, "_ensure_init", None)
        if callable(ensure_init):
            await ensure_init()
        return list(getattr(self._real, "chunks", []) or [])


# ============================================================================
# 其他服务（桩）
# ============================================================================

class GraphRepository:
    """图谱仓库桩 — D 同学实现"""
    async def get_graph_data(self, kb_id: str) -> dict:
        return {}


class TraceRepository:
    """Trace 仓库桩 — C 同学实现"""
    async def get_trace(self, trace_id: str) -> list:
        return []


class EvaluationRunner:
    """评测执行器桩 — D 同学实现"""
    async def run_evaluation(self, mode: str, kb_id: str) -> dict:
        return {}

    async def get_result(self, run_id: str) -> dict:
        return {}


# ============================================================================
# 服务单例与依赖注入函数
# ============================================================================

_knowledge_service = KnowledgeService()
_rag_service = RAGService()
_graph_repo = GraphRepository()
_trace_repo = TraceRepository()
_eval_runner = EvaluationRunner()
_runtime_config = get_runtime_config()
_chat_store = SQLiteChatStore(_runtime_config.chat_db_path)
_chat_session_service = SessionService(_chat_store)
_chat_message_service = MessageService(_chat_store)
_chat_memory_service = MemoryService(_chat_message_service)
_model_registry = ModelRegistry(_runtime_config.model_config_path)
_preset_service = PresetService(_chat_store)
_chat_search_service = SearchService(_chat_store)
_chat_token_service = TokenService(_chat_store)
_chat_export_service = ExportService(
    _chat_session_service,
    _chat_message_service,
    _chat_token_service,
)
_structured_logger = StructuredLogger()
_retry_policy = RetryPolicy(
    timeout_seconds=_runtime_config.retry_timeout_seconds,
    max_retries=_runtime_config.retry_max_retries,
    backoff_seconds=_runtime_config.retry_backoff_seconds,
)
_chat_rag_gateway = RAGGateway(_rag_service, retry_policy=_retry_policy, logger=_structured_logger)
_chat_application_service = ChatApplicationService(
    _chat_session_service,
    _chat_message_service,
    _chat_memory_service,
    _chat_rag_gateway,
    _model_registry,
    _preset_service,
    _structured_logger,
)


async def get_knowledge_service() -> KnowledgeService:
    return _knowledge_service


async def get_rag_service() -> RAGService:
    return _rag_service


async def get_graph_repo() -> GraphRepository:
    return _graph_repo


async def get_trace_repo() -> TraceRepository:
    return _trace_repo


async def get_eval_runner() -> EvaluationRunner:
    return _eval_runner


async def get_chat_session_service() -> SessionService:
    return _chat_session_service


async def get_chat_message_service() -> MessageService:
    return _chat_message_service


async def get_chat_application_service() -> ChatApplicationService:
    return _chat_application_service


async def get_model_registry() -> ModelRegistry:
    return _model_registry


async def get_preset_service() -> PresetService:
    return _preset_service


async def get_chat_search_service() -> SearchService:
    return _chat_search_service


async def get_chat_token_service() -> TokenService:
    return _chat_token_service


async def get_chat_export_service() -> ExportService:
    return _chat_export_service


# ============================================================================
# 类型别名
# ============================================================================

KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
GraphRepoDep = Annotated[GraphRepository, Depends(get_graph_repo)]
TraceRepoDep = Annotated[TraceRepository, Depends(get_trace_repo)]
EvalRunnerDep = Annotated[EvaluationRunner, Depends(get_eval_runner)]
ChatSessionServiceDep = Annotated[SessionService, Depends(get_chat_session_service)]
ChatMessageServiceDep = Annotated[MessageService, Depends(get_chat_message_service)]
ChatApplicationServiceDep = Annotated[
    ChatApplicationService,
    Depends(get_chat_application_service),
]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
PresetServiceDep = Annotated[PresetService, Depends(get_preset_service)]
ChatSearchServiceDep = Annotated[SearchService, Depends(get_chat_search_service)]
ChatTokenServiceDep = Annotated[TokenService, Depends(get_chat_token_service)]
ChatExportServiceDep = Annotated[ExportService, Depends(get_chat_export_service)]


class RealtimeEvaluationRunner:
    """Realtime multi-mode evaluation runner."""

    def __init__(
        self,
        rag_service: Any,
        model_registry: ModelRegistry,
        *,
        dataset_path: str = DEFAULT_DATASET_PATH,
    ) -> None:
        self._rag_service = rag_service
        self._model_registry = model_registry
        self._dataset_path = dataset_path
        self._runs: dict[str, dict[str, Any]] = {}

    async def run_evaluation(self, request: Any) -> dict:
        from src.api.routes.evaluation import EvaluationRunRequest

        payload = request if isinstance(request, EvaluationRunRequest) else EvaluationRunRequest(**request)
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        model = self._resolve_model(payload.model_id)
        self._runs[run_id] = {
            "run_id": run_id,
            "status": "running",
            "kb_id": payload.kb_id,
            "modes": list(payload.modes),
            "model_id": model.id,
            "requested_model_id": payload.model_id,
            "sample_limit": payload.sample_limit,
            "message": f"评测已启动，将在 {len(payload.modes)} 种模式下运行",
        }

        chunks = await self._rag_service.evaluation_chunks()
        samples = build_chunk_evaluation_samples(
            chunks,
            sample_limit=payload.sample_limit,
        )
        report = await MultiModeEvaluationRunner(
            dataset_path=f"dynamic:{payload.kb_id}",
            modes=payload.modes,
            sample_limit=payload.sample_limit,
            samples=samples,
            runner_factory=lambda mode: create_runner(
                mode,
                use_mock=False,
                service=self._rag_service,
                kb_id=payload.kb_id,
                top_k=5,
                model_config=model.model_dump(),
            ),
        ).run()
        report["run_id"] = run_id
        report["kb_id"] = payload.kb_id
        report["model_id"] = model.id
        report["requested_model_id"] = payload.model_id
        report["modes"] = list(payload.modes)
        report["sample_limit"] = payload.sample_limit
        self._runs[run_id] = report
        return {
            "run_id": run_id,
            "status": "completed",
            "kb_id": payload.kb_id,
            "modes": list(payload.modes),
            "model_id": model.id,
            "sample_limit": payload.sample_limit,
            "message": f"评测已完成，将在 {len(payload.modes)} 种模式下运行",
        }

    async def get_result(self, run_id: str) -> dict:
        result = self._runs.get(run_id)
        if result is None:
            raise KeyError(run_id)
        return result

    def _resolve_model(self, model_id: str | None):
        if model_id:
            try:
                return self._model_registry.get_enabled(model_id)
            except Exception:
                pass
        return self._model_registry.default_model()


_eval_runner = RealtimeEvaluationRunner(_rag_service, _model_registry)
