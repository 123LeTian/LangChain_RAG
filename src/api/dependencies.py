"""
API Dependencies — Owner: A
FastAPI 依赖注入层。
所有业务服务（KnowledgeService / RAGService 等）通过此层注入，
路由层不直接实例化服务，便于测试 Mock 和替换实现。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends


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


# ============================================================================
# 类型别名
# ============================================================================

KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
GraphRepoDep = Annotated[GraphRepository, Depends(get_graph_repo)]
TraceRepoDep = Annotated[TraceRepository, Depends(get_trace_repo)]
EvalRunnerDep = Annotated[EvaluationRunner, Depends(get_eval_runner)]