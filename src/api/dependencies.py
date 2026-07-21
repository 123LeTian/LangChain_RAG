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
# 服务依赖（占位实现 — 后续由 B/C/D 同学提供真实服务后替换）
# ============================================================================

class KnowledgeService:
    """知识库服务桩 — B 同学实现"""
    async def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        return []

    async def get_knowledge_base(self, kb_id: str) -> dict | None:
        return None

    async def create_knowledge_base(self, owner_id: str, name: str, description: str = "") -> dict:
        return {}


class RAGService:
    """RAG 服务桩 — C 同学实现"""
    async def query(self, request: Any) -> Any:
        return None

    async def query_stream(self, request: Any):
        yield ""


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

# 全局单例（后续换真实实现时替换即可）
_knowledge_service = KnowledgeService()
_rag_service = RAGService()
_graph_repo = GraphRepository()
_trace_repo = TraceRepository()
_eval_runner = EvaluationRunner()


async def get_knowledge_service() -> KnowledgeService:
    """注入 KnowledgeService"""
    return _knowledge_service


async def get_rag_service() -> RAGService:
    """注入 RAGService"""
    return _rag_service


async def get_graph_repo() -> GraphRepository:
    """注入 GraphRepository"""
    return _graph_repo


async def get_trace_repo() -> TraceRepository:
    """注入 TraceRepository"""
    return _trace_repo


async def get_eval_runner() -> EvaluationRunner:
    """注入 EvaluationRunner"""
    return _eval_runner


# ============================================================================
# 类型别名 — 简化路由函数签名
# ============================================================================

KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
GraphRepoDep = Annotated[GraphRepository, Depends(get_graph_repo)]
TraceRepoDep = Annotated[TraceRepository, Depends(get_trace_repo)]
EvalRunnerDep = Annotated[EvaluationRunner, Depends(get_eval_runner)]
