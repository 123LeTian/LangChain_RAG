"""
Evaluation Model — Owner: Shared
评测数据模型与指标定义。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from .rag import RAGMode


# ============================================================================
# 评测样本
# ============================================================================

class EvaluationSample(BaseModel):
    """固定评测样本 — 评测集的每条数据"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="样本唯一 ID")
    question: str = Field(..., description="测试问题")
    expected_answer: str = Field(default="", description="期望答案要点（用于答案质量对比）")
    expected_sources: list[str] = Field(default_factory=list, description="期望来源文档 ID 列表")
    tags: list[str] = Field(
        default_factory=list,
        description="问题标签：fact/multi_hop/relation/global/ambiguous/unanswerable/agent_route"
    )
    kb_id: str = Field(default="default", description="关联知识库 ID（评测可按知识库分组）")


# ============================================================================
# 评测指标
# ============================================================================

class RetrievalMetrics(BaseModel):
    """检索维度指标"""
    hit_at_1: float = Field(default=0.0, description="Hit@1")
    hit_at_3: float = Field(default=0.0, description="Hit@3")
    hit_at_5: float = Field(default=0.0, description="Hit@5")
    hit_at_10: float = Field(default=0.0, description="Hit@10")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_10: Optional[float] = Field(default=None, description="nDCG@10（可选）")


class AnswerMetrics(BaseModel):
    """生成答案维度指标"""
    coverage: float = Field(default=0.0, description="答案要点覆盖率")
    faithfulness: float = Field(default=0.0, description="忠实度（答案是否被上下文支持）")
    relevance: float = Field(default=0.0, description="答案与问题的相关度")


class CitationMetrics(BaseModel):
    """引用维度指标"""
    precision: float = Field(default=0.0, description="引用精确率")
    recall: float = Field(default=0.0, description="引用召回率")


class SystemMetrics(BaseModel):
    """系统性能维度指标"""
    first_token_latency_ms: float = Field(default=0.0, description="首 Token 延迟（毫秒）")
    total_latency_ms: float = Field(default=0.0, description="总耗时（毫秒）")
    total_tokens: int = Field(default=0, description="总 Token 消耗")
    estimated_cost_usd: float = Field(default=0.0, description="估算费用（美元）")


class GraphMetrics(BaseModel):
    """图谱命中维度指标"""
    relation_hit_rate: float = Field(default=0.0, description="关系查询命中率")
    local_search_success: float = Field(default=0.0, description="Local Search 成功率")
    global_search_success: float = Field(default=0.0, description="Global Search 成功率")


class AgentMetrics(BaseModel):
    """Agent 工具选择维度指标"""
    tool_selection_accuracy: float = Field(default=0.0, description="工具选择准确率")
    trajectory_success_rate: float = Field(default=0.0, description="轨迹成功率")
    avg_steps: float = Field(default=0.0, description="平均执行步数")


class MetricsSnapshot(BaseModel):
    """一次评测的完整指标快照"""
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = Field(default_factory=AnswerMetrics)
    citation: CitationMetrics = Field(default_factory=CitationMetrics)
    system: SystemMetrics = Field(default_factory=SystemMetrics)
    graph: Optional[GraphMetrics] = Field(default=None, description="GraphRAG 模式特有指标")
    agent: Optional[AgentMetrics] = Field(default=None, description="Agentic 模式特有指标")


# ============================================================================
# 评测结果
# ============================================================================

class EvaluationResult(BaseModel):
    """单次评测结果 — 一次模式运行在评测集上的汇总"""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="评测运行 ID")
    mode: RAGMode = Field(..., description="评测的 RAG 模式")
    metrics: MetricsSnapshot = Field(..., description="指标快照")
    sample_count: int = Field(default=0, description="评测样本数")
    latency_ms: float = Field(default=0.0, description="总耗时（毫秒）")
    token_usage: dict[str, int] = Field(default_factory=dict, description="Token 用量统计")
    details: list[dict[str, Any]] = Field(default_factory=list, description="逐样本详细结果")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
