"""
RAG Model — Owner: Shared
共享 RAG 契约：定义了 RAG 管道的统一接口和数据模型。
所有 RAG 策略必须实现此契约。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 核心枚举 — 五种 RAG 模式与执行阶段
# ============================================================================

class RAGMode(str, Enum):
    """RAG 运行模式（与前端模式选择器一一对应）"""
    NAIVE = "naive"          # 基础向量检索流水线
    ADVANCED = "advanced"    # 查询优化 + 重排 + 压缩
    MODULAR = "modular"      # 可配置模块组合
    GRAPH = "graph"          # 知识图谱检索
    AGENTIC = "agentic"      # Agent 自主路由


class TraceStage(str, Enum):
    """Trace 执行阶段（前端用来渲染时间线各个节点）"""
    INTENT = "intent"                # 意图识别
    REWRITE = "rewrite"              # 查询改写
    RETRIEVE = "retrieve"            # 向量检索
    RERANK = "rerank"                # 重排序
    COMPRESS = "compress"            # 上下文压缩
    GRAPH_SEARCH = "graph_search"    # 图谱搜索
    TOOL_CALL = "tool_call"          # Agent 工具调用
    GENERATE = "generate"            # LLM 生成回答
    VERIFY = "verify"                # 答案校验
    COMPLETE = "complete"            # 正常完成
    ERROR = "error"                  # 异常终止


# ============================================================================
# 统一检索结果模型
# ============================================================================

class RetrievalHit(BaseModel):
    """统一召回结果 — 所有检索器（向量/图谱/混合）都返回此结构"""
    chunk_id: str = Field(..., description="稳定 Chunk ID，可追溯到 DocumentRecord")
    text: str = Field(..., description="Chunk 原文片段")
    score: float = Field(..., description="相似度得分（0-1），图谱命中为 1.0）")
    rank: int = Field(..., description="召回排名（1-based）")
    retriever: str = Field(..., description="来源检索器名称，如 vector/graph/hybrid")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据（页面/章节等）")


# ============================================================================
# 引用模型
# ============================================================================

class Citation(BaseModel):
    """最终引用 — 回答中每个事实陈述必须对应一个 Citation"""
    document_id: str = Field(..., description="来源文档 ID")
    chunk_id: str = Field(..., description="来源 Chunk ID")
    filename: str = Field(..., description="文档文件名")
    page: Optional[int] = Field(default=None, description="页码（如有）")
    quote: str = Field(..., description="引用原文片段")
    score: float = Field(default=0.0, description="该引用的相关度")


# ============================================================================
# Trace 追踪模型
# ============================================================================

class TraceEvent(BaseModel):
    """单次执行事件 — 前端 Trace 时间线的每个节点"""
    trace_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="本次查询的全局追踪 ID"
    )
    stage: TraceStage = Field(..., description="执行阶段")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="阶段开始时间"
    )
    duration_ms: float = Field(default=0.0, description="阶段耗时（毫秒）")
    input_summary: str = Field(default="", description="阶段输入摘要（截断后）")
    output_summary: str = Field(default="", description="阶段输出摘要（截断后）")
    metadata: dict[str, Any] = Field(default_factory=dict, description="阶段额外数据（如 tokens、重排前后列表）")


# ============================================================================
# 统一 RAG 请求/结果模型
# ============================================================================

class RAGRequest(BaseModel):
    """统一 RAG 请求 — 五种模式共用同一入口"""
    query: str = Field(..., min_length=1, description="用户原始问题")
    kb_id: str = Field(..., description="目标知识库 ID")
    mode: RAGMode = Field(default=RAGMode.NAIVE, description="RAG 运行模式")
    session_id: Optional[str] = Field(default=None, description="聊天会话 ID（多轮场景）")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="模式特定选项：top_k/rewrite_enabled/rerank_enabled/max_steps 等"
    )


class RAGResult(BaseModel):
    """统一 RAG 结果 — 五种模式共用同一返回格式"""
    answer: str = Field(..., description="最终回答文本")
    citations: list[Citation] = Field(default_factory=list, description="引用来源列表")
    hits: list[RetrievalHit] = Field(default_factory=list, description="检索命中列表（用于调试面板）")
    trace: list[TraceEvent] = Field(default_factory=list, description="完整执行轨迹")
    usage: dict[str, Any] = Field(default_factory=dict, description="Token 用量与耗时统计")
    warnings: list[str] = Field(default_factory=list, description="执行过程中的警告信息")
    mode: RAGMode = Field(..., description="实际执行的 RAG 模式")
