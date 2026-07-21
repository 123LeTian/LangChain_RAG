"""
Graph Routes — Owner: A
REST 端点：知识图谱可视化数据（节点、边、社区）。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.api.dependencies import GraphRepoDep
from src.api.errors import NotFoundError

router = APIRouter(prefix="/api/graphs", tags=["Graph"])


# ============================================================================
# Mock 数据 — ECharts 力导向图所需的 nodes + links 格式
# ============================================================================

_MOCK_GRAPH_DATA = {
    "nodes": [
        {"id": "e1", "name": "处理器 A100", "type": "entity", "chunk_id": "chunk_0003", "label": "Processor"},
        {"id": "e2", "name": "3.2 GHz", "type": "entity", "chunk_id": "chunk_0003", "label": "Frequency"},
        {"id": "e3", "name": "散热系统 X-200", "type": "entity", "chunk_id": "chunk_0012", "label": "Cooling"},
        {"id": "e4", "name": "85 °C", "type": "entity", "chunk_id": "chunk_0012", "label": "Temperature"},
        {"id": "e5", "name": "能耗比 V3", "type": "entity", "chunk_id": "chunk_0007", "label": "Metric"},
        {"id": "c1", "name": "社区：芯片性能", "type": "community", "chunk_id": "", "label": "Community"},
    ],
    "links": [
        {"source": "e1", "target": "e2", "relation": "最高频率", "chunk_id": "chunk_0003"},
        {"source": "e1", "target": "e3", "relation": "配备散热", "chunk_id": "chunk_0012"},
        {"source": "e1", "target": "e5", "relation": "能耗指标", "chunk_id": "chunk_0007"},
        {"source": "e3", "target": "e4", "relation": "满载温度", "chunk_id": "chunk_0012"},
    ],
    "communities": [
        {
            "id": "c1",
            "name": "芯片性能",
            "summary": "该社区聚焦于处理器 A100 的性能指标，包括最高频率 3.2GHz 和满载温度控制。",
            "entity_count": 4,
            "relation_count": 4,
        }
    ],
    "hit_path": ["e1", "e2"],  # 当前查询命中的节点路径（前端高亮用）
}


# ============================================================================
# 路由
# ============================================================================

@router.get("/{kb_id}")
async def get_graph(
    kb_id: str,
    graph_repo: GraphRepoDep = None,  # type: ignore
) -> dict:
    """获取指定知识库的图谱数据（节点、边、社区、命中路径）"""
    return _MOCK_GRAPH_DATA
