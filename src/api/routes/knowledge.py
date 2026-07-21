"""
Knowledge Routes — Owner: A
REST 端点：知识库 CRUD、文档上传、索引管理。
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from src.api.dependencies import KnowledgeServiceDep
from src.api.errors import NotFoundError, ValidationError
from src.models.knowledge import (
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    KnowledgeBase,
    KnowledgeBaseStatus,
)

router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge"])


# ============================================================================
# Mock 数据
# ============================================================================

_MOCK_JOBS: dict[str, dict] = {}  # job_id → job 状态


def _mock_kb(kb_id: str = "kb_demo", name: str = "示例知识库") -> dict:
    """构造 Mock KnowledgeBase"""
    return KnowledgeBase(
        id=kb_id,
        owner_id="user_001",
        name=name,
        description="用于开发和演示的示例知识库",
        status=KnowledgeBaseStatus.READY,
        doc_count=3,
        chunk_count=156,
    ).model_dump()


_MOCK_KBS = [
    _mock_kb("kb_001", "产品技术手册"),
    _mock_kb("kb_002", "公司内部制度"),
    _mock_kb("kb_003", "行业白皮书合集"),
]

_MOCK_DOCS = [
    DocumentRecord(id="doc_0001", kb_id="kb_001", filename="产品手册.pdf", type=DocumentType.PDF, checksum="a1b2c3d4", status=DocumentStatus.INDEXED, chunk_count=42, size_bytes=2048000).model_dump(),
    DocumentRecord(id="doc_0002", kb_id="kb_001", filename="技术规格.docx", type=DocumentType.DOCX, checksum="e5f6g7h8", status=DocumentStatus.INDEXED, chunk_count=68, size_bytes=1536000).model_dump(),
    DocumentRecord(id="doc_0003", kb_id="kb_001", filename="FAQ.md", type=DocumentType.MD, checksum="i9j0k1l2", status=DocumentStatus.INDEXED, chunk_count=46, size_bytes=512000).model_dump(),
]


# ============================================================================
# 知识库 CRUD
# ============================================================================

@router.post("", status_code=201)
async def create_knowledge_base(
    name: str = Form(..., description="知识库名称"),
    description: str = Form("", description="知识库描述（可选）"),
    owner_id: str = Form("user_001", description="所有者 ID"),
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """创建知识库"""
    kb = _mock_kb(kb_id=f"kb_{uuid.uuid4().hex[:8]}", name=name)
    kb["description"] = description
    kb["status"] = KnowledgeBaseStatus.CREATING.value
    kb["doc_count"] = 0
    kb["chunk_count"] = 0
    return kb


@router.get("")
async def list_knowledge_bases(
    owner_id: str = "user_001",
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    """获取知识库列表"""
    return _MOCK_KBS


@router.get("/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """获取单个知识库详情"""
    for kb in _MOCK_KBS:
        if kb["id"] == kb_id:
            return kb
    raise NotFoundError(f"知识库 {kb_id} 不存在")


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    """删除知识库"""
    return None


# ============================================================================
# 文档管理
# ============================================================================

@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(..., description="上传文档（支持 pdf/docx/txt/md）"),
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """上传文档到知识库"""
    # ===== 文件类型检查 =====
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    allowed = {"pdf", "docx", "txt", "md"}
    if ext not in allowed:
        raise ValidationError(f"不支持的文档格式「.{ext}」，仅支持：{', '.join(allowed)}")

    doc = DocumentRecord(
        kb_id=kb_id,
        filename=file.filename or "unknown",
        type=DocumentType(ext),
        checksum=f"mock_{uuid.uuid4().hex[:8]}",
        status=DocumentStatus.UPLOADED,
        size_bytes=0,
    )
    return doc.model_dump()


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    """获取知识库内的文档列表"""
    return [d for d in _MOCK_DOCS if d["kb_id"] == kb_id]


@router.delete("/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    """从知识库删除文档"""
    return None


# ============================================================================
# 索引管理
# ============================================================================

@router.post("/{kb_id}/index", status_code=202)
async def create_index(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """创建/重建向量索引，返回异步任务 ID"""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _MOCK_JOBS[job_id] = {
        "job_id": job_id,
        "kb_id": kb_id,
        "status": "running",
        "progress": 0,
        "started_at": time.time(),
    }
    return {"job_id": job_id, "status": "running", "message": "索引构建已启动"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """查询索引/解析任务的进度"""
    job = _MOCK_JOBS.get(job_id)
    if not job:
        raise NotFoundError(f"任务 {job_id} 不存在")

    # Mock 进度递增
    elapsed = time.time() - job["started_at"]
    progress = min(100, int(elapsed * 10))
    job["progress"] = progress
    if progress >= 100:
        job["status"] = "completed"
    return job
