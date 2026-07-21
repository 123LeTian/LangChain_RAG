"""
Knowledge Routes — Knowledge base CRUD, document upload, index management.
Uploaded files are saved to documents/ and trigger RAG index rebuild.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from src.api.dependencies import KnowledgeServiceDep
from src.api.errors import NotFoundError, ValidationError
from src.api.api_models import (
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    KnowledgeBase,
    KnowledgeBaseStatus,
)

router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge"])

# ============================================================================
# In-memory storage
# ============================================================================

_MOCK_JOBS: dict[str, dict] = {}
_MOCK_DOCS: list[dict] = []


def _make_kb(kb_id: str, name: str, description: str = "", status: str = "ready", doc_count: int = 0, chunk_count: int = 0) -> dict:
    return {
        "id": kb_id,
        "name": name,
        "description": description,
        "status": status,
        "owner_id": "user_001",
        "doc_count": doc_count,
        "chunk_count": chunk_count,
    }


_MOCK_KBS: list[dict] = [
    _make_kb("kb_001", "默认知识库", "documents/ 目录下的文档", "ready", 0, 0),
]


# ============================================================================
# Knowledge base CRUD
# ============================================================================

@router.post("", status_code=201)
async def create_knowledge_base(
    name: str = Form(...),
    description: str = Form(""),
    owner_id: str = Form("user_001"),
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """创建知识库"""
    kb_id = f"kb_{uuid.uuid4().hex[:8]}"
    kb = _make_kb(kb_id, name, description, "ready", 0, 0)
    _MOCK_KBS.append(kb)
    return kb


@router.get("")
async def list_knowledge_bases(
    owner_id: str = "user_001",
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    return list(_MOCK_KBS)


@router.get("/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    for kb in _MOCK_KBS:
        if kb["id"] == kb_id:
            return kb
    raise NotFoundError(f"知识库 {kb_id} 不存在")


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    global _MOCK_KBS
    _MOCK_KBS = [kb for kb in _MOCK_KBS if kb["id"] != kb_id]
    return None


# ============================================================================
# Document management — saves file to documents/ and rebuilds RAG index
# ============================================================================

@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    """上传文档 — 保存到 documents/ 并触发 RAG 索引重建"""
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    allowed = {"pdf", "docx", "txt", "md"}
    if ext not in allowed:
        raise ValidationError(f"不支持的文档格式: .{ext}")

    # Read and save file
    content = await file.read()
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "documents"
    docs_dir.mkdir(exist_ok=True)
    save_path = docs_dir / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    # Create doc record
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    doc = {
        "id": doc_id,
        "kb_id": kb_id,
        "filename": file.filename,
        "type": ext,
        "checksum": f"mock_{uuid.uuid4().hex[:8]}",
        "status": "indexed",
        "chunk_count": 0,
        "size_bytes": len(content),
    }
    _MOCK_DOCS.append(doc)

    # Update KB doc count
    for kb in _MOCK_KBS:
        if kb["id"] == kb_id:
            kb["doc_count"] = kb.get("doc_count", 0) + 1
            break

    # Trigger RAG index rebuild (reset the lazy init flag)
    try:
        from src.api.dependencies import _rag_service
        if hasattr(_rag_service, '_real'):
            _rag_service._real._initialized = False
            print(f"[KB] File uploaded: {file.filename}, RAG index will rebuild on next query", flush=True)
    except Exception as e:
        print(f"[KB] RAG rebuild trigger failed: {e}", flush=True)

    return doc


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    return [d for d in _MOCK_DOCS if d["kb_id"] == kb_id]


@router.delete("/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    global _MOCK_DOCS
    _MOCK_DOCS = [d for d in _MOCK_DOCS if not (d["id"] == doc_id and d["kb_id"] == kb_id)]
    return None


# ============================================================================
# Index management
# ============================================================================

@router.post("/{kb_id}/index", status_code=202)
async def create_index(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _MOCK_JOBS[job_id] = {
        "job_id": job_id,
        "kb_id": kb_id,
        "status": "running",
        "progress": 0,
        "started_at": time.time(),
    }

    # Trigger RAG rebuild
    try:
        from src.api.dependencies import _rag_service
        if hasattr(_rag_service, '_real'):
            _rag_service._real._initialized = False
    except Exception:
        pass

    return {"job_id": job_id, "status": "running", "message": "索引构建已启动"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    job = _MOCK_JOBS.get(job_id)
    if not job:
        raise NotFoundError(f"任务 {job_id} 不存在")
    elapsed = time.time() - job["started_at"]
    progress = min(100, int(elapsed * 10))
    job["progress"] = progress
    if progress >= 100:
        job["status"] = "completed"
    return job