"""
Knowledge Routes — Knowledge base CRUD, document upload, index management.
Knowledge bases are persisted to a JSON file so they survive restarts.
Uploaded files are saved under documents/<kb_id>/ and trigger RAG index rebuild.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, File, Form, UploadFile
from pathlib import Path

from src.api.document_store import (
    SUPPORTED_DOCUMENT_SUFFIXES,
    apply_document_metadata,
    document_storage_relative_path,
    documents_root,
    file_checksum,
    resolve_document_path,
    safe_filename as _store_safe_filename,
)
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
# Persistence — save KBs and docs to JSON file
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_FILE = _PROJECT_ROOT / "data" / "knowledge_bases.json"


def _load_data() -> dict:
    """Load persisted KBs and docs from JSON file."""
    if _DATA_FILE.exists():
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"kbs": [], "docs": []}


def _save_data(data: dict) -> None:
    """Save KBs and docs to JSON file."""
    _DATA_FILE.parent.mkdir(exist_ok=True)
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Initialize from persisted data
_persisted = _load_data()
_MOCK_KBS: list[dict] = _persisted.get("kbs", [])
_MOCK_DOCS: list[dict] = _persisted.get("docs", [])
_MOCK_JOBS: dict[str, dict] = {}

# Ensure default KB exists
if not any(kb["id"] == "kb_001" for kb in _MOCK_KBS):
    _MOCK_KBS.insert(0, {
        "id": "kb_001",
        "name": "default",
        "description": "documents/ dir",
        "status": "ready",
        "owner_id": "user_001",
        "doc_count": 0,
        "chunk_count": 0,
    })
    _save_data({"kbs": _MOCK_KBS, "docs": _MOCK_DOCS})


def _persist() -> None:
    """Save current state to disk."""
    _save_data({"kbs": _MOCK_KBS, "docs": _MOCK_DOCS})


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


def _safe_filename(filename: str) -> str:
    return _store_safe_filename(filename) or f"upload_{uuid.uuid4().hex[:8]}"


def _documents_root() -> Path:
    return documents_root(_PROJECT_ROOT)


def _document_path(doc: dict) -> Path:
    return resolve_document_path(doc, project_root=_PROJECT_ROOT)


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def _storage_path_referenced(path: Path) -> bool:
    key = _path_key(path)
    for doc in _MOCK_DOCS:
        try:
            if _path_key(_document_path(doc)) == key:
                return True
        except Exception:
            continue
    return False


def _delete_file_if_unreferenced(doc: dict) -> None:
    path = _document_path(doc)
    if not path.exists() or _storage_path_referenced(path):
        return
    try:
        path.unlink()
        print(f"[KB] Deleted file: {path}", flush=True)
        _remove_empty_parent_dirs(path)
    except Exception as exc:
        print(f"[KB] Delete file skipped: {path}: {exc}", flush=True)


def _remove_empty_parent_dirs(path: Path) -> None:
    root = _documents_root().resolve()
    parent = path.parent.resolve()
    while parent != root and root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _trigger_rag_rebuild() -> None:
    try:
        from src.api.dependencies import _rag_service
        if hasattr(_rag_service, '_real'):
            _rag_service._real._initialized = False
            print("[KB] RAG index will rebuild on next query", flush=True)
    except Exception as e:
        print(f"[KB] RAG rebuild trigger failed: {e}", flush=True)


def _find_document(kb_id: str, doc_id: str) -> dict | None:
    for doc in _MOCK_DOCS:
        if doc["kb_id"] == kb_id and doc["id"] == doc_id:
            return doc
    return None


def _document_chunk_count(doc: dict) -> int:
    fpath = _document_path(doc)
    if not fpath.exists():
        return 0
    try:
        from src.ingestion import LoaderFactory
        from src.ingestion.splitter import split_documents

        loaded = LoaderFactory.load(
            fpath,
            kb_id=str(doc.get("kb_id") or ""),
            document_id=str(doc.get("id") or fpath.stem),
        )
        loaded = apply_document_metadata(loaded, doc, fpath, docs_dir=_documents_root())
        return len(split_documents(loaded, chunk_size=1500, chunk_overlap=300))
    except Exception:
        return int(doc.get("chunk_count") or 0)


def _decorate_document(doc: dict) -> dict:
    payload = dict(doc)
    payload["chunk_count"] = _document_chunk_count(doc)
    return payload


def _decorate_kb(kb: dict) -> dict:
    payload = dict(kb)
    docs = [doc for doc in _MOCK_DOCS if doc.get("kb_id") == kb.get("id")]
    payload["doc_count"] = len(docs)
    payload["chunk_count"] = sum(_document_chunk_count(doc) for doc in docs)
    return payload


def _refresh_kb_counts(kb_id: str) -> None:
    for doc in _MOCK_DOCS:
        if doc.get("kb_id") == kb_id:
            doc["chunk_count"] = _document_chunk_count(doc)
    for kb in _MOCK_KBS:
        if kb.get("id") == kb_id:
            docs = [doc for doc in _MOCK_DOCS if doc.get("kb_id") == kb_id]
            kb["doc_count"] = len(docs)
            kb["chunk_count"] = sum(int(doc.get("chunk_count") or 0) for doc in docs)
            break


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
    kb_id = f"kb_{uuid.uuid4().hex[:8]}"
    kb = _make_kb(kb_id, name, description, "ready", 0, 0)
    _MOCK_KBS.append(kb)
    _persist()
    print(f"[KB] Created: {kb_id} ({name}), persisted to {_DATA_FILE}", flush=True)
    return kb


@router.get("")
async def list_knowledge_bases(
    owner_id: str = "user_001",
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    return [_decorate_kb(kb) for kb in _MOCK_KBS]


@router.get("/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    for kb in _MOCK_KBS:
        if kb["id"] == kb_id:
            return _decorate_kb(kb)
    raise NotFoundError(f"KB {kb_id} not found")


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    global _MOCK_KBS, _MOCK_DOCS
    if not any(kb["id"] == kb_id for kb in _MOCK_KBS):
        raise NotFoundError(f"KB {kb_id} not found")
    docs_to_delete = [doc for doc in _MOCK_DOCS if doc.get("kb_id") == kb_id]
    _MOCK_KBS = [kb for kb in _MOCK_KBS if kb["id"] != kb_id]
    _MOCK_DOCS = [doc for doc in _MOCK_DOCS if doc.get("kb_id") != kb_id]
    for doc in docs_to_delete:
        _delete_file_if_unreferenced(doc)
    _persist()
    _trigger_rag_rebuild()
    return None


# ============================================================================
# Document upload — saves to documents/ and rebuilds RAG index
# ============================================================================

@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    print(f"[KB] Upload start: kb={kb_id}, file={file.filename}", flush=True)

    if not any(kb["id"] == kb_id for kb in _MOCK_KBS):
        raise NotFoundError(f"KB {kb_id} not found")

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if f".{ext}" not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValidationError(f"Unsupported format: .{ext}")

    content = await file.read()
    print(f"[KB] Read {len(content)} bytes from upload", flush=True)

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    safe_name = _safe_filename(file.filename or f"upload.{ext}")
    docs_dir = _documents_root()
    docs_dir.mkdir(parents=True, exist_ok=True)
    storage_path = document_storage_relative_path(kb_id, doc_id, safe_name)
    save_path = docs_dir / storage_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "wb") as f:
            f.write(content)
        print(f"[KB] Saved to: {save_path}", flush=True)
    except Exception as e:
        print(f"[KB] Save FAILED: {e}", flush=True)
        raise ValidationError(f"Failed to save file: {e}")

    if not save_path.exists():
        raise ValidationError("File save verification failed")

    doc = {
        "id": doc_id,
        "kb_id": kb_id,
        "filename": safe_name,
        "storage_path": storage_path,
        "type": ext,
        "checksum": file_checksum(content),
        "status": "indexed",
        "chunk_count": 0,
        "size_bytes": len(content),
    }
    doc["chunk_count"] = _document_chunk_count(doc)
    _MOCK_DOCS.append(doc)

    _refresh_kb_counts(kb_id)

    _persist()

    # Trigger RAG index rebuild
    _trigger_rag_rebuild()

    print(f"[KB] Upload complete: {safe_name} ({len(content)} bytes)", flush=True)
    return doc


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> list[dict]:
    return [_decorate_document(d) for d in _MOCK_DOCS if d["kb_id"] == kb_id]


@router.get("/{kb_id}/documents/{doc_id}/preview")
async def preview_document(
    kb_id: str,
    doc_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    doc = _find_document(kb_id, doc_id)
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    fpath = _document_path(doc)
    if not fpath.exists():
        raise NotFoundError(f"Document file {doc['filename']} not found")

    try:
        from src.ingestion import LoaderFactory

        loaded = LoaderFactory.load(fpath, kb_id=kb_id, document_id=doc_id)
        loaded = apply_document_metadata(loaded, doc, fpath, docs_dir=_documents_root())
        text = getattr(loaded, "text", "") or ""
        metadata = getattr(loaded, "metadata", {}) or {}
    except Exception as exc:
        raise ValidationError(f"Document preview failed: {type(exc).__name__}: {exc}")

    max_chars = 20000
    preview_text = text[:max_chars]
    return {
        "id": doc_id,
        "kb_id": kb_id,
        "filename": doc["filename"],
        "type": doc.get("type") or fpath.suffix.lstrip("."),
        "size_bytes": doc.get("size_bytes", fpath.stat().st_size),
        "text": preview_text,
        "truncated": len(text) > max_chars,
        "metadata": metadata,
    }


@router.delete("/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
):
    global _MOCK_DOCS
    doc = _find_document(kb_id, doc_id)
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")
    _MOCK_DOCS = [d for d in _MOCK_DOCS if not (d["id"] == doc_id and d["kb_id"] == kb_id)]
    _delete_file_if_unreferenced(doc)
    _refresh_kb_counts(kb_id)
    _persist()
    _trigger_rag_rebuild()
    return None


# ============================================================================
# Index management
# ============================================================================

@router.post("/{kb_id}/index", status_code=202)
async def create_index(
    kb_id: str,
    service: KnowledgeServiceDep = None,  # type: ignore
) -> dict:
    _refresh_kb_counts(kb_id)
    _persist()
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _MOCK_JOBS[job_id] = {
        "job_id": job_id,
        "kb_id": kb_id,
        "status": "running",
        "progress": 0,
        "started_at": time.time(),
    }

    try:
        _trigger_rag_rebuild()
    except Exception:
        pass

    return {"job_id": job_id, "status": "running", "message": "Index rebuild started"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    job = _MOCK_JOBS.get(job_id)
    if not job:
        raise NotFoundError(f"Job {job_id} not found")
    elapsed = time.time() - job["started_at"]
    progress = min(100, int(elapsed * 10))
    job["progress"] = progress
    if progress >= 100:
        job["status"] = "completed"
    return job
