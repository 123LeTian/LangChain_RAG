"""Helpers for API-managed knowledge-base document files.

The API persists lightweight document records in ``data/knowledge_bases.json``.
Those records should never force callers to reconstruct file paths by hand:
future uploads use a stable per-KB ``storage_path`` while legacy records keep
working through their old flat ``filename`` fallback.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def documents_root(project_root: Path | None = None) -> Path:
    """Return the root directory that stores uploaded document files."""
    return (project_root or _DEFAULT_PROJECT_ROOT) / "documents"


def safe_filename(filename: str) -> str:
    """Return a display-safe filename without path traversal components."""
    safe = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", filename)
    safe = safe.replace("..", "_")
    return safe or "upload"


def safe_path_segment(value: str, fallback: str) -> str:
    """Return a single safe path segment for KB/document identifiers."""
    safe = re.sub(r"[^\w.\-]", "_", value.strip())
    safe = safe.replace("..", "_").strip("._")
    return safe or fallback


def file_checksum(content: bytes) -> str:
    """Return a stable checksum for uploaded file bytes."""
    return hashlib.sha256(content).hexdigest()


def document_storage_relative_path(kb_id: str, doc_id: str, filename: str) -> str:
    """Build the canonical storage path for a newly uploaded document."""
    kb_segment = safe_path_segment(kb_id, "kb")
    doc_segment = safe_path_segment(doc_id, "doc")
    display_name = safe_filename(filename)
    return Path(kb_segment, f"{doc_segment}__{display_name}").as_posix()


def resolve_document_path(
    doc: dict[str, Any],
    *,
    project_root: Path | None = None,
    docs_dir: Path | None = None,
) -> Path:
    """Resolve the filesystem path for a persisted document record.

    ``storage_path`` is authoritative for new records. Legacy records that only
    have ``filename`` still resolve to ``documents/<filename>``.
    """
    root = docs_dir or documents_root(project_root)
    storage_path = str(doc.get("storage_path") or "").strip()
    if storage_path:
        resolved = _resolve_relative_path(root, storage_path)
        if resolved is not None:
            return resolved

    filename = safe_filename(str(doc.get("filename") or ""))
    return root / filename


def document_display_filename(doc: dict[str, Any], path: Path | None = None) -> str:
    """Return the filename that should be shown in citations and API payloads."""
    filename = str(doc.get("filename") or "").strip()
    if filename:
        return filename
    if path is not None:
        return path.name
    return "unknown"


def relative_storage_path(path: Path, *, docs_dir: Path) -> str:
    """Return a normalized relative path under the documents root."""
    try:
        return path.resolve().relative_to(docs_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def apply_document_metadata(
    loaded_doc: Any,
    record: dict[str, Any],
    path: Path,
    *,
    docs_dir: Path,
) -> Any:
    """Annotate a loaded ingestion document with stable source metadata."""
    display_name = document_display_filename(record, path)
    storage_path = str(record.get("storage_path") or "").strip()
    if not storage_path:
        storage_path = relative_storage_path(path, docs_dir=docs_dir)

    if hasattr(loaded_doc, "filename"):
        loaded_doc.filename = display_name

    metadata = dict(getattr(loaded_doc, "metadata", {}) or {})
    metadata.update(
        {
            "filename": display_name,
            "file_name": display_name,
            "source": display_name,
            "storage_path": storage_path,
        }
    )
    if hasattr(loaded_doc, "metadata"):
        loaded_doc.metadata = metadata
    return loaded_doc


def iter_supported_document_files(docs_dir: Path) -> list[Path]:
    """Return all supported document files under ``docs_dir``."""
    files: list[Path] = []
    for suffix in sorted(SUPPORTED_DOCUMENT_SUFFIXES):
        files.extend(docs_dir.glob(f"**/*{suffix}"))
    return sorted(files)


def _resolve_relative_path(root: Path, value: str) -> Path | None:
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        return None

    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate
