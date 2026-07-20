"""Utilities for splitting loaded documents into retrieval-sized chunks.

The loaders in this project return :class:`DocumentRecord` instances, while
downstream LangChain integrations generally use ``langchain_core.Document``.
This module accepts both forms and always returns new LangChain ``Document``
objects so that the original loaded records are never mutated.
"""

from hashlib import sha256
from typing import Callable, Iterable, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import DocumentRecord


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
<<<<<<< HEAD
DEFAULT_SEPARATORS = ["\n\n", "\n", "\u3002", "\uff0c", "\uff1f", "\uff01", "\uff1b", " ", ""]
=======
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
>>>>>>> 9809988b04f1e98081fdcb0a83d52a85d9021022

DocumentInput = Union[DocumentRecord, Document]
DocumentInputs = Union[DocumentInput, Iterable[DocumentInput]]


def _validate_options(
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    length_function: Callable[[str], int],
) -> None:
    """Validate public splitter options before constructing LangChain's splitter."""
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
    ):
        raise ValueError("chunk_overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not callable(length_function):
        raise TypeError("length_function must be callable")
    if (
        isinstance(separators, str)
        or not separators
        or not all(isinstance(separator, str) for separator in separators)
    ):
        raise ValueError("separators must be a non-empty sequence of strings")


def _resolve_separators(separators: Optional[Sequence[str]]) -> List[str]:
    """Return a detached separator list and reject a common string misuse."""
    if isinstance(separators, str):
        raise ValueError("separators must be a non-empty sequence of strings")
    return list(DEFAULT_SEPARATORS if separators is None else separators)


def _document_content_and_metadata(document: DocumentInput) -> tuple[str, dict]:
    """Convert supported inputs to content and a detached metadata dictionary."""
    if isinstance(document, DocumentRecord):
        metadata = dict(document.metadata)
        # Keep the loader's top-level source information available to chunks.
        metadata.setdefault("document_id", document.id)
        metadata.setdefault("kb_id", document.kb_id)
        metadata.setdefault("source", document.filename)
        metadata.setdefault("file_name", document.filename)
        metadata.setdefault("file_type", document.file_type)
        metadata.setdefault("checksum", document.checksum)
        return document.text, metadata
    if isinstance(document, Document):
        return document.page_content, dict(document.metadata)
    raise TypeError(
        "documents must contain DocumentRecord or langchain Document objects"
    )


def _stable_document_key(content: str, metadata: dict) -> str:
    """Prefer an existing document ID and otherwise derive a deterministic key."""
    document_key = None
    for field in ("document_id", "id", "source"):
        value = metadata.get(field)
        if value is not None and str(value):
            document_key = str(value)
            break
    if document_key is None:
        document_key = sha256(content.encode("utf-8")).hexdigest()

    page = metadata.get("page")
    if page is not None:
        return f"{document_key}:page-{page}"
    return document_key


def split_document(
    document: DocumentInput,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[Sequence[str]] = None,
    length_function: Callable[[str], int] = len,
) -> List[Document]:
    """Split one loaded document into new LangChain ``Document`` chunks.

    ``length_function`` is applied by LangChain to each candidate text and
    defaults to Python's ``len`` (character count). Empty and whitespace-only
    documents produce no chunks.
    """
    chosen_separators = _resolve_separators(separators)
    _validate_options(chunk_size, chunk_overlap, chosen_separators, length_function)
    content, metadata = _document_content_and_metadata(document)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=chosen_separators,
        length_function=length_function,
        keep_separator="end",
    )
    chunks = splitter.split_text(content)
    document_key = _stable_document_key(content, metadata)
    total_chunks = len(chunks)

    result: List[Document] = []
    for index, chunk in enumerate(chunks):
        chunk_metadata = dict(metadata)
        # setdefault preserves fields supplied by a loader or caller.
        chunk_metadata.setdefault("chunk_index", index)
        chunk_metadata.setdefault("total_chunks", total_chunks)
        chunk_metadata.setdefault("chunk_id", f"{document_key}:chunk-{index}")
        result.append(Document(page_content=chunk, metadata=chunk_metadata))
    return result


def split_documents(
    documents: DocumentInputs,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[Sequence[str]] = None,
    length_function: Callable[[str], int] = len,
) -> List[Document]:
    """Split one document or an iterable of documents while preserving order."""
    if isinstance(documents, (DocumentRecord, Document)):
        document_list = [documents]
    else:
        try:
            document_list = list(documents)
        except TypeError as exc:
            raise TypeError(
                "documents must be a DocumentRecord, Document, or iterable of either"
            ) from exc

    chosen_separators = _resolve_separators(separators)
    _validate_options(chunk_size, chunk_overlap, chosen_separators, length_function)

    result: List[Document] = []
    for document in document_list:
        result.extend(
            split_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=chosen_separators,
                length_function=length_function,
            )
        )
    return result


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_SEPARATORS",
    "split_document",
    "split_documents",
]
