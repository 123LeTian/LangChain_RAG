"""Deterministic, metadata-preserving text normalization."""

from copy import deepcopy
from dataclasses import replace
import re
import unicodedata
from typing import Union

from langchain_core.documents import Document

from .exceptions import EmptyContentError
from .models import DocumentRecord


NormalizedDocument = Union[DocumentRecord, Document]


class TextNormalizer:
    """Normalize loaded text without mutating the caller's document."""

    _INVISIBLE_CHARACTERS = dict.fromkeys(
        # U+200D (ZWJ) is intentionally preserved because it can compose emoji.
        map(ord, ("\u200b", "\u200c", "\u2060", "\ufeff")),
        None,
    )
    _HORIZONTAL_SPACE_RE = re.compile(r"[ \t\v\f]+")
    _EXCESS_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")

    def normalize_text(self, text: str) -> str:
        """Return NFC-normalized text or raise when no usable content remains."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.translate(self._INVISIBLE_CHARACTERS)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        lines = []
        for line in normalized.split("\n"):
            # Keep Markdown line boundaries while removing horizontal noise.
            lines.append(self._HORIZONTAL_SPACE_RE.sub(" ", line).strip())
        normalized = "\n".join(lines)
        normalized = self._EXCESS_BLANK_LINES_RE.sub("\n\n", normalized)
        normalized = normalized.strip()

        if not normalized:
            raise EmptyContentError("normalized document")
        return normalized

    def normalize_document(self, document: NormalizedDocument) -> NormalizedDocument:
        """Return a new document with normalized text and detached metadata."""

        if isinstance(document, DocumentRecord):
            return replace(
                document,
                text=self.normalize_text(document.text),
                metadata=deepcopy(document.metadata),
            )
        if isinstance(document, Document):
            return Document(
                page_content=self.normalize_text(document.page_content),
                metadata=deepcopy(document.metadata),
            )
        raise TypeError(
            "document must be a DocumentRecord or langchain_core Document"
        )


__all__ = ["TextNormalizer"]
