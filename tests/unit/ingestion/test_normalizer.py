from copy import deepcopy

import pytest

from src.ingestion.exceptions import EmptyContentError
from src.ingestion.normalizer import TextNormalizer
from src.models.knowledge import DocumentRecord


def make_document(text: str, metadata=None) -> DocumentRecord:
    return DocumentRecord(
        id="doc-1",
        kb_id="kb-1",
        filename="guide.md",
        file_type="markdown",
        text=text,
        checksum="abc",
        metadata={} if metadata is None else metadata,
    )


def test_normalizes_newlines_unicode_bom_invisible_chars_and_spacing():
    normalizer = TextNormalizer()
    source = "\ufeffCafe\u0301\r\n\r\nTitle\t  text\r\n\r\n\r\nNext\u200b"

    assert normalizer.normalize_text(source) == "Café\n\nTitle text\n\nNext"


def test_preserves_markdown_headings_paragraphs_chinese_and_symbols():
    source = "  # 标题  \n\n  普通   段落：RAG © 2026 👩\u200d💻  "

    assert TextNormalizer().normalize_text(source) == (
        "# 标题\n\n普通 段落：RAG © 2026 👩\u200d💻"
    )


@pytest.mark.parametrize("value", ["", " \t\r\n\u200b\ufeff", "\r\n\r\n"])
def test_empty_normalized_text_raises_structured_error(value):
    with pytest.raises(EmptyContentError, match="contains no usable text"):
        TextNormalizer().normalize_text(value)


def test_non_string_input_has_clear_error():
    with pytest.raises(TypeError, match="text must be a string"):
        TextNormalizer().normalize_text(None)  # type: ignore[arg-type]


def test_normalize_document_preserves_metadata_and_does_not_mutate_input():
    metadata = {
        "filename": "guide.md",
        "page": 2,
        "section": "Setup",
        "source": "guide.md",
        "nested": {"value": 1},
    }
    original_metadata = deepcopy(metadata)
    source = make_document("# Setup\r\n\r\nText\t here", metadata)

    normalized = TextNormalizer().normalize_document(source)
    normalized.metadata["nested"]["value"] = 99

    assert normalized is not source
    assert normalized.filename == source.filename
    assert normalized.text == "# Setup\n\nText here"
    assert source.text == "# Setup\r\n\r\nText\t here"
    assert source.metadata == original_metadata


def test_normalization_is_idempotent():
    normalizer = TextNormalizer()
    once = normalizer.normalize_text("\ufeffA\t  B\r\n\r\n\r\nC")

    assert normalizer.normalize_text(once) == once
