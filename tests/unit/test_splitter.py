"""Unit tests for the ingestion text splitter."""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "src" / "ingestion").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


sys.path.insert(0, str(find_project_root()))

from src.ingestion import DocumentRecord, TextLoader, split_document, split_documents


def make_record(text: str, document_id: str = "doc-1", metadata=None) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        kb_id="kb-1",
        filename="example.txt",
        file_type="txt",
        text=text,
        checksum="checksum",
        metadata={} if metadata is None else metadata,
    )


def test_regular_text_is_split_and_returns_new_documents():
    source = Document(page_content="第一句。第二句。", metadata={"source": "a.txt"})

    chunks = split_documents(source, chunk_size=5, chunk_overlap=0)

    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert [chunk.page_content for chunk in chunks] == ["第一句。", "第二句。"]
    assert chunks[0] is not source
    assert source.metadata == {"source": "a.txt"}


def test_long_text_generates_multiple_chunks_and_overlap_is_applied():
    chunks = split_document(
        make_record("abcdefghij"),
        chunk_size=5,
        chunk_overlap=2,
        separators=[""],
    )

    assert [chunk.page_content for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert chunks[0].page_content[-2:] == chunks[1].page_content[:2]
    assert chunks[1].page_content[-2:] == chunks[2].page_content[:2]


def test_short_text_does_not_generate_extra_chunks():
    chunks = split_document(make_record("短文本"), chunk_size=500)

    assert len(chunks) == 1
    assert chunks[0].page_content == "短文本"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_and_whitespace_documents_produce_no_chunks(text):
    assert split_document(make_record(text)) == []


def test_metadata_is_preserved_and_chunk_metadata_is_deterministic():
    original_metadata = {"source": "manual.md", "page": 3, "file_name": "kept.md"}
    source = Document(page_content="abcdefgh", metadata=original_metadata)

    first = split_documents(source, chunk_size=4, chunk_overlap=0, separators=[""])
    second = split_documents(source, chunk_size=4, chunk_overlap=0, separators=[""])

    assert [chunk.metadata["page"] for chunk in first] == [3, 3]
    assert [chunk.metadata["file_name"] for chunk in first] == ["kept.md", "kept.md"]
    assert [chunk.metadata["chunk_index"] for chunk in first] == [0, 1]
    assert [chunk.metadata["total_chunks"] for chunk in first] == [2, 2]
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert original_metadata == {
        "source": "manual.md",
        "page": 3,
        "file_name": "kept.md",
    }


def test_document_record_source_metadata_is_available_on_chunks():
    chunks = split_document(
        make_record("正文", document_id="doc-42", metadata={"page_count": 2})
    )

    assert chunks[0].metadata["document_id"] == "doc-42"
    assert chunks[0].metadata["source"] == "example.txt"
    assert chunks[0].metadata["page_count"] == 2
    assert chunks[0].metadata["chunk_id"] == "doc-42:chunk-0"


def test_loader_output_can_be_passed_directly_to_splitter(tmp_path):
    source_file = tmp_path / "loaded.txt"
    source_file.write_text("加载后直接切分。第二句话。", encoding="utf-8")
    loaded = TextLoader().load(source_file, kb_id="kb-9", document_id="loaded-1")

    chunks = split_documents(loaded, chunk_size=8, chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].metadata["document_id"] == "loaded-1"
    assert chunks[0].metadata["kb_id"] == "kb-9"
    assert chunks[0].metadata["source"] == "loaded.txt"


def test_chunk_index_starts_at_zero_for_each_input_document():
    sources = [
        Document(page_content="abcdefgh", metadata={"document_id": "one"}),
        Document(page_content="ijklmnop", metadata={"document_id": "two"}),
    ]

    chunks = split_documents(sources, chunk_size=4, chunk_overlap=0, separators=[""])

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 0, 1]
    assert [chunk.metadata["document_id"] for chunk in chunks] == [
        "one",
        "one",
        "two",
        "two",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be a positive integer"),
        (-1, 0, "chunk_size must be a positive integer"),
        (5, -1, "chunk_overlap must be a non-negative integer"),
        (5, 5, "chunk_overlap must be smaller than chunk_size"),
        (5, 6, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_invalid_chunk_options_raise_clear_errors(chunk_size, chunk_overlap, message):
    with pytest.raises(ValueError, match=message):
        split_document(
            make_record("text"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_separators_must_be_a_sequence_not_a_string():
    with pytest.raises(ValueError, match="separators must be a non-empty sequence"):
        split_document(make_record("text"), separators="\n")


def test_chinese_paragraph_and_sentence_boundaries_are_preferred():
    text = "第一段内容。\n\n第二段内容！\n\n第三段内容？"

    chunks = split_document(make_record(text), chunk_size=8, chunk_overlap=0)

    assert len(chunks) == 3
    assert all(chunk.page_content[-1] in "。！？" for chunk in chunks)


def test_page_metadata_survives_for_multiple_page_documents():
    pages = [
        Document(page_content="第一页内容", metadata={"source": "book.pdf", "page": 1}),
        Document(page_content="第二页内容", metadata={"source": "book.pdf", "page": 2}),
    ]

    chunks = split_documents(pages, chunk_size=500)

    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        "book.pdf:page-1:chunk-0",
        "book.pdf:page-2:chunk-0",
    ]
