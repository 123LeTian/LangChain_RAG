from src.evaluation.dataset import ExpectedSource
from src.evaluation.retrieval_metrics import hit_at_k, mrr, source_matches
from src.models.knowledge import ChunkRecord
from src.models.rag import RAGChunk, RAGCitation, RAGSource
from src.models.schemas import RetrievalHit


def test_hit_at_k_uses_expected_source_identifiers():
    expected = [ExpectedSource(document_id="doc-2", chunk_id="chunk-2")]
    hits = [
        {"document_id": "doc-1", "chunk_id": "chunk-1"},
        {"document_id": "doc-2", "chunk_id": "chunk-2"},
    ]

    assert hit_at_k(hits, expected, k=1) == 0.0
    assert hit_at_k(hits, expected, k=3) == 1.0


def test_mrr_returns_first_matching_reciprocal_rank():
    expected = [ExpectedSource(document_id="doc-3")]
    hits = [
        {"document_id": "doc-1"},
        {"document_id": "doc-2"},
        {"document_id": "doc-3"},
    ]

    assert mrr(hits, expected) == 1 / 3


def test_empty_hits_are_legal_scores():
    expected = [ExpectedSource(document_id="doc-1")]

    assert hit_at_k([], expected, k=5) == 0.0
    assert mrr([], expected) == 0.0


def test_source_matching_supports_rag_chunk_and_citation_contracts():
    expected = [ExpectedSource(document_id="doc-1", chunk_id="chunk-1")]
    chunk = RAGChunk(
        chunk_id="chunk-1",
        content="text",
        source=RAGSource(document_id="doc-1", source_path="folder/rag.md"),
    )
    citation = RAGCitation(
        document_id="doc-1",
        chunk_id="chunk-1",
        text_snippet="text",
    )

    assert hit_at_k([chunk], expected, k=1) == 1.0
    assert source_matches(expected[0], citation)


def test_source_matching_supports_legacy_retrieval_hit_and_filename_only():
    expected = ExpectedSource(filename="guide.pdf")
    hit = RetrievalHit(
        chunk=ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            kb_id="kb-1",
            text="text",
            index=0,
            metadata={"filename": "guide.pdf"},
        ),
        score=0.9,
        rank=1,
        metadata={"filename": "guide.pdf"},
    )

    assert source_matches(expected, hit)


def test_source_matching_rejects_conflicting_chunk_id():
    expected = ExpectedSource(document_id="doc-1", chunk_id="chunk-expected")
    candidate = {"document_id": "doc-1", "chunk_id": "chunk-other"}

    assert not source_matches(expected, candidate)
