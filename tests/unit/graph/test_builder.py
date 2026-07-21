from src.graph.builder import GraphIndexBuilder
from src.models.graph import GraphBuildResult
from src.models.knowledge import ChunkRecord


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "rag.md", "page": 1, "section": "graph"},
    )


def test_builder_builds_complete_graph_result_from_chunks():
    chunks = [
        make_chunk("chunk-1", "RAG depends on Retrieval."),
        make_chunk("chunk-2", "Retrieval contains Vector Search."),
    ]

    result = GraphIndexBuilder().build_from_chunks(kb_id="kb-1", chunks=chunks)

    assert isinstance(result, GraphBuildResult)
    assert result.kb_id == "kb-1"
    assert result.entity_count == 3
    assert result.relationship_count == 2
    assert result.community_count == 1
    assert result.report_count == 1
    assert result.duration_ms >= 0
    assert result.metadata["networkx_node_count"] == 3
    assert result.metadata["networkx_edge_count"] == 2
    assert all(entity.source_refs for entity in result.entities)
    assert all(relationship.source_refs for relationship in result.relationships)
    assert result.reports[0].source_refs


def test_builder_handles_empty_input_with_warning():
    result = GraphIndexBuilder().build_from_chunks(kb_id="kb-empty", chunks=[])

    assert result.entity_count == 0
    assert result.relationship_count == 0
    assert result.community_count == 0
    assert result.report_count == 0
    assert "no chunks provided" in result.warnings


def test_builder_handles_no_relationship_input():
    result = GraphIndexBuilder().build_from_chunks(
        kb_id="kb-1",
        chunks=[make_chunk("chunk-1", "`Standalone Entity` appears here.")],
    )

    assert result.entity_count == 1
    assert result.relationship_count == 0
    assert result.community_count == 1
    assert result.report_count == 1
