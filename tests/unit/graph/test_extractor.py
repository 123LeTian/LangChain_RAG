from src.graph.extractor import RuleBasedGraphExtractor, stable_entity_id
from src.models.knowledge import ChunkRecord


def make_chunk(
    chunk_id="chunk-1",
    text="RAG is retrieval augmented generation. RAG depends on Vector Search.",
):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={
            "filename": "rag.md",
            "page": 2,
            "section": "intro",
        },
    )


def test_extracts_entities_from_chunk_like_object():
    result = RuleBasedGraphExtractor().extract_from_chunk(
        make_chunk(text="`Graph Index` contains Entity nodes.")
    )

    names = {entity.name for entity in result.entities}
    assert "Graph Index" in names
    assert "Entity nodes" in names


def test_extracts_relationships_and_source_refs_from_chunk():
    result = RuleBasedGraphExtractor().extract_from_chunk(make_chunk())

    assert result.relationships
    relationship = result.relationships[0]
    assert relationship.source_entity_id in {entity.id for entity in result.entities}
    assert relationship.target_entity_id in {entity.id for entity in result.entities}
    assert relationship.source_refs[0].document_id == "doc-1"
    assert relationship.source_refs[0].chunk_id == "chunk-1"


def test_entity_source_refs_include_document_and_chunk_ids():
    result = RuleBasedGraphExtractor().extract_from_chunk(make_chunk())

    for entity in result.entities:
        assert entity.source_refs[0].document_id == "doc-1"
        assert entity.source_refs[0].chunk_id == "chunk-1"
        assert entity.source_refs[0].filename == "rag.md"


def test_duplicate_entities_are_normalized_across_chunks():
    result = RuleBasedGraphExtractor().extract_from_chunks(
        [
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "rag depends on Generation."),
        ]
    )

    rag_id = stable_entity_id("RAG")
    rag_entities = [entity for entity in result.entities if entity.id == rag_id]
    assert len(rag_entities) == 1
    assert {ref.chunk_id for ref in rag_entities[0].source_refs} == {
        "chunk-1",
        "chunk-2",
    }


def test_bad_chunk_returns_warning_instead_of_raising():
    result = RuleBasedGraphExtractor().extract_from_chunk(
        {"id": "missing-doc", "text": "RAG depends on Retrieval."}
    )

    assert result.entities == []
    assert result.relationships == []
    assert result.warnings
