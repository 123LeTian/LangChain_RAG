from src.graph.community import stable_community_id
from src.graph.extractor import stable_entity_id, stable_relationship_id
from src.graph.reports import CommunityReportBuilder
from src.models.graph import (
    GraphCommunity,
    GraphEntity,
    GraphRelationship,
    GraphSourceRef,
)


def source_ref():
    return GraphSourceRef(document_id="doc-1", chunk_id="chunk-1", quote="RAG depends on retrieval.")


def test_community_report_is_generated_with_source_refs():
    rag = GraphEntity(id=stable_entity_id("RAG"), name="RAG", source_refs=[source_ref()])
    retrieval = GraphEntity(
        id=stable_entity_id("Retrieval"),
        name="Retrieval",
        source_refs=[source_ref()],
    )
    relationship = GraphRelationship(
        id=stable_relationship_id(rag.id, retrieval.id, "depends_on"),
        source_entity_id=rag.id,
        target_entity_id=retrieval.id,
        relation_type="depends_on",
        description="RAG depends_on Retrieval",
        source_refs=[source_ref()],
    )
    community = GraphCommunity(
        id=stable_community_id([rag.id, retrieval.id]),
        entity_ids=[rag.id, retrieval.id],
        relationship_ids=[relationship.id],
    )

    report = CommunityReportBuilder().build(
        community,
        entities=[rag, retrieval],
        relationships=[relationship],
    )

    assert report.community_id == community.id
    assert report.title.startswith("Community:")
    assert report.key_entities == ["RAG", "Retrieval"]
    assert report.key_relationships == ["RAG depends_on Retrieval"]
    assert report.source_refs[0].document_id == "doc-1"
    assert report.source_refs[0].chunk_id == "chunk-1"


def test_empty_community_gets_legal_report():
    report = CommunityReportBuilder().build(
        GraphCommunity(id="comm-empty"),
        entities=[],
        relationships=[],
    )

    assert report.title == "Empty Graph Community"
    assert report.source_refs == []
