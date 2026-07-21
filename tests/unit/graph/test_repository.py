import pytest

from src.graph.extractor import stable_entity_id, stable_relationship_id
from src.graph.index import NetworkXGraphIndex
from src.graph.repository import DanglingRelationshipError, InMemoryGraphRepository
from src.models.graph import GraphEntity, GraphRelationship, GraphSourceRef


def source_ref(chunk_id="chunk-1"):
    return GraphSourceRef(document_id="doc-1", chunk_id=chunk_id, quote="quote")


def entity(name):
    return GraphEntity(
        id=stable_entity_id(name),
        name=name,
        source_refs=[source_ref()],
    )


def relationship(source, target):
    return GraphRelationship(
        id=stable_relationship_id(source.id, target.id, "depends_on"),
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation_type="depends_on",
        source_refs=[source_ref()],
    )


def test_repository_adds_and_lists_entities_by_kb():
    repo = InMemoryGraphRepository()
    repo.add_entities("kb-1", [entity("RAG")])
    repo.add_entities("kb-2", [entity("Graph")])

    assert [item.name for item in repo.list_entities("kb-1")] == ["RAG"]
    assert [item.name for item in repo.list_entities("kb-2")] == ["Graph"]


def test_repository_merges_duplicate_entities_source_refs():
    repo = InMemoryGraphRepository()
    first = entity("RAG")
    second = first.model_copy(update={"source_refs": [source_ref("chunk-2")]})

    repo.add_entities("kb-1", [first, second])

    stored = repo.get_entity("kb-1", first.id)
    assert stored is not None
    assert {ref.chunk_id for ref in stored.source_refs} == {"chunk-1", "chunk-2"}


def test_repository_rejects_dangling_relationships():
    repo = InMemoryGraphRepository()
    source = entity("RAG")
    target = entity("Vector Search")

    with pytest.raises(DanglingRelationshipError):
        repo.add_relationships("kb-1", [relationship(source, target)])


def test_networkx_graph_has_nodes_edges_and_source_refs():
    repo = InMemoryGraphRepository()
    source = entity("RAG")
    target = entity("Vector Search")
    rel = relationship(source, target)
    repo.add_entities("kb-1", [source, target])
    repo.add_relationships("kb-1", [rel])

    graph = NetworkXGraphIndex(repo).to_networkx("kb-1")

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.nodes[source.id]["source_refs"][0]["chunk_id"] == "chunk-1"
    edge_data = graph.get_edge_data(source.id, target.id)
    assert edge_data["relationship_ids"] == [rel.id]
    assert edge_data["source_refs"][0]["chunk_id"] == "chunk-1"
