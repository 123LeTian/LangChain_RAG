from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphRetriever
from src.models.knowledge import ChunkRecord


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "graph.md", "page": 1, "section": "demo"},
    )


def build_retriever():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    return GraphRetriever(builder.repository)


def test_local_search_hits_entity_and_expands_one_hop():
    result = build_retriever().local_search("RAG", kb_id="kb-1", top_k=3)

    assert not result.warnings
    assert result.hits
    hit = result.hits[0]
    assert hit.entity_name == "RAG"
    assert hit.relationships
    assert [entity.name for entity in hit.neighbor_entities] == ["Retrieval"]


def test_local_search_returns_path_and_chunk_traceability():
    result = build_retriever().local_search("Retrieval", kb_id="kb-1", top_k=1)

    hit = result.hits[0]
    assert hit.path[0].startswith("query:")
    assert any(part.startswith("source_chunk:doc-1/") for part in hit.path)
    assert hit.source_refs[0].document_id == "doc-1"
    assert hit.source_refs[0].chunk_id in {"chunk-1", "chunk-2"}


def test_local_search_empty_or_no_match_is_legal():
    empty = GraphRetriever(GraphIndexBuilder().repository).local_search(
        "RAG", kb_id="missing", top_k=3
    )
    no_match = build_retriever().local_search("Nonexistent", kb_id="kb-1", top_k=3)

    assert empty.hits == []
    assert empty.warnings
    assert no_match.hits == []
    assert no_match.warnings == ["no matching graph entities found"]


def test_global_search_hits_community_report_without_raw_graph_text_stuffing():
    result = build_retriever().global_search("overall Retrieval theme", kb_id="kb-1")

    assert result.hits
    hit = result.hits[0]
    assert hit.community_id
    assert hit.report_id
    assert "Community:" in hit.title
    assert "contains" in hit.summary
    assert "RAG depends on Retrieval." not in hit.summary
    assert hit.source_refs[0].document_id == "doc-1"


def test_graph_search_protocol_returns_rag_context():
    context = build_retriever().graph_search(
        "overall Retrieval theme",
        kb_id="kb-1",
        scope="global",
        top_k=1,
    )

    assert context.retrieval_method == "graph_global"
    assert len(context.chunks) == 1
    assert context.chunks[0].source.metadata["graph_scope"] == "global"
