from src.graph.builder import GraphIndexBuilder
from src.graph.repository import InMemoryGraphRepository
from src.graph.retriever import GraphRetriever, resolve_graph_scope
from src.models.graph import CommunityReport, GraphSourceRef
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


def test_resolve_graph_scope_prefers_overview_and_strategy_terms_for_global():
    assert resolve_graph_scope("贵州茅台2025年的经营重点可以归纳成哪些板块？") == "global"
    assert resolve_graph_scope("年报摘要里最能体现公司经营逻辑的主线是什么？") == "global"
    assert resolve_graph_scope("报告里提到的几项经营优势如何共同支撑公司的长期发展？") == "global"


def source_ref(chunk_id="chunk-x"):
    return GraphSourceRef(
        document_id="doc-1",
        chunk_id=chunk_id,
        filename="graph.md",
        quote="source quote",
    )


def report(report_id, entities, relationships, refs=1):
    return CommunityReport(
        id=report_id,
        community_id=f"comm-{report_id}",
        title="Community: " + ", ".join(entities),
        summary="summary",
        key_entities=entities,
        key_relationships=relationships,
        source_refs=[source_ref(f"{report_id}-{index}") for index in range(refs)],
    )


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


def test_local_search_finds_chinese_sales_channel_relationships():
    text = (
        "\u9500\u552e\u6a21\u5f0f\u4e3a\uff1a\u516c\u53f8\u4ea7\u54c1\u901a\u8fc7"
        "\u76f4\u9500\u548c\u6279\u53d1\u4ee3\u7406\u6e20\u9053\u8fdb\u884c\u9500\u552e\u3002"
        "\u76f4\u9500\u6e20\u9053\u6307\u81ea\u8425\u548c\u201ci \u8305\u53f0\u201d"
        "\u7b49\u6570\u5b57\u8425\u9500\u5e73\u53f0\u6e20\u9053\uff0c"
        "\u6279\u53d1\u4ee3\u7406\u6e20\u9053\u6307\u793e\u4f1a\u7ecf\u9500\u5546\u3001"
        "\u5546\u8d85\u3001\u7535\u5546\u7b49\u6e20\u9053\u3002"
    )
    builder = GraphIndexBuilder()
    builder.build_from_chunks(kb_id="kb-1", chunks=[make_chunk("moutai-1", text)])

    result = GraphRetriever(builder.repository).local_search(
        (
            "\u8d35\u5dde\u8305\u53f0\u7684\u76f4\u9500\u6e20\u9053\u548c"
            "\u6279\u53d1\u4ee3\u7406\u6e20\u9053\u5206\u522b\u5305\u62ec\u54ea\u4e9b\u4e3b\u4f53\uff1f"
            "\u5b83\u4eec\u4e0e\u9500\u552e\u6a21\u5f0f\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f"
        ),
        kb_id="kb-1",
        top_k=5,
    )

    assert not result.warnings
    hit_names = {hit.entity_name for hit in result.hits}
    assert "\u76f4\u9500\u6e20\u9053" in hit_names
    assert "\u6279\u53d1\u4ee3\u7406\u6e20\u9053" in hit_names
    assert "\u9500\u552e\u6a21\u5f0f" in hit_names
    descriptions = {
        relationship.description
        for hit in result.hits
        for relationship in hit.relationships
    }
    assert "\u76f4\u9500\u6e20\u9053 contains \u81ea\u8425" in descriptions
    assert "\u76f4\u9500\u6e20\u9053 contains i \u8305\u53f0" in descriptions
    assert "\u6279\u53d1\u4ee3\u7406\u6e20\u9053 contains \u793e\u4f1a\u7ecf\u9500\u5546" in descriptions
    assert "\u6279\u53d1\u4ee3\u7406\u6e20\u9053 contains \u5546\u8d85" in descriptions
    assert "\u6279\u53d1\u4ee3\u7406\u6e20\u9053 contains \u7535\u5546" in descriptions


def test_local_search_prefers_process_graph_over_same_chunk_noise():
    text = (
        "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b\u4e3a\uff1a"
        "\u5236\u66f2\u2014\u5236\u9152\u2014\u8d2e\u5b58\u2014\u52fe\u5151\u2014\u5305\u88c5\u3002"
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d\u662f"
        "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\u3002"
        "\u751f\u4ea7\u5de5\u827a\u72ec\u7279\u3002"
    )
    builder = GraphIndexBuilder()
    builder.build_from_chunks(kb_id="kb-1", chunks=[make_chunk("process-1", text)])

    result = GraphRetriever(builder.repository).local_search(
        (
            "\u8d35\u5dde\u8305\u53f0\u7684\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a"
            "\u6d41\u7a0b\u5305\u62ec\u54ea\u4e9b\u73af\u8282\uff1f"
            "\u8fd9\u4e9b\u73af\u8282\u4e4b\u95f4\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f"
        ),
        kb_id="kb-1",
        top_k=3,
    )

    assert not result.warnings
    assert result.hits[0].entity_name == "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b"
    top_descriptions = [rel.description for rel in result.hits[0].relationships]
    assert "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b contains \u5236\u66f2" in top_descriptions
    assert "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b contains \u5305\u88c5" in top_descriptions


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


def test_global_search_falls_back_to_reports_when_query_terms_do_not_match():
    result = build_retriever().global_search("unmatched executive overview", kb_id="kb-1")

    assert result.hits
    assert result.warnings == []
    assert result.hits[0].metadata["score_reason"] == "global_fallback_no_keyword_match"
    assert result.hits[0].source_refs


def test_global_fallback_prefers_quality_topics_over_weak_reports():
    repository = InMemoryGraphRepository()
    repository.add_reports(
        "kb-1",
        [
            report(
                "weak",
                ["上市", "上市公司股东的", "归"],
                ["归 belongs_to 上市"],
                refs=10,
            ),
            report(
                "quality",
                ["产品品质卓越", "品质根基更加牢固"],
                ["产品品质卓越 relates_to 品质根基更加牢固"],
                refs=1,
            ),
            report(
                "market",
                ["品牌美誉度高", "市场基本盘保持稳固"],
                ["品牌美誉度高 relates_to 市场基本盘保持稳固"],
                refs=1,
            ),
        ],
    )

    result = GraphRetriever(repository).global_search(
        "unmatched executive overview",
        kb_id="kb-1",
        top_k=2,
    )

    titles = [hit.title for hit in result.hits]
    assert all("上市公司股东的" not in title for title in titles)
    assert any("产品品质卓越" in title for title in titles)
    assert any("品牌美誉度高" in title for title in titles)


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
