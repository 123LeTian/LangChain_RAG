import asyncio

from src.graph.builder import GraphIndexBuilder
from src.graph.retriever import GraphGlobalSearchHit, GraphRetriever, GraphSearchHit
from src.models.graph import GraphEntity, GraphRelationship, GraphSourceRef
from src.models.knowledge import ChunkRecord
from src.models.rag import RAGContext, RAGMode, RAGRequest, RAGResult, TraceStage
from src.rag.strategies.graph_rag import (
    GraphRAGStrategy,
    REFUSAL_ANSWER,
    _focus_local_hits_for_query,
    _select_local_hits_for_query,
    _select_global_hits_for_query,
    _source_refs_from_global,
)


def run(coro):
    return asyncio.run(coro)


def make_chunk(chunk_id, text):
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        kb_id="kb-1",
        text=text,
        index=0,
        metadata={"filename": "graph.md", "page": 1, "section": "demo"},
    )


def build_strategy():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk("chunk-1", "RAG depends on Retrieval."),
            make_chunk("chunk-2", "Retrieval contains Vector Search."),
        ],
    )
    return GraphRAGStrategy(GraphRetriever(builder.repository))


def build_process_strategy():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk(
                "process-1",
                (
                    "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b\u4e3a\uff1a"
                    "\u5236\u66f2\u2014\u5236\u9152\u2014\u8d2e\u5b58\u2014\u52fe\u5151\u2014\u5305\u88c5\u3002"
                ),
            ),
        ],
    )
    return GraphRAGStrategy(GraphRetriever(builder.repository))


def build_noisy_process_strategy():
    builder = GraphIndexBuilder()
    builder.build_from_chunks(
        kb_id="kb-1",
        chunks=[
            make_chunk(
                "process-1",
                (
                    "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b\u4e3a\uff1a"
                    "\u5236\u66f2\u2014\u5236\u9152\u2014\u8d2e\u5b58\u2014\u52fe\u5151\u2014\u5305\u88c5\u3002"
                    "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d\u662f"
                    "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\u3002"
                    "\u751f\u4ea7\u5de5\u827a\u72ec\u7279\u3002"
                ),
            ),
        ],
    )
    return GraphRAGStrategy(GraphRetriever(builder.repository))


def context(llm=None):
    return RAGContext(
        query="RAG",
        chunks=[],
        llm=llm,
        metadata={"trace_id": "trace-graph", "kb_id": "kb-1"},
    )


def graph_ref():
    return GraphSourceRef(document_id="doc-1", chunk_id="chunk-1", quote="source")


def graph_entity(name):
    return GraphEntity(id=f"ent-{abs(hash(name))}", name=name, source_refs=[graph_ref()])


def graph_relationship(source, relation_type, target):
    return GraphRelationship(
        id=f"rel-{abs(hash((source, relation_type, target)))}",
        source_entity_id=f"ent-{abs(hash(source))}",
        target_entity_id=f"ent-{abs(hash(target))}",
        relation_type=relation_type,
        description=f"{source} {relation_type} {target}",
        source_refs=[graph_ref()],
    )


def graph_hit(name, relationships, neighbors=None, score=1.0):
    return GraphSearchHit(
        entity_id=f"ent-{abs(hash(name))}",
        entity_name=name,
        score=score,
        matched_text=name,
        relationships=relationships,
        neighbor_entities=[graph_entity(item) for item in (neighbors or [])],
        source_refs=[graph_ref()],
    )


def test_graph_rag_strategy_local_returns_answer_citations_and_trace():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(),
        )
    )

    assert isinstance(result, RAGResult)
    assert "\u5c40\u90e8\u68c0\u7d22" in result.answer
    assert "Graph local search" not in result.answer
    assert result.citations
    assert result.citations[0].document_id == "doc-1"
    assert result.citations[0].chunk_id
    assert result.hits[0].source.metadata["graph_scope"] == "local"
    assert [event.stage for event in result.trace] == [
        TraceStage.GRAPH_SEARCH,
        TraceStage.GENERATE,
        TraceStage.COMPLETE,
    ]


def test_graph_rag_strategy_global_returns_report_answer_citations_and_trace():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="overall Retrieval theme",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "global"},
            ),
            context(),
        )
    )

    assert "\u5168\u5c40\u68c0\u7d22" in result.answer
    assert "Graph global search" not in result.answer
    assert "community report" not in result.answer
    assert result.citations
    assert result.hits[0].source.metadata["community_id"]
    assert result.hits[0].source.metadata["report_id"]
    assert result.trace[0].stage == TraceStage.GRAPH_SEARCH
    assert "scope=global" in result.trace[0].input_summary


def test_global_source_refs_filter_weak_citation_quotes_for_display():
    weak = GraphSourceRef(
        document_id="doc-1",
        chunk_id="shareholders",
        filename="moutai.pdf",
        quote=(
            "\u5b63\u5ea6\u6570\u636e\u4e0e\u5df2\u62ab\u9732\u5b9a\u671f"
            "\u62a5\u544a\u6570\u636e\u5dee\u5f02\u8bf4\u660e\u30024\u3001"
            "\u80a1\u4e1c\u60c5\u51b5\uff1a\u524d\u5341\u540d\u80a1\u4e1c"
            "\u6301\u80a1\u60c5\u51b5\u3002"
        ),
    )
    cover = GraphSourceRef(
        document_id="doc-1",
        chunk_id="cover",
        filename="moutai.pdf",
        quote=(
            "\u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f8"
            "2025\u5e74\u5e74\u5ea6\u62a5\u544a\u6458\u8981\u3002"
            "\u516c\u53f8\u4ee3\u7801\uff1a600519\u3002"
            "\u6295\u8d44\u8005\u5e94\u5f53\u9605\u8bfb\u5e74\u62a5\u5168\u6587\u3002"
        ),
    )
    quality = GraphSourceRef(
        document_id="doc-1",
        chunk_id="advantages",
        filename="moutai.pdf",
        quote=(
            "\u516c\u53f8\u7ade\u4e89\u4f18\u52bf\uff1a\u4e00\u662f"
            "\u4ea7\u54c1\u54c1\u8d28\u5353\u8d8a\uff0c\u4e8c\u662f"
            "\u5e02\u573a\u57fa\u672c\u76d8\u4fdd\u6301\u7a33\u56fa\uff0c"
            "\u9500\u552e\u6e20\u9053\u6301\u7eed\u8986\u76d6\u3002"
        ),
    )
    hit = GraphGlobalSearchHit(
        community_id="comm-1",
        report_id="report-1",
        title="\u516c\u53f8\u6574\u4f53\u7ecf\u8425\u4e3b\u9898",
        summary="\u516c\u53f8\u6574\u4f53\u7ecf\u8425\u6d89\u53ca\u54c1\u8d28\u548c\u5e02\u573a\u3002",
        score=1.0,
        key_entities=["\u54c1\u8d28", "\u5e02\u573a"],
        key_relationships=["\u54c1\u8d28 relates_to \u5e02\u573a"],
        source_refs=[weak, cover, quality],
    )

    refs = _source_refs_from_global(
        "\u516c\u53f8\u6574\u4f53\u7ecf\u8425\u60c5\u51b5\u53ef\u4ee5\u6982\u62ec\u4e3a\u54ea\u4e9b\u4e3b\u8981\u4e3b\u9898\uff1f",
        [hit],
    )

    assert [ref.chunk_id for ref in refs] == ["advantages"]


def test_global_context_filters_weak_reports_before_generation():
    weak = GraphGlobalSearchHit(
        community_id="comm-shareholders",
        report_id="report-shareholders",
        title=(
            "Community: 45 \u672a\u77e5\u672a\u77e5"
            "\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216"
            "\u4e00\u81f4\u884c\u52a8\u7684\u8bf4\u660e"
        ),
        summary=(
            "Key relationship: 45 \u672a\u77e5\u672a\u77e5"
            "\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb "
            "is_a \u5426\u4e3a\u4e00\u81f4\u884c\u52a8\u4eba."
        ),
        score=99.0,
        key_entities=[
            "45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb",
            "\u5426\u4e3a\u4e00\u81f4\u884c\u52a8\u4eba",
        ],
        key_relationships=[
            "45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb "
            "is_a \u5426\u4e3a\u4e00\u81f4\u884c\u52a8\u4eba"
        ],
        source_refs=[
            GraphSourceRef(
                document_id="doc-1",
                chunk_id="shareholders",
                quote="\u80a1\u4e1c\u60c5\u51b5\uff1a\u524d\u5341\u540d\u80a1\u4e1c\u6301\u80a1\u60c5\u51b5\u3002",
            )
        ],
    )
    quality = GraphGlobalSearchHit(
        community_id="comm-quality",
        report_id="report-quality",
        title="\u516c\u53f8\u7ade\u4e89\u4f18\u52bf",
        summary=(
            "\u516c\u53f8\u7ade\u4e89\u4f18\u52bf\u5305\u62ec"
            "\u4ea7\u54c1\u54c1\u8d28\u5353\u8d8a\u548c"
            "\u5e02\u573a\u57fa\u672c\u76d8\u4fdd\u6301\u7a33\u56fa\u3002"
        ),
        score=1.0,
        key_entities=["\u4ea7\u54c1\u54c1\u8d28\u5353\u8d8a", "\u5e02\u573a\u57fa\u672c\u76d8"],
        key_relationships=["\u4ea7\u54c1\u54c1\u8d28\u5353\u8d8a relates_to \u5e02\u573a\u57fa\u672c\u76d8"],
        source_refs=[
            GraphSourceRef(
                document_id="doc-1",
                chunk_id="quality",
                quote="\u4ea7\u54c1\u54c1\u8d28\u5353\u8d8a\uff0c\u5e02\u573a\u57fa\u672c\u76d8\u4fdd\u6301\u7a33\u56fa\u3002",
            )
        ],
    )

    selected = _select_global_hits_for_query(
        "\u91cd\u8981\u4e8b\u9879\u4e3b\u8981\u6d89\u53ca\u54ea\u4e9b\u4e3b\u9898\uff1f",
        [weak, quality],
    )

    assert [hit.report_id for hit in selected] == ["report-quality"]


def test_global_context_keeps_shareholder_reports_for_shareholder_questions():
    weak = GraphGlobalSearchHit(
        community_id="comm-shareholders",
        report_id="report-shareholders",
        title="\u80a1\u4e1c\u5173\u8054\u5173\u7cfb",
        summary="\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216\u4e00\u81f4\u884c\u52a8\u8bf4\u660e\u3002",
        score=1.0,
        key_entities=["\u80a1\u4e1c\u5173\u8054\u5173\u7cfb"],
        key_relationships=["\u80a1\u4e1c\u5173\u8054\u5173\u7cfb relates_to \u4e00\u81f4\u884c\u52a8"],
        source_refs=[
            GraphSourceRef(
                document_id="doc-1",
                chunk_id="shareholders",
                quote="\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216\u4e00\u81f4\u884c\u52a8\u8bf4\u660e\u3002",
            )
        ],
    )

    selected = _select_global_hits_for_query(
        "\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u548c\u4e00\u81f4\u884c\u52a8\u60c5\u51b5\u662f\u4ec0\u4e48\uff1f",
        [weak],
    )

    assert selected == [weak]


def test_graph_rag_strategy_auto_selects_global_for_summary_queries():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="summarize the overall Retrieval theme",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )

    assert result.hits[0].source.metadata["graph_scope"] == "global"


def test_graph_rag_strategy_auto_selects_global_for_strategic_overview_queries():
    result_a = run(
        build_strategy().run(
            RAGRequest(
                query="贵州茅台2025年的经营重点可以归纳成哪些板块？",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )
    result_b = run(
        build_strategy().run(
            RAGRequest(
                query="年报摘要里最能体现公司经营逻辑的主线是什么？",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )

    assert result_a.hits[0].source.metadata["graph_scope"] == "global"
    assert result_b.hits[0].source.metadata["graph_scope"] == "global"
    assert (
        run(
            build_strategy().run(
                RAGRequest(
                    query="品牌影响力再跃新阶说明什么经营结果？",
                    kb_id="kb-1",
                    mode=RAGMode.GRAPH,
                    options={"graph_scope": "auto"},
                ),
                context(),
            )
        ).hits[0].source.metadata["graph_scope"]
        == "global"
    )


def test_graph_rag_strategy_auto_selects_local_by_default():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "auto"},
            ),
            context(),
        )
    )

    assert result.hits[0].source.metadata["graph_scope"] == "local"


def test_graph_rag_strategy_local_summarizes_process_sequence():
    result = run(
        build_process_strategy().run(
            RAGRequest(
                query=(
                    "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b"
                    "\u5305\u62ec\u54ea\u4e9b\u73af\u8282\uff1f"
                    "\u8fd9\u4e9b\u73af\u8282\u4e4b\u95f4\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f"
                ),
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(),
        )
    )

    assert "\u6d41\u7a0b\u987a\u5e8f" in result.answer
    assert "\u5236\u66f2 -> \u5236\u9152 -> \u8d2e\u5b58 -> \u52fe\u5151 -> \u5305\u88c5" in result.answer


def test_graph_rag_strategy_local_filters_process_noise_from_answer():
    result = run(
        build_noisy_process_strategy().run(
            RAGRequest(
                query=(
                    "\u4ea7\u54c1\u751f\u4ea7\u5de5\u827a\u6d41\u7a0b"
                    "\u5305\u62ec\u54ea\u4e9b\u73af\u8282\uff1f"
                    "\u8fd9\u4e9b\u73af\u8282\u4e4b\u95f4\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f"
                ),
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local", "top_k": 5},
            ),
            context(),
        )
    )

    assert "\u6d41\u7a0b\u987a\u5e8f" in result.answer
    assert "\u4e3b\u5bfc\u4ea7\u54c1" not in result.answer
    assert "\u5178\u578b\u4ee3\u8868" not in result.answer
    assert len(result.hits) == 1


def test_local_selector_focuses_regular_entity_relationships_and_filters_noise():
    useful = graph_hit(
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d",
        [
            graph_relationship(
                "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d",
                "is_a",
                "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\uff0c\u516c\u53f8\u8425\u9500\u7f51\u7edc\u8986\u76d6\u56fd\u5185\u5e02\u573a",
            )
        ],
        neighbors=[
            "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\uff0c\u516c\u53f8\u8425\u9500\u7f51\u7edc\u8986\u76d6\u56fd\u5185\u5e02\u573a"
        ],
        score=3,
    )
    noisy = graph_hit(
        "\u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a\u6458\u8981\u4e09",
        [
            graph_relationship(
                "\u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a\u6458\u8981\u4e09",
                "is_a",
                "\u54c1\u724c\u5f71\u54cd\u529b\u518d\u8dc3\u65b0\u9636",
            )
        ],
        score=10,
    )

    selected = _select_local_hits_for_query(
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d\u7684\u5b9a\u4f4d\u662f\u4ec0\u4e48\uff1f\u5b83\u4e0e\u516c\u53f8\u8425\u9500\u7f51\u7edc\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f",
        [noisy, useful],
    )

    assert [hit.entity_name for hit in selected] == [
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d"
    ]


def test_local_selector_cleans_main_business_noise_and_keeps_target_relation():
    useful = graph_hit(
        "com \u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a\u6458\u8981 2\u3001 \u62a5\u544a\u671f\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1\u7b80\u4ecb\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1",
        [
            graph_relationship(
                "com \u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f82025\u5e74\u5e74\u5ea6\u62a5\u544a\u6458\u8981 2\u3001 \u62a5\u544a\u671f\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1\u7b80\u4ecb\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1",
                "is_a",
                "\u8305\u53f0\u9152\u53ca\u7cfb\u5217\u9152\u7684\u751f\u4ea7\u4e0e\u9500\u552e",
            )
        ],
        neighbors=["\u8305\u53f0\u9152\u53ca\u7cfb\u5217\u9152\u7684\u751f\u4ea7\u4e0e\u9500\u552e"],
        score=2,
    )
    noisy = graph_hit(
        "45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216\u4e00\u81f4\u884c\u52a8\u7684\u8bf4\u660e",
        [graph_relationship("45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb", "is_a", "\u5426\u4e3a\u4e00\u81f4\u884c\u52a8\u4eba")],
        score=20,
    )
    off_topic = graph_hit(
        "\u8d35\u5dde\u8305\u53f0\u9152\u6838\u5fc3\u4ea7\u533a\u7684\u72ec\u7279\u7279\u5f81",
        [
            graph_relationship(
                "\u7279\u6b8a\u7684\u5730\u5f62\u5730\u8c8c\u3001\u6c14\u5019\u73af\u5883",
                "is_a",
                "\u8d35\u5dde\u8305\u53f0\u9152\u6838\u5fc3\u4ea7\u533a\u7684\u72ec\u7279\u7279\u5f81",
            )
        ],
        score=10,
    )

    selected = _select_local_hits_for_query(
        "\u516c\u53f8\u4e3b\u8981\u4e1a\u52a1\u662f\u4ec0\u4e48\uff1f\u5b83\u4e0e\u8305\u53f0\u9152\u53ca\u7cfb\u5217\u9152\u7684\u751f\u4ea7\u4e0e\u9500\u552e\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f",
        [noisy, off_topic, useful],
    )

    assert len(selected) == 1
    assert selected[0].entity_name.startswith("com ")


def test_local_focus_keeps_relevant_relationships_and_drops_noisy_ones():
    useful = graph_hit(
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d",
        [
            graph_relationship(
                "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d",
                "is_a",
                "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\uff0c\u96c6\u56fd\u5bb6\u5730\u7406\u6807\u5fd7\u4ea7\u54c1\u3001\u6709\u673a\u98df\u54c1\u548c\u56fd\u5bb6\u975e\u7269\u8d28\u6587\u5316\u9057\u4ea7\u4e8e\u4e00\u8eab\uff0c\u516c\u53f8\u8425\u9500\u7f51\u7edc\u8986\u76d6\u56fd\u5185\u5e02\u573a\u53ca\u4e94\u5927\u6d32 66 \u4e2a\u56fd\u5bb6\u548c\u5730\u533a",
            ),
            graph_relationship(
                "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d",
                "is_a",
                "45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216\u4e00\u81f4\u884c\u52a8\u7684\u8bf4\u660e",
            ),
        ],
        neighbors=[
            "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\uff0c\u96c6\u56fd\u5bb6\u5730\u7406\u6807\u5fd7\u4ea7\u54c1\u3001\u6709\u673a\u98df\u54c1\u548c\u56fd\u5bb6\u975e\u7269\u8d28\u6587\u5316\u9057\u4ea7\u4e8e\u4e00\u8eab\uff0c\u516c\u53f8\u8425\u9500\u7f51\u7edc\u8986\u76d6\u56fd\u5185\u5e02\u573a\u53ca\u4e94\u5927\u6d32 66 \u4e2a\u56fd\u5bb6\u548c\u5730\u533a",
            "45 \u672a\u77e5\u672a\u77e5\u4e0a\u8ff0\u80a1\u4e1c\u5173\u8054\u5173\u7cfb\u6216\u4e00\u81f4\u884c\u52a8\u7684\u8bf4\u660e",
        ],
        score=10,
    )

    focused = _focus_local_hits_for_query(
        "\u4e3b\u5bfc\u4ea7\u54c1\u201c\u8d35\u5dde\u8305\u53f0\u9152\u201d\u7684\u5b9a\u4f4d\u662f\u4ec0\u4e48\uff1f\u5b83\u4e0e\u516c\u53f8\u8425\u9500\u7f51\u7edc\u662f\u4ec0\u4e48\u5173\u7cfb\uff1f",
        [useful],
    )

    assert len(focused) == 1
    assert len(focused[0].relationships) == 1
    assert "45 \u672a\u77e5" not in focused[0].relationships[0].description
    assert [entity.name for entity in focused[0].neighbor_entities] == [
        "\u6211\u56fd\u5927\u66f2\u9171\u9999\u578b\u767d\u9152\u7684\u5178\u578b\u4ee3\u8868\uff0c\u96c6\u56fd\u5bb6\u5730\u7406\u6807\u5fd7\u4ea7\u54c1\u3001\u6709\u673a\u98df\u54c1\u548c\u56fd\u5bb6\u975e\u7269\u8d28\u6587\u5316\u9057\u4ea7\u4e8e\u4e00\u8eab\uff0c\u516c\u53f8\u8425\u9500\u7f51\u7edc\u8986\u76d6\u56fd\u5185\u5e02\u573a\u53ca\u4e94\u5927\u6d32 66 \u4e2a\u56fd\u5bb6\u548c\u5730\u533a"
    ]


def test_graph_rag_strategy_no_match_refuses_with_warning():
    result = run(
        build_strategy().run(
            RAGRequest(
                query="Nonexistent",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(),
        )
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.warnings == ["no matching graph entities found"]


def test_graph_rag_strategy_uses_injected_llm_when_available():
    class FakeLLM:
        async def generate_with_tokens(self, prompt, context):
            return "LLM graph answer", {"prompt": 3, "completion": 4, "total": 7}

    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(llm=FakeLLM()),
        )
    )

    assert result.answer == "LLM graph answer"
    assert result.usage == {"prompt": 3, "completion": 4, "total": 7}


def test_graph_rag_strategy_uses_langchain_invoke_llm_when_available():
    class FakeMessage:
        content = "Natural answer from invoke"
        usage_metadata = {
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
        }

    class FakeLLM:
        def __init__(self):
            self.prompt = ""

        def invoke(self, prompt):
            self.prompt = prompt
            return FakeMessage()

    llm = FakeLLM()

    result = run(
        build_strategy().run(
            RAGRequest(
                query="RAG",
                kb_id="kb-1",
                mode=RAGMode.GRAPH,
                options={"graph_scope": "local"},
            ),
            context(llm=llm),
        )
    )

    assert result.answer == "Natural answer from invoke"
    assert result.usage == {
        "input_tokens": 11,
        "output_tokens": 5,
        "total_tokens": 16,
    }
    assert "Question:\nRAG" in llm.prompt
    assert "Graph evidence:" in llm.prompt
    assert "For global, overview, important-matter, or theme questions" in llm.prompt
    assert "Do not refuse solely" in llm.prompt
    assert "do not say 'the two are the same relationship'" in llm.prompt
    assert "If the evidence is insufficient" in llm.prompt
    assert "relationship:depends_on" in llm.prompt
    assert "source_chunk:doc-1/chunk-1" in llm.prompt
    assert "\u5c40\u90e8\u68c0\u7d22" not in result.answer
