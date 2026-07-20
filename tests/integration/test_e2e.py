"""端到端统合测试：Loader -> Splitter -> Retrieval"""
import sys
import os
import tempfile
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
pylibs = os.path.join(os.environ.get("TEMP", "/tmp"), "pylibs")
if os.path.isdir(pylibs):
    sys.path.insert(0, pylibs)

from src.ingestion import LoaderFactory
from src.ingestion.models import DocumentRecord
from src.ingestion.splitter import split_document
from src.models.schemas import ChunkRecord
from src.retrieval import (
    HashEmbedder, InMemoryVectorIndex, Retriever,
    SimpleReranker, ContextCompressor, CitationBuilder,
    SimpleKnowledgeRepository,
)
from langchain_core.documents import Document


def document_to_chunk(doc: Document) -> ChunkRecord:
    """把 LangChain Document 转成 ChunkRecord。"""
    meta = doc.metadata or {}
    return ChunkRecord(
        id=meta.get("chunk_id", ""),
        document_id=meta.get("document_id", ""),
        kb_id=meta.get("kb_id", ""),
        text=doc.page_content,
        index=meta.get("chunk_index", 0),
        metadata=meta,
    )


def test_full_pipeline():
    """测试完整链路：加载 -> 切分 -> 索引 -> 检索 -> 重排 -> 压缩 -> 引用"""
    print("=" * 60)
    print("  端到端统合测试")
    print("=" * 60)

    # 1. 创建测试文档
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(
            "LangChain 是一个用于开发大语言模型应用的开源框架。\\n\\n"
            "它提供了链式调用、记忆管理和工具集成等核心功能。\\n\\n"
            "RAG 全称是检索增强生成，通过先检索相关文档再生成答案来提升质量。\\n\\n"
            "RAG 的核心流程包括：文档加载、文本切分、向量化、索引构建和检索生成。\\n\\n"
            "向量数据库如 Chroma 和 FAISS 用于存储文本的嵌入向量并支持相似度检索。\\n\\n"
            "Embedding 模型把文本转成高维向量，语义相近的文本向量距离也近。\\n\\n"
            "重排器用更精细的模型对初步检索结果重新排序，提高相关性。\\n\\n"
            "上下文压缩控制传给 LLM 的文本量，避免超出 Token 限制。\\n\\n"
            "引用生成让每个答案都能追溯到原始文档和具体段落。\\n\\n"
            "GraphRAG 把文档里的实体和关系建成图，适合回答关系类问题。"
        )
        temp_path = f.name

    try:
        # === 第 1 步：Loader 加载文档 ===
        print("\\n[1] Loader 加载文档...")
        doc = LoaderFactory.load(temp_path, kb_id="kb-test", document_id="doc-001")
        print(f"    文件名: {doc.filename}")
        print(f"    类型: {doc.file_type}")
        print(f"    文本长度: {len(doc.text)} 字符")
        print(f"    校验和: {doc.checksum[:16]}...")
        assert isinstance(doc, DocumentRecord)
        assert len(doc.text) > 100
        print("    -> PASS")

        # === 第 2 步：Splitter 切分 ===
        print("\\n[2] Splitter 切分文本...")
        chunks_docs = split_document(doc, chunk_size=200, chunk_overlap=30)
        print(f"    切分出 {len(chunks_docs)} 个 Chunk")
        assert len(chunks_docs) > 1
        assert all(isinstance(c, Document) for c in chunks_docs)
        # 检查元数据
        first = chunks_docs[0]
        assert "chunk_id" in first.metadata
        assert "chunk_index" in first.metadata
        assert "document_id" in first.metadata
        assert first.metadata["document_id"] == "doc-001"
        print(f"    第一个 Chunk ID: {first.metadata[chr(99)+chr(104)+chr(117)+chr(110)+chr(107)+chr(95)+chr(105)+chr(100)]}")
        print(f"    第一个 Chunk 长度: {len(first.page_content)} 字符")
        print(f"    总 Chunk 数: {first.metadata.get(chr(116)+chr(111)+chr(116)+chr(97)+chr(108)+chr(95)+chr(99)+chr(104)+chr(117)+chr(110)+chr(107)+chr(115), chr(63))}")
        print("    -> PASS")

        # === 第 3 步：转换并索引 ===
        print("\\n[3] 向量化并建索引...")
        chunks = [document_to_chunk(d) for d in chunks_docs]
        embedder = HashEmbedder(dim=256)
        index = InMemoryVectorIndex(embedder)
        repo = SimpleKnowledgeRepository(embedder, index)
        repo.add_chunks(chunks)
        print(f"    索引了 {repo.count()} 个 Chunk")
        assert repo.count() == len(chunks)
        print("    -> PASS")

        # === 第 4 步：检索 ===
        print("\\n[4] 向量检索...")
        query = "RAG 是什么"
        print(f"    查询: {query}")
        retriever = Retriever(embedder, index)
        hits = retriever.retrieve(query, top_k=5)
        print(f"    检索到 {len(hits)} 个结果")
        for h in hits:
            print(f"      #{h.rank} score={h.score:.4f} | {h.chunk.text[:40]}...")
        assert len(hits) > 0
        assert all(h.rank == i for i, h in enumerate(hits))
        print("    -> PASS")

        # === 第 5 步：重排 ===
        print("\\n[5] 重排...")
        reranker = SimpleReranker()
        reranked = reranker.rerank(query, hits, top_k=3)
        print(f"    重排后 {len(reranked)} 个结果")
        for h in reranked:
            print(f"      #{h.rank} score={h.score:.4f} retriever={h.retriever}")
        assert len(reranked) <= 3
        assert all(h.retriever == "rerank" for h in reranked)
        print("    -> PASS")

        # === 第 6 步：压缩 ===
        print("\\n[6] 上下文压缩...")
        compressor = ContextCompressor(max_tokens=100)
        compressed = compressor.compress(reranked)
        print(f"    压缩后 {len(compressed)} 个 Chunk")
        total_chars = sum(len(h.chunk.text) for h in compressed)
        print(f"    总字符数: {total_chars}")
        assert total_chars <= 100 * 2 + 10
        print("    -> PASS")

        # === 第 7 步：引用生成 ===
        print("\\n[7] 引用生成...")
        builder = CitationBuilder()
        citations = builder.build_citations(compressed)
        formatted = builder.format_citations(citations)
        print(f"    生成 {len(citations)} 条引用")
        print(formatted)
        assert len(citations) > 0
        assert all(c.document_id == "doc-001" for c in citations)
        print("    -> PASS")

        print("\\n" + "=" * 60)
        print("  全部测试通过!")
        print("=" * 60)
        print("\\n完整链路验证成功:")
        print("  Loader(加载) -> Splitter(切分) -> Embedder(向量化)")
        print("  -> VectorIndex(索引) -> Retriever(检索)")
        print("  -> Reranker(重排) -> Compressor(压缩)")
        print("  -> CitationBuilder(引用)")

    finally:
        os.unlink(temp_path)


def test_empty_document():
    """测试空文档不报错。"""
    print("\\n--- 附加测试：空文档 ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("")
        temp_path = f.name
    try:
        doc = LoaderFactory.load(temp_path, "kb-1", "doc-empty")
        chunks = split_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 0
        print("  空文档切分结果: 0 个 Chunk -> PASS")
    finally:
        os.unlink(temp_path)


def test_short_document():
    """测试短文档只产生一个 Chunk。"""
    print("\\n--- 附加测试：短文档 ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("这是一段很短的文字。")
        temp_path = f.name
    try:
        doc = LoaderFactory.load(temp_path, "kb-1", "doc-short")
        chunks = split_document(doc, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        print(f"  短文档切分结果: {len(chunks)} 个 Chunk -> PASS")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_full_pipeline()
    test_empty_document()
    test_short_document()
