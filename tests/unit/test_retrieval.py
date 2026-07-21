"""B2 检索模块单元测试"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import ChunkRecord, RetrievalHit, Citation
from src.retrieval import (
    HashEmbedder, InMemoryVectorIndex, Retriever,
    SimpleReranker, ContextCompressor, CitationBuilder,
    KeywordSearcher, HybridRetriever, SimpleKnowledgeRepository,
)


def make_chunks():
    return [
        ChunkRecord(id="c1", document_id="d1", kb_id="kb1", text="LangChain 是一个用于开发 LLM 应用的框架。", index=0, metadata={"filename": "intro.pdf", "page": 1}),
        ChunkRecord(id="c2", document_id="d1", kb_id="kb1", text="RAG 是检索增强生成的缩写，结合了检索和生成。", index=1, metadata={"filename": "intro.pdf", "page": 2}),
        ChunkRecord(id="c3", document_id="d2", kb_id="kb1", text="向量数据库用于存储和检索文本的嵌入向量。", index=0, metadata={"filename": "vector.pdf", "page": 1}),
    ]


def test_embedder():
    emb = HashEmbedder(dim=128)
    vecs = emb.embed_texts(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128
    print("test_embedder passed")


def test_vector_index():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    chunks = make_chunks()
    index.add_chunks(chunks)
    assert index.count() == 3
    qvec = emb.embed_query("LangChain 框架")
    hits = index.search(qvec, kb_id="kb1", top_k=2)
    assert len(hits) == 2
    assert hits[0].rank == 0
    print("test_vector_index passed")


def test_retriever():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    index.add_chunks(make_chunks())
    retriever = Retriever(emb, index)
    hits = retriever.retrieve("RAG 是什么", top_k=2)
    assert len(hits) == 2
    assert all(isinstance(h, RetrievalHit) for h in hits)
    print("test_retriever passed")


def test_reranker():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    index.add_chunks(make_chunks())
    retriever = Retriever(emb, index)
    hits = retriever.retrieve("RAG", top_k=3)
    reranker = SimpleReranker()
    reranked = reranker.rerank("RAG", hits, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].retriever == "rerank"
    print("test_reranker passed")


def test_compressor():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    index.add_chunks(make_chunks())
    retriever = Retriever(emb, index)
    hits = retriever.retrieve("RAG", top_k=3)
    comp = ContextCompressor(max_tokens=50)
    compressed = comp.compress(hits)
    total_chars = sum(len(h.chunk.text) for h in compressed)
    assert total_chars <= 50 * 2 + 10
    print("test_compressor passed")


def test_citation_builder():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    index.add_chunks(make_chunks())
    retriever = Retriever(emb, index)
    hits = retriever.retrieve("RAG", top_k=2)
    builder = CitationBuilder()
    citations = builder.build_citations(hits)
    assert len(citations) == 2
    assert all(isinstance(c, Citation) for c in citations)
    assert citations[0].filename == "intro.pdf"
    formatted = builder.format_citations(citations)
    assert "[1]" in formatted
    print("test_citation_builder passed")


def test_keyword_searcher():
    ks = KeywordSearcher()
    ks.add_chunks(make_chunks())
    hits = ks.search("RAG 检索", top_k=2)
    assert len(hits) > 0
    assert hits[0].retriever == "keyword"
    print("test_keyword_searcher passed")


def test_hybrid_retriever():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    hybrid = HybridRetriever(emb, index, alpha=0.6)
    hybrid.add_chunks(make_chunks())
    hits = hybrid.retrieve("RAG 检索", top_k=2)
    assert len(hits) == 2
    assert all(h.retriever == "hybrid" for h in hits)
    print("test_hybrid_retriever passed")


def test_knowledge_repository():
    emb = HashEmbedder(dim=128)
    index = InMemoryVectorIndex(emb)
    repo = SimpleKnowledgeRepository(emb, index)
    repo.add_chunks(make_chunks())
    assert repo.count() == 3
    chunk = repo.get_chunk("c1")
    assert chunk is not None
    assert chunk.text.startswith("LangChain")
    hits = repo.search("RAG", top_k=2)
    assert len(hits) == 2
    repo.clear()
    assert repo.count() == 0
    print("test_knowledge_repository passed")


if __name__ == "__main__":
    test_embedder()
    test_vector_index()
    test_retriever()
    test_reranker()
    test_compressor()
    test_citation_builder()
    test_keyword_searcher()
    test_hybrid_retriever()
    test_knowledge_repository()
    print("\nAll tests passed!")
