"""RAG + DeepSeek：向量检索(bge-small-zh) + LLM 生成回答"""
import sys
import os
from pathlib import Path

_temp_pylibs = os.path.join(os.environ.get("TEMP", ""), "pylibs")
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(_temp_pylibs)]

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_env(env_path):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
load_env(str(PROJECT_ROOT / ".env"))

from src.ingestion import LoaderFactory
from src.ingestion.splitter import split_documents
from src.models.schemas import ChunkRecord
from src.rag.naive_support import build_naive_prompt
from src.retrieval import (
    HuggingFaceEmbedder, InMemoryVectorIndex, Retriever,
    SimpleReranker, ContextCompressor, CitationBuilder,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

DOCS_DIR = PROJECT_ROOT / "documents"


def create_llm():
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.1,
        max_tokens=1000,
    )


def document_to_chunk(doc):
    meta = doc.metadata or {}
    return ChunkRecord(
        id=meta.get("chunk_id", ""),
        document_id=meta.get("document_id", ""),
        kb_id=meta.get("kb_id", ""),
        text=doc.page_content,
        index=meta.get("chunk_index", 0),
        metadata=meta,
    )


def load_documents_from_dir(docs_dir):
    supported = [".txt", ".md", ".markdown", ".pdf", ".docx"]
    files = []
    for ext in supported:
        files.extend(Path(docs_dir).glob(f"**/*{ext}"))
    if not files:
        print("  [警告] documents/ 文件夹是空的！")
        return []
    documents = []
    for i, fpath in enumerate(files):
        try:
            doc = LoaderFactory.load(str(fpath), kb_id="kb-1", document_id=f"doc-{i:03d}")
            documents.append(doc)
            print(f"  加载: {fpath.name} ({len(doc.text)} 字符)")
        except Exception as e:
            print(f"  [失败] {fpath.name}: {e}")
    return documents


def rag_query(llm, retriever, reranker, compressor, citation_builder, query, top_k=5):
    # 1. 向量检索
    hits = retriever.retrieve(query, top_k=top_k)
    print(f"  检索到 {len(hits)} 个结果")
    for h in hits:
        preview = h.chunk.text[:50].replace("\n", " ")
        print(f"    #{h.rank} score={h.score:.4f} | {preview}...")

    # 2. 重排
    reranked = reranker.rerank(query, hits, top_k=3)

    # 3. 压缩
    compressed = compressor.compress(reranked)

    # 4. 组装上下文
    context = "\n\n---\n\n".join([
        f"[片段{j+1}] (来源: {h.chunk.metadata.get('source', h.chunk.metadata.get('file_name', '未知'))}, chunk_{h.chunk.index})\n{h.chunk.text}"
        for j, h in enumerate(compressed)
    ])

    # 5. 生成引用
    citations = citation_builder.build_citations(compressed)
    citation_text = citation_builder.format_citations(citations)

    # 6. LLM 生成
    prompt = build_naive_prompt(query, context)

    print("  正在调用 DeepSeek 生成回答...")
    response = llm.invoke(prompt)
    answer = response.content
    return answer, context, citation_text, compressed


def main():
    print("=" * 60)
    print("  RAG + DeepSeek (向量检索 + LLM)")
    print("  Embedding: BAAI/bge-small-zh-v1.5")
    print("  LLM: deepseek-chat")
    print("=" * 60)

    DOCS_DIR.mkdir(exist_ok=True)

    # 1. 加载文档
    print("\n[1] 从 documents/ 加载文档...")
    print("  目录:", DOCS_DIR)
    documents = load_documents_from_dir(DOCS_DIR)
    if not documents:
        print("\n  没有可用的文档，退出。")
        print("  请把 .txt/.md/.pdf/.docx 文件放到 documents/ 文件夹后重试。")
        return

    # 2. 切分
    print("\n[2] 切分文本...")
    chunks_docs = split_documents(documents, chunk_size=500, chunk_overlap=100)
    print("  共切分出", len(chunks_docs), "个 Chunk")

    # 3. 初始化 Embedder（语义向量模型）
    print("\n[3] 初始化语义 Embedding 模型...")
    embedder = HuggingFaceEmbedder(model_name="BAAI/bge-small-zh-v1.5")

    # 4. 向量化并建索引
    print("\n[4] 向量化并建立索引...")
    chunks = [document_to_chunk(d) for d in chunks_docs]
    index = InMemoryVectorIndex(embedder)
    index.add_chunks(chunks)
    print("  索引了", index.count(), "个 Chunk")

    # 5. 初始化检索组件
    retriever = Retriever(embedder, index)
    reranker = SimpleReranker()
    compressor = ContextCompressor(max_tokens=1000)
    citation_builder = CitationBuilder()

    # 6. 初始化 LLM
    print("\n[5] 初始化 DeepSeek LLM...")
    llm = create_llm()
    print("  模型:", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    # 7. 问答循环
    print("\n" + "=" * 60)
    print("  问答系统已就绪！输入 quit 退出")
    print("  (向量语义检索 + DeepSeek 生成)")
    print("=" * 60)

    while True:
        print()
        query = input("你的问题> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not query:
            continue
        print("\n处理中...")
        try:
            answer, context, citation_text, compressed = rag_query(
                llm, retriever, reranker, compressor, citation_builder, query
            )
            print("\n" + "-" * 60)
            print("回答：")
            print(answer)
            print("-" * 60)
            print("引用来源：")
            print(citation_text)
            print("-" * 60)
            print("使用的 Chunk 数:", len(compressed))
            print("上下文长度:", len(context), "字符")
        except Exception as e:
            print("错误:", e)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
