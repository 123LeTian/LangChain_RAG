# LangChain RAG

一个面向学习、验证与后续集成的模块化 Python RAG 项目。当前已完成 B1–B6，覆盖知识库管理、文档加载与索引、向量检索、Naive RAG、可配置的 Advanced RAG，以及可供 Agent 调用的 `vector_search` 工具。

当前分支没有把 Agent Workflow、GraphRAG、完整 FastAPI 服务或前端标记为已完成。详细开发进度见 [docs/PROGRESS.md](docs/PROGRESS.md)。

## 当前状态

| 模块 | 状态 | 主要能力 |
| --- | --- | --- |
| B1 Knowledge | 已完成 | 共享模型、Repository、状态机、`KnowledgeService` |
| B2 Ingestion | 已完成 | TXT/Markdown/PDF/DOCX、`TextNormalizer`、`Indexer`、失败回滚与重索引 |
| B3 Retrieval | 已完成 | Embedding、VectorIndex、`kb_id` 隔离、`VectorRetrieverAdapter` |
| B4 Naive RAG | 已完成 | 检索、生成、Citation、Trace、无依据拒答 |
| B5 Advanced RAG | 已完成 | Rewrite、Multi-Query、Hybrid、Rerank、Compression |
| B6 Vector Tool | 已完成 | `vector_search`、Tool Result、Registry/Executor 适配 |

当前文档基线共收集并通过 199 项离线测试，其中包括 191 项单元测试、6 项集成测试和 2 项合同测试。

## 核心特性

- `KnowledgeBase`、`DocumentRecord`、`ChunkRecord` 共享模型，以及可替换的 Repository 抽象和线程安全 In-Memory 实现。
- 知识库和文档状态机，支持合法转换校验、失败摘要以及级联删除。
- 文档在同一知识库内按 `kb_id + checksum` 去重，不同知识库可接收相同内容。
- TXT、Markdown、PDF、DOCX 加载，以及 Unicode、换行、不可见字符和段落空白规范化。
- 稳定 Chunk ID、来源 metadata、索引失败补偿、文档重索引和向量清理。
- 确定性的 `HashEmbedder`，以及延迟初始化的 HuggingFace/BGE、OpenAI Embedding 适配器。
- `InMemoryVectorIndex` 和可选的 `ChromaVectorIndex`。
- 在候选排序和 Top-K 之前执行 `kb_id` 与 metadata 过滤，避免跨知识库结果污染。
- `VectorRetriever.search()` 统一返回带 score、rank、retriever 和来源 metadata 的 `RetrievalHit`。
- `NaiveRAGStrategy` 提供有界上下文、无依据拒答、可定位 Citation 和阶段 Trace。
- `AdvancedRAGStrategy` 提供可独立开关的 Rewrite、Multi-Query、Hybrid、Rerank、Compression 流程。该流程提供可配置能力，不代表未经评测的回答质量提升。
- `VectorSearchTool` 通过一次标准检索调用返回 JSON 安全结果，并适配 Tool Registry/Executor 合同。

## 项目结构

```text
src/
├── knowledge/          # Repository、状态机与 KnowledgeService
├── ingestion/          # Loader、Normalizer、Splitter 与 Indexer
├── retrieval/          # Embedding、VectorIndex、Retriever 与 Vector Tool
├── rag/
│   └── strategies/     # Naive 与 Advanced RAG Strategy
└── models/             # 共享 Knowledge 与 Retrieval 模型

tests/
├── unit/               # 单元与边界测试
├── integration/        # 离线端到端集成测试
└── contract/           # Tool 等跨模块合同测试
```

## 安装

仓库没有声明强制 Python 小版本，建议使用 Python 3.10+。

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\activate
```

macOS/Linux：

```bash
source .venv/bin/activate
```

安装基础依赖：

```bash
python -m pip install -r requirements.txt
```

基础依赖支持 In-Memory 检索、文档加载、文本切分和离线测试。可选集成按需安装：

```bash
# HuggingFace / BGE Embedding
python -m pip install -r requirements-bge.txt

# OpenAI 兼容 Embedding 和客户端
python -m pip install -r requirements-openai.txt

# Chroma VectorIndex
python -m pip install -r requirements-chroma.txt
```

- `requirements-bge.txt` 增加 `sentence-transformers`；首次使用未缓存模型时可能需要联网下载。
- `requirements-openai.txt` 增加 `langchain-openai` 与 `openai`；在线调用需要由运行环境提供凭据。
- `requirements-chroma.txt` 增加 `chromadb`；持久化路径和部署参数由使用方配置。
- 三个可选依赖文件都包含基础依赖，不需要重复单独安装 `requirements.txt`。

## 快速使用

以下示例采用当前代码的公开构造参数和方法名。请将示例文件路径替换为真实存在的本地文档。

### 创建知识库

```python
from src.knowledge import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryKnowledgeRepository,
    KnowledgeService,
)

service = KnowledgeService(
    knowledge_repository=InMemoryKnowledgeRepository(),
    document_repository=InMemoryDocumentRepository(),
    chunk_repository=InMemoryChunkRepository(),
)

knowledge_base = service.create_knowledge_base(
    owner_id="demo-user",
    name="产品文档",
    embedding_model="hash",
)
print(knowledge_base.id)
```

### 索引文档

`Indexer` 会依次执行加载、规范化、切分、Embedding、Chunk 持久化和向量写入；失败时执行补偿清理并更新状态。

```python
from src.ingestion import Indexer
from src.retrieval import HashEmbedder, InMemoryVectorIndex

embedder = HashEmbedder(dim=256)
vector_index = InMemoryVectorIndex(embedder)
indexer = Indexer(
    knowledge_service=service,
    embedder=embedder,
    vector_index=vector_index,
)

document = indexer.index_document(
    kb_id=knowledge_base.id,
    file_path="documents/example.md",
)
print(document.id, document.status.value)
```

支持的加载器由 `LoaderFactory` 按扩展名选择：`.txt`、`.md`/`.markdown`、`.pdf` 和 `.docx`。

### 向量检索

```python
from src.retrieval import VectorRetriever

retriever = VectorRetriever(embedder=embedder, index=vector_index)
hits = retriever.search(
    query="项目支持哪些文档格式？",
    kb_id=knowledge_base.id,
    top_k=5,
)

for hit in hits:
    print(hit.rank, hit.score, hit.chunk.id, hit.chunk.text)
```

`kb_id` 是标准检索接口的必填参数；`filters` 不能覆盖为其他知识库。

### Naive 与 Advanced RAG

两种 Strategy 的核心合同一致：

```python
from src.rag.strategies import AdvancedRAGStrategy, NaiveRAGStrategy

naive = NaiveRAGStrategy()
advanced = AdvancedRAGStrategy()

# request 与 context 必须遵循共享 RAGRequest/RAGContext 合同，
# 并由 context 注入 retriever、llm 和 trace_recorder 等依赖。
naive_result = await naive.run(request, context)
advanced_result = await advanced.run(request, context)
```

当前分支通过延迟合同工厂和 Adapter 与共享合同衔接。完整应用初始化应在共享 C 合同合入后进行跨分支集成复测。

### Vector Search Tool

```python
from dataclasses import dataclass
from typing import Any, Optional

from src.retrieval import VectorSearchTool


@dataclass
class LocalToolResult:
    """共享 C ToolResult 合入前用于本地调用的同形结果。"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tool_name: str = ""


tool = VectorSearchTool(retriever, result_factory=LocalToolResult)
result = await tool.execute(
    query="项目如何隔离知识库？",
    kb_id=knowledge_base.id,
    top_k=5,
)

if result.success:
    print(result.data)
else:
    print(result.error)
```

工具名称固定为 `vector_search`，同时提供同步 `invoke()` 和异步 `ainvoke()`/`execute()` 接口。共享 C `ToolResult` 合入后可省略 `result_factory`；当前分支的独立调用需要像示例或测试一样注入同形结果工厂。

## 测试

```bash
python -m compileall src
python -m pytest --collect-only -q
python -m pytest tests/integration -q
python -m pytest -q
```

当前测试完全离线：不需要真实 API Key，不访问在线模型，不下载 HuggingFace 模型，也不连接持久化 Chroma。测试覆盖 `unit`、`integration` 和 `contract`，本次文档基线为 `199 passed`。

## 设计原则

- **依赖注入**：Strategy、Indexer、Retriever 和 Tool 使用构造参数或 Context 注入依赖。
- **共享模型优先**：知识库、文档、Chunk 和检索命中使用统一模型，减少重复转换。
- **可选依赖延迟导入**：未使用 BGE、OpenAI 或 Chroma 时，不强制安装对应包或初始化客户端。
- **知识库严格隔离**：标准向量检索要求显式 `kb_id`，并在 Top-K 之前过滤。
- **失败补偿与状态可追踪**：索引、重索引和删除路径具备状态更新及补偿清理。
- **Citation 可定位**：引用由实际进入上下文的 `document_id + chunk_id` 和受控原文片段构建。
- **Trace 可观测**：RAG 阶段记录输入/输出摘要与非负耗时，异常使用结构化 ERROR 阶段。
- **离线可测试**：核心测试使用确定性组件和测试替身，不依赖网络服务。

## 当前限制

- 当前主要完成范围是 Step 0 与 B1–B6。
- Agent Router、Agent Workflow、Agentic RAG 尚未在当前完成范围内实现。
- Modular RAG、GraphRAG、完整 FastAPI 应用和前端尚未完成。
- 与 C 的共享合同已有 B 侧 Adapter，但在真实合并环境中仍需执行一次跨分支集成测试。
- `HashEmbedder` 适合确定性测试，不代表生产语义检索质量。
- HuggingFace/BGE、OpenAI 和 Chroma 属于可选在线或持久化集成，需要额外依赖、凭据或部署配置。
- Advanced RAG 消融测试验证流程、排序、上下文、Citation 与 Trace 差异，不代表回答质量必然提升。

## 开发进度

各阶段实现文件、测试证据、提交历史、风险与待办事项见 [docs/PROGRESS.md](docs/PROGRESS.md)。
