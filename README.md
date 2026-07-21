# LangChain RAG

一个面向学习、验证与后续集成的模块化 Python RAG 项目。当前已完成知识库、摄入、检索、Naive/Advanced/Modular/Agentic RAG、评测基线、Graph Index 基础设施以及 Vue 3 前端骨架。

## 当前状态

| 模块 | Owner | 状态 | 主要能力 |
| --- | --- | --- | --- |
| B1 Knowledge | B | 完成 | 共享模型、Repository、状态机、`KnowledgeService` |
| B2 Ingestion | B | 完成 | TXT/MD/PDF/DOCX、`TextNormalizer`、`Indexer`、回滚 |
| B3 Retrieval | B | 完成 | Embedding、VectorIndex、`kb_id` 隔离、Hybrid、Adapter |
| B4 Naive RAG | B | 完成 | 检索到生成、Citation、Trace、阈值、拒答 |
| B5 Advanced RAG | B | 完成 | Rewrite、Multi-Query、Hybrid、Rerank、Compression |
| B6 Vector Tool | B | 完成 | `vector_search`、Tool Result、Registry/Executor |
| C1 Data Contract | C | 完成 | `RAGRequest` / `RAGResult`、`RAGContext`、`TraceEvent`、5 种 `RAGMode` |
| C2 Strategy API | C | 完成 | `RAGStrategy` 统一接口、Protocol、DI、hooks |
| C3 Service Layer | C | 完成 | `RAGService.run()`、超时、取消、Registry、TraceRecorder |
| C4 Modular RAG | C | 完成 | 5 模块可配置流水线、Preset、对比、非法组合校验 |
| C5 Agentic RAG | C | 完成 | 有限状态 Agent、工具、降级、Citation |
| C6 Agent Eval | C | 完成 | Agent 评测数据集与 `AgentMetrics` |
| D1 Evaluation Dataset | D | 完成 | `datasets/evaluation/rag_eval_v1.jsonl`，24 条 JSONL 样例 |
| D2 Naive Baseline Eval | D | 完成 | JSONL 加载校验、Hit@K、MRR、Latency、mock/real runner、结构化报告 |
| D3 Graph Index | D | 完成 | Graph 模型、规则抽取、NetworkX 索引、内存仓储、社区发现、CommunityReport |
| Frontend Skeleton | A | 完成 | Vue 3 + Vite 前端骨架、Chat/Evaluation/Graph/Knowledge/Trace 页面 |

## 测试证据

| 检查 | 结果 |
| --- | --- |
| `python -m pytest tests/unit/graph -q` | 17 passed |
| `python -m pytest tests/unit/evaluation -q` | 15 passed |
| `python -m pytest -q` | 578 passed, 1 warning |

当前已知提示：Windows 环境下 pytest 退出后偶发临时目录清理 `WinError 5` 提示，但测试命令退出码为 0。

## 安装

```bash
# 基础依赖：RAG 核心、Agent、Graph Index
pip install -r requirements.txt

# 开发测试依赖：pytest + pytest-asyncio
pip install -r requirements-dev.txt

# 可选：Chroma 向量库
pip install -r requirements-chroma.txt

# 可选：BGE/HuggingFace Embedding
pip install -r requirements-bge.txt

# 可选：OpenAI Embedding / LLM
pip install -r requirements-openai.txt
```

完整测试需要安装 `requirements-dev.txt`。

## 结构

```text
src/
├── models/            # 共享数据契约
│   ├── knowledge.py   # KnowledgeBase / DocumentRecord / ChunkRecord
│   ├── rag.py         # RAGRequest / RAGResult / RAGContext / TraceEvent
│   ├── graph.py       # GraphEntity / GraphRelationship / GraphCommunity / CommunityReport
│   └── schemas.py     # RetrievalHit / Citation
├── ingestion/         # 文档摄入、规范化、切分、索引
├── retrieval/         # 向量检索、混合检索、重写、压缩、引用构建
├── knowledge/         # 知识库管理与内存仓储
├── rag/               # 统一 RAG 编排层与策略
│   ├── service.py     # RAGService
│   ├── registry.py    # RAGMode -> RAGStrategy
│   └── strategies/    # naive / advanced / modular / graph / agentic
├── agent/             # Agent 工作流、工具、路由、状态
├── graph/             # Graph Index 基础设施
│   ├── extractor.py   # 规则实体/关系抽取
│   ├── repository.py  # InMemoryGraphRepository
│   ├── index.py       # NetworkXGraphIndex
│   ├── community.py   # connected_components 社区发现
│   ├── reports.py     # CommunityReportBuilder
│   └── builder.py     # GraphIndexBuilder
├── evaluation/        # RAG 评测集、指标、runner、报告
└── api/               # FastAPI 应用与路由

frontend/              # Vue 3 + Vite 前端骨架
datasets/evaluation/   # 评测数据集与 baseline 报告
tests/                 # unit / integration / contract / e2e 测试
docs/                  # 架构、API 契约、进度与演示文档
```

## 常用命令

```bash
# 运行全部测试
python -m pytest -q

# 运行 D1/D2 评测专项测试
python -m pytest tests/unit/evaluation -q

# 运行 D3 Graph Index 专项测试
python -m pytest tests/unit/graph -q

# 离线生成 Naive baseline 报告
python -m src.evaluation.runner --dataset datasets/evaluation/rag_eval_v1.jsonl --mode naive --mock
```

## D3 Graph Index 说明

D3 提供可被后续 GraphRAG Local / Global Search 复用的索引基础，但尚未实现 D4/D5 检索能力。当前 MVP 使用确定性规则抽取，不依赖真实 API Key、外部 LLM、远程 Neo4j 或本地模型缓存。

追溯链路：

```text
ChunkRecord -> GraphSourceRef -> GraphEntity / GraphRelationship -> GraphCommunity -> CommunityReport
```

节点和边均保留 `document_id + chunk_id`，报告会聚合社区内实体与关系的源 Chunk 引用。

## 详细进度

参见 [docs/PROGRESS.md](docs/PROGRESS.md)。
