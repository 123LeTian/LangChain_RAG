# LangChain RAG

一个面向学习、验证与后续集成的模块化 Python RAG 项目。当前已完成 B1–B6（Owner B：知识库、索引、检索、Naive/Advanced RAG、VectorTool）和 C1–C6（Owner C：RAG 统一框架、Modular RAG、Agentic RAG、Agent 评测）。

## 当前状态

| 模块 | Owner | 状态 | 主要能力 |
| --- | --- | --- | --- |
| B1 Knowledge | B | ✅ | 共享模型、Repository、状态机、`KnowledgeService` |
| B2 Ingestion | B | ✅ | TXT/MD/PDF/DOCX、`TextNormalizer`、`Indexer`、回滚 |
| B3 Retrieval | B | ✅ | Embedding、VectorIndex、`kb_id` 隔离、Hybrid、Adapter |
| B4 Naive RAG | B | ✅ | 检索→生成、Citation、Trace、阈值、拒答 |
| B5 Advanced RAG | B | ✅ | Rewrite、Multi-Query、Hybrid、Rerank、Compression |
| B6 Vector Tool | B | ✅ | `vector_search`、Tool Result、Registry/Executor |
| C1 Data Contract | C | ✅ | `RAGRequest`/`Result`、`RAGContext`、`TraceEvent`、5 种 `RAGMode` |
| C2 Strategy API | C | ✅ | `RAGStrategy` 统一接口、5 种 Protocol、DI、hooks |
| C3 Service Layer | C | ✅ | `RAGService.run()`、超时/取消、Registry、TraceRecorder |
| C4 Modular RAG | C | ✅ | 5 模块可配置流水线、Preset、一键对比、非法组合校验 |
| C5 Agentic RAG | C | ✅ | 有限状态 Agent（max 4 步）、4 工具、降级、Citation |
| C6 Agent Eval | C | ✅ | 15 题评测数据集、`AgentMetrics`（工具准确率 + 轨迹成功率） |

## 测试证据

| 检查 | 结果 |
| --- | --- |
| C 层测试 (`tests/unit/rag/` + `tests/unit/agent/`) | **413 passed** |
| B 层测试 (离线可用子集) | **131 passed** |
| 全量 | **544 passed** (2 skip — `pypdf`/`python-docx` 可选依赖) |

## 安装

```bash
# 基础依赖（RAG 核心 + Agent）
pip install -r requirements.txt

# 开发测试依赖（pytest + pytest-asyncio）
pip install -r requirements-dev.txt

# 可选：Chroma 向量库
pip install -r requirements-chroma.txt

# 可选：BGE/HuggingFace Embedding
pip install -r requirements-bge.txt

# 可选：OpenAI Embedding / LLM
pip install -r requirements-openai.txt
```

完整测试运行需要 `requirements-dev.txt`（提供 `pytest` 和 `pytest-asyncio`），否则 async 测试在干净环境中会失败。

## 架构

```
src/
├── models/          # 共享数据契约 (C + B)
│   ├── rag.py       #   RAGRequest/Result/Context/TraceEvent
│   └── knowledge.py #   KnowledgeBase/Document/Chunk
├── rag/             # RAG 编排层 (Owner: C)
│   ├── base.py      #   RAGStrategy 接口 + 5 个 Protocol
│   ├── service.py   #   统一入口 RAGService.run()
│   ├── registry.py  #   策略注册表 (RAGMode → RAGStrategy)
│   ├── trace.py     #   Span 追踪 + TraceRecorder
│   └── strategies/
│       ├── naive.py      # Naive RAG (Owner: B)
│       ├── advanced.py   # Advanced RAG (Owner: B)
│       ├── modular.py    # Modular RAG (Owner: C)
│       └── agentic.py    # Agentic RAG (Owner: C)
├── agent/           # Agent 工作流 (Owner: C)
│   ├── workflow.py  #   有限状态 AgentWorkflow (max 4 步)
│   ├── tools.py     #   4 个 Tool 接口 + Registry/Executor
│   ├── router.py    #   确定性的关键词工具路由器
│   └── state.py     #   AgentRunState
├── ingestion/       # 文档摄入 (Owner: B)
├── retrieval/       # 向量检索 (Owner: B)
├── knowledge/       # 知识库管理 (Owner: B)
├── storage/         # 存储层
├── graph/           # 知识图谱 (Owner: D, 未实现)
└── evaluation/      # 评测指标 (Owner: D, 未实现)
```

## 详细进度

见 [docs/PROGRESS.md](docs/PROGRESS.md)。
