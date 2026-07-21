# LangChain RAG

一个面向学习、验证与答辩演示的多范式 RAG 实验平台。项目在统一数据契约、统一知识库入口和统一评测集上，实现了 Naive / Advanced / Modular / Graph / Agentic 五种 RAG 模式，并提供 FastAPI 后端与 Vue 3 前端骨架。

## 当前结论

核心算法模块、GraphRAG MVP、Agentic RAG、五模式离线评测、FastAPI 集成和 Vue 页面骨架已经完成。当前 `main` 还接入了真实查询链路：上传文档保存到 `documents/`，查询时使用 DeepSeek 做 Query Rewrite 和生成，使用 HuggingFace Embedding 做混合检索、重排、压缩和引用生成。

需要注意：离线最终评测仍使用 `mock_*` runner 验证链路、报告结构和指标计算；接入真实知识库、Embedding 与 LLM 环境后，应重新运行最终评测再作为线上质量结论。

## 功能状态

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
| D4 GraphRAG | D | 完成 | Local entity-relation search、Global community-report search、Citation、Trace、GraphRAGStrategy |
| D5 Graph Tool | D | 完成 | `graph_search` 标准 Tool 接口、参数校验、ToolResult Adapter、Registry/Executor 合同测试 |
| D6 Final Evaluation | D | 完成 | 五模式 runner、Answer/System/Graph/Agent 指标、JSON/Markdown/CSV 报告、回归检查 |
| A Frontend/API | A | 基本完成 | FastAPI 路由、真实 Chat/RAG 查询、知识库上传持久化、Vue 3 五页面骨架 |

## 已实现的真实查询链路

当前 FastAPI 的 `/api/rag/query` 与 `/api/rag/query/stream` 已接入 `RealRAGService`：

```text
上传文档 -> 保存到 documents/ -> 加载 TXT/MD/PDF/DOCX
-> 文档切分 -> HuggingFace Embedding -> Hybrid Retrieval
-> DeepSeek Query Rewrite -> 合并去重 -> Rerank
-> Context Compression -> DeepSeek 生成 -> Citation + Trace
```

简单聊天类问题会绕过文档检索，直接调用 DeepSeek；文档问答会走完整 RAG pipeline。

## 接口边界

| 接口/页面 | 当前数据来源 |
| --- | --- |
| Chat / `/api/rag/query` | 真实 DeepSeek + 文档检索链路 |
| Chat Stream / `/api/rag/query/stream` | SSE 事件，返回 trace / chunk / done / error |
| Knowledge / `/api/knowledge-bases` | API 内存态 + `data/knowledge_bases.json` 持久化 |
| Graph / `/api/graphs/{kb_id}` | 前端展示用 Mock 图数据；算法层 GraphRAG 已在 `src/graph` 和 `src/rag/strategies/graph_rag.py` 完成 |
| Trace / `/api/traces/{trace_id}` | Mock trace 查询；真实 query 响应中已返回 trace |
| Evaluation / `/api/evaluations/*` | 前端展示用 Mock 评测结果；离线 D6 报告已生成 |

## 安装

基础算法和测试依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

运行真实后端问答还需要 FastAPI、SSE、上传和模型相关依赖：

```bash
pip install fastapi uvicorn sse-starlette python-multipart
pip install -r requirements-openai.txt
pip install -r requirements-bge.txt
```

可选依赖：

```bash
# Chroma 向量库
pip install -r requirements-chroma.txt
```

## 环境变量

真实 Chat/RAG 查询使用 OpenAI 兼容接口调用 DeepSeek。可以在项目根目录创建 `.env`：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

`.env`、`data/`、运行数据库和模型缓存不应提交到仓库。

## 启动

后端：

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

默认前端地址是 `http://localhost:5173`，后端健康检查是 `http://127.0.0.1:8000/health`。

## 使用流程

1. 在前端知识库页面上传 `.txt`、`.md`、`.pdf` 或 `.docx` 文件，文件会保存到项目根目录的 `documents/`。
2. 打开 Chat 页面，选择 RAG 模式并提问。
3. 首次文档问答会加载文档、切分、生成向量并建立内存索引，可能需要等待一段时间。
4. 查询结果会返回答案、引用、检索命中和 Trace。
5. Graph 与 Evaluation 页面当前用于展示结构和演示数据；算法层真实能力可通过测试和离线 runner 验证。

## 测试证据

| 检查 | 结果 |
| --- | --- |
| `python -m pytest tests/unit/graph -q` | 36 passed, 1 warning |
| `python -m pytest tests/unit/evaluation -q` | 34 passed |
| `python -m pytest tests/unit/rag/test_graph_rag_strategy.py -q` | 6 passed, 1 warning |
| `python -m pytest tests/contract/test_graph_tool_contract.py -q` | 4 passed, 1 warning |
| `python -m pytest -q` | 628 passed, 1 warning |

当前已知提示：Windows 环境下 pytest 退出后偶发临时目录清理 `WinError 5` 提示，但测试命令退出码为 0。

## 常用命令

```bash
# 运行全部测试
python -m pytest -q

# 运行 D1/D2 评测专项测试
python -m pytest tests/unit/evaluation -q

# 运行 D3 Graph Index 专项测试
python -m pytest tests/unit/graph -q

# 运行 D4 GraphRAG 专项测试
python -m pytest tests/unit/rag/test_graph_rag_strategy.py -q

# 运行 D5 Graph Tool 合同测试
python -m pytest tests/contract/test_graph_tool_contract.py -q

# 离线生成 Naive baseline 报告
python -m src.evaluation.runner --dataset datasets/evaluation/rag_eval_v1.jsonl --mode naive --mock

# 离线生成 D6 五模式最终评测报告
python -m src.evaluation.runner --dataset datasets/evaluation/rag_eval_v1.jsonl --modes naive advanced modular graph agentic --mock --output datasets/evaluation/reports/final_evaluation.json --markdown docs/evaluation-report.md --csv datasets/evaluation/reports/final_evaluation.csv
```

## 项目结构

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
datasets/evaluation/   # 评测数据集与 baseline/final 报告
tests/                 # unit / integration / contract / e2e 测试
docs/                  # 架构、API 契约、进度与演示文档
```

## GraphRAG 与 Graph Tool

当前 GraphRAG MVP 使用确定性规则抽取、NetworkX 图索引和内存图仓储，不依赖真实 API Key、外部 LLM、远程 Neo4j 或本地模型缓存。

追溯链路：

```text
ChunkRecord -> GraphSourceRef -> GraphEntity / GraphRelationship -> GraphCommunity -> CommunityReport
```

已完成能力：

- `GraphRetriever.local_search()`：面向实体、关系和一跳邻居的局部图检索。
- `GraphRetriever.global_search()`：面向 CommunityReport 的全局主题检索。
- `GraphRAGStrategy`：接入统一 `RAGStrategy`，返回 `RAGResult`、Citation、Trace 和 warning。
- `GraphSearchTool`：提供标准 `graph_search(query, kb_id, scope, top_k)` Tool 接口，兼容 C 的 `ToolRegistry` / `ToolExecutor`。

## 最终评测报告

D6 已生成五模式离线最终评测报告：

- JSON：`datasets/evaluation/reports/final_evaluation.json`
- Markdown：`docs/evaluation-report.md`
- CSV：`datasets/evaluation/reports/final_evaluation.csv`

当前最终评测使用 `mock_*` runner，用于验证五模式评测链路、报告结构、指标计算和回归检查。接入真实 `RAGService`、知识库和模型环境后，应重新运行最终评测。

## 详细进度

参见 [docs/PROGRESS.md](docs/PROGRESS.md)。
