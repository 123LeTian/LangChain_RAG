# 开发进度

本文记录 `feature/c-orchestration-agent` → `main` 合并后的完整实现、测试证据与剩余工作。

## 当前快照

| 项目 | 结果 |
| --- | --- |
| 分支 | `main`（已合并 `feature/b-complete-validated` + `feature/c-orchestration-agent`） |
| 合并提交 | `141649a Merge branch 'feature/c-orchestration-agent'` |
| C 层测试收集 | 413（RAG + Agent） |
| C 层全量通过 | 413 |
| B 层测试收集 | 131（离线可用子集） |
| B 层全量通过 | 131 |
| 全量收集 | 546 |
| 全量通过 | 546（已安装 `requirements-dev.txt`，包含 `pytest` 与 `pytest-asyncio`） |

## 总体进度

| 阶段 | 状态 | 关键交付 | 主要测试 |
| --- | --- | --- | --- |
| Step 0 | ✅ 完成 | Embedder 清理、可选依赖与安装说明 | Embedding 导入、延迟初始化、兼容性测试 |
| B1 | ✅ 完成 | Knowledge Repository、状态机、`KnowledgeService` | CRUD、状态转换、级联删除、隔离与并发 |
| B2 | ✅ 完成 | `TextNormalizer`、`Indexer`、四类文档处理 | TXT/MD/PDF/DOCX、回滚、去重、重索引 |
| B3 | ✅ 完成 | `kb_id` 隔离、VectorIndex、Retriever Adapter | Top-K 前隔离、Hybrid、Chroma where、合同转换 |
| B4 | ✅ 完成 | `NaiveRAGStrategy` | Citation、Trace、阈值、上下文预算、拒答 |
| B5 | ✅ 完成 | `AdvancedRAGStrategy` | Rewrite、Multi-Query、Hybrid、Rerank、Compression |
| B6 | ✅ 完成 | `VectorSearchTool` | Tool 合同、Registry/Executor、离线隔离集成 |
| C1 | ✅ 完成 | 共享 RAG 数据契约 (`src/models/rag.py`) | RAGRequest/Result/Query/Response、TraceEvent、RAGMode |
| C2 | ✅ 完成 | `RAGStrategy` 统一接口 + `RAGStrategyBase` | Protocol 注入、hooks、trace_recorder、Context DI |
| C3 | ✅ 完成 | `RAGService` + `RAGStrategyRegistry` + `TraceRecorder` | 超时、取消、`run_safe`、策略注册/查找、Span 追踪 |
| C4 | ✅ 完成 | `ModularRAGStrategy`（可配置流水线） | 5 模块开关、非法组合校验、Preset、一键对比 |
| C5 | ✅ 完成 | `AgenticRAGStrategy` + `AgentWorkflow`（有限状态） | 4 工具、固定步数、降级、Citation、AgentMetrics |
| C6 | ✅ 完成 | Agent 评测样例（15 题数据集 + 可统计指标） | 工具选择准确率、轨迹成功率、`AgentMetrics.report()` |

## 已完成能力

### B1：Knowledge 契约、Repository、状态机与 Service

- **实现文件**：`src/models/knowledge.py`、`src/knowledge/repositories.py`、`src/knowledge/state_machine.py`、`src/knowledge/service.py`。
- **核心接口**：`KnowledgeRepository`、`DocumentRepository`、`ChunkRepository`、三个 In-Memory 实现，以及 `KnowledgeService` 的知识库和文档生命周期方法。
- **测试证据**：`tests/unit/knowledge/test_repositories.py`、`test_state_machine.py`、`test_service.py`。
- **验收结论**：CRUD、合法状态转换、同库 checksum 去重、级联删除、向量清理失败保护和线程安全写入均已覆盖。

### B2：Normalizer、Indexer 与四类文档

- **实现文件**：`src/ingestion/normalizer.py`、`loaders.py`、`splitter.py`、`indexer.py`。
- **测试证据**：`tests/unit/ingestion/test_normalizer.py`、`test_indexer.py`。
- **验收结论**：TXT/MD/PDF/DOCX 处理、checksum、稳定 Chunk ID、失败回滚、重索引已覆盖。

### B3：向量检索与知识库隔离

- **实现文件**：`src/retrieval/embeddings.py`、`vector_index.py`、`retriever.py`、`adapter.py`、`hybrid.py`。
- **验收结论**：`kb_id` 隔离、Hybrid、Top-K 过滤、Adapter 满足 C 的 `RetrieverProtocol`。

### B4：Naive RAG

- **实现文件**：`src/rag/strategies/naive.py`、`src/rag/naive_support.py`、`src/rag/c_contract.py`。
- **验收结论**：检索→生成、Citation、Trace、阈值、上下文预算、拒答全覆盖。

### B5：Advanced RAG

- **实现文件**：`src/rag/strategies/advanced.py`、`src/retrieval/query_rewriter.py`、`multi_query.py`、`hybrid.py`、`reranker.py`、`compressor.py`。
- **验收结论**：5 模块可独立开关、异常降级、去重、重排追踪、压缩回退全覆盖。

### B6：Vector Search Tool

- **实现文件**：`src/retrieval/vector_tool.py`。
- **验收结论**：`vector_search` 参数校验、JSON 安全结果、Registry/Executor 调用、跨 KB 隔离。

---

### C1：共享 RAG 数据契约

- **实现文件**：`src/models/rag.py`（~560 行）。
- **核心类型**：`RAGRequest`、`RAGResult`、`RAGQuery`、`RAGResponse`、`RAGChunk`、`RAGCitation`、`RAGSource`、`RAGContext`、`TraceEvent`、`RAGMode`、`TraceStage`。
- **测试证据**：`tests/unit/rag/test_base.py`（`TestRAGStrategy` 部分）。
- **验收结论**：所有 Pydantic 模型通过验证，DI 字段排除序列化，`RAGRequest.to_rag_query()` 转换正确。

### C2：RAGStrategy 统一接口

- **实现文件**：`src/rag/base.py`（~445 行）。
- **核心接口**：5 个 Protocol（`RetrieverProtocol`、`GeneratorProtocol`、`GraphRetrieverProtocol`、`RerankerProtocol`、`EmbedderProtocol`）、`RAGStrategyBase`、`RAGStrategy`。
- **测试证据**：`tests/unit/rag/test_base.py`。
- **验收结论**：策略通过 `context` 注入依赖，框架不创建 LLM/Retriever 实例；hooks（`pre_run`、`post_run`、`on_error`）+ `_record()` trace 辅助全覆盖。

### C3：RAGService + Registry + Trace

- **实现文件**：`src/rag/service.py`（~330 行）、`src/rag/registry.py`（~250 行）、`src/rag/trace.py`（~570 行）。
- **核心接口**：`RAGService.run(request, timeout, cancel_event)`、`run_safe()`、`RAGStrategyRegistry`、`TraceRecorder`、`TraceStore`。
- **测试证据**：`tests/unit/rag/test_service.py`、`tests/unit/rag/test_registry.py`、`tests/unit/rag/test_trace.py`。
- **验收结论**：超时/取消、`run_safe` 永不抛异常、策略 CRUD、Span 追踪、SSE 订阅全覆盖。

### C4：Modular RAG（可配置流水线）

- **实现文件**：`src/rag/strategies/modular.py`（~680 行）。
- **核心接口**：`ModuleConfig`（5 开关）、`ModularRAGStrategy.run()`、`validate_module_config()`、`save_preset()`/`load_preset()`/`apply_preset()`、`compare()`/`compare_presets()`。
- **测试证据**：`tests/unit/rag/test_modular.py`（58 个测试，含 29 个验收标准测试）。
- **验收标准**：
  1. ✅ 5 模块独立开关
  2. ✅ 非法组合校验（rerank 需 retrieve）
  3. ✅ 每次运行保存配置到 `context.metadata["pipeline_config"]`
  4. ✅ 相同问题一键对比两套配置（`compare()` + `compare_presets()`）
  5. ✅ Trace 顺序与实际启用模块一致

### C5：Agentic RAG（有限状态工作流）

- **实现文件**：`src/rag/strategies/agentic.py`（~220 行）、`src/agent/workflow.py`（~800 行）、`src/agent/tools.py`（~460 行）、`src/agent/router.py`（~360 行）、`src/agent/state.py`（~260 行）。
- **核心接口**：`AgenticRAGStrategy.run(request, context)` → `RAGResult`（新接口，接入 `RAGService`）、`AgentWorkflow`（有限状态，4 步）、`AgentRouter`（关键词路由）、`VectorSearchTool`/`GraphSearchTool`/`DocumentSummaryTool`/`AnswerVerifyTool`。
- **测试证据**：`tests/unit/agent/test_agentic.py`（76 个测试，含 44 个验收标准测试）、`tests/unit/agent/test_router.py`、`tests/unit/agent/test_tools.py`、`tests/unit/agent/test_workflow.py`、`tests/unit/agent/test_state.py`。
- **验收标准**：
  1. ✅ 三类问题选择正确工具（fact→vector, entity→graph, summary→doc_summary）
  2. ✅ 最多 4 步，防止无限循环
  3. ✅ 每次工具调用记录名称、参数摘要、结果摘要、耗时和错误（`state.tool_calls`）
  4. ✅ graph_search 失败 → 降级到 vector_search；全部失败 → 明确标记
  5. ✅ 最终回答包含引用（`state.citations`）
  6. ✅ 单独统计工具选择准确率和轨迹成功率（`AgentMetrics`）

### C6：Agent 评测样例

- **数据文件**：`datasets/evaluation/agent_test_questions.json`（15 题，覆盖 fact/entity/summary/complex 四类）。
- **评测指标**：`AgentMetrics.tool_selection_accuracy`、`AgentMetrics.trajectory_success_rate`。
- **测试证据**：`tests/unit/rag/test_service.py`（`TestAgenticEvalDataset`，4 个数据集驱动测试）。

---

## 提交历史

| 阶段 | 关键提交 | 信息 |
| --- | --- | --- |
| B1-B6 | 多个 | B 层完整实现（见 `feature/b-complete-validated` 分支） |
| C1 | `3930e39` | `feat(C3): implement unified RAG orchestration service` |
| C2 | `bfc06e5` | `feat: implement TraceRecorder for stage-level RAG pipeline tracing` |
| C3 | `05bc7a5` | `feat: implement configurable Modular RAG pipeline with validation` |
| C4 | `38c84de` | `feat: implement Agentic RAG with finite-state agent workflow` |
| C5 | `3070e8c` | `feat: implement Agent error degradation with sanitized error handling` |
| C6 | `983c90a` | `feat: implement Modular RAG acceptance criteria 3 & 4` |
| C7 | `746ecb3` | `feat: implement Agentic RAG acceptance criteria 3, 5, 6` |
| C8 | `9f8fce4` | `fix: resolve 4 Agentic RAG integration issues` |
| 合并 | `d489213` | `Merge feature/c-orchestration-agent into main` |

## 已知交付说明

### 1. 测试依赖由 `requirements-dev.txt` 管理

`requirements.txt` 包含 RAG 与 Agent 运行时依赖；开发测试依赖单独放在 `requirements-dev.txt`，其中包含 `pytest` 和 `pytest-asyncio`。运行完整测试前安装：

```bash
pip install -r requirements-dev.txt
```

### 2. Agentic RAG 的 GraphSearchTool 等待 D 提供真实后端

- C 已完整实现：`GraphSearchTool`（`src/agent/tools.py`）包装 `GraphRetrieverProtocol`，与 `AgentWorkflow` 的降级逻辑（graph → vector）全部就绪。
- 当前状态：无真实 `GraphRetrieverProtocol` 实现时，graph_search 必然失败并降级到 vector_search，降级路径有测试覆盖。
- 依赖 D 交付：`src/graph/` 模块（`graph/retriever.py` 等）和 `GraphRAGStrategy`（`src/rag/strategies/graph_rag.py`）。
- 不影响验收：C 的 Agentic 已满足 `vector_search` + `document_summary` + `answer_verify` 三类工具的可验收运行；graph 通道是"可接 D"而非"已完整跑通"。

## 质量证据

| 检查 | 实际结果 |
| --- | --- |
| `python -m pytest tests/unit/rag/ tests/unit/agent/ -q` | **413 passed** (C 层) |
| `python -m pytest tests/ -q` | **546 passed** (全量，已安装 `requirements-dev.txt` 的 `pytest-asyncio`) |
| 干净环境测试 | 安装 `requirements-dev.txt` 后全量通过 |
| `langgraph` 依赖 | 已在 `requirements.txt` 声明 `langgraph>=0.2.0`；源码延迟导入保护 |
| GraphRAG 真实后端 | 等待 Owner D 提供 `GraphRetrieverProtocol` 实现和 `GraphRAGStrategy` |

## 待办事项

### 后续集成

- `pypdf` / `python-docx` 已纳入基础依赖；PDF/DOCX 相关 B 层测试在干净开发环境中通过。
- 如需要 Chroma/OpenAI/BGE 集成测试，安装对应 `requirements-*.txt`。

### 未实现功能

- GraphRAG（Owner: D）。
- 完整 FastAPI 应用（Owner: A）。
- 前端。
- 正式质量评估基准。

### 非阻塞环境问题

- Windows 上旧 pytest 临时目录可能在退出清理阶段产生权限提示；当前测试命令退出码仍为 0。
