# 开发进度

本文记录 `feature/b-complete-validated` 分支在文档编写基线上的真实实现、测试证据与剩余工作。状态来自源码、Git 历史和本地离线测试，不把尚未交付的模块标记为完成。

## 当前快照

| 项目 | 结果 |
| --- | --- |
| 分支 | `feature/b-complete-validated` |
| 文档编写基线 HEAD | `8d5c611f2dd2b823a4cbe7847b3b04c0acf0bddd` |
| 测试收集数量 | 199 |
| 全量通过数量 | 199 |
| 单元测试 | 191 passed |
| 集成测试 | 6 passed |
| 合同测试 | 2 passed |
| 工作区状态 | 文档修改前干净，且与远端分支同步 |

测试均为离线执行。Windows 退出清理阶段出现过旧 pytest 临时目录 `pytest-269` 的 `PermissionError`，但相关 pytest 命令退出码均为 0，不影响测试结论。

## 总体进度

| 阶段 | 状态 | 关键交付 | 主要测试 |
| --- | --- | --- | --- |
| Step 0 | 完成 | Embedder 清理、可选依赖与安装说明 | Embedding 导入、延迟初始化、兼容性测试 |
| B1 | 完成 | Knowledge Repository、状态机、`KnowledgeService` | CRUD、状态转换、级联删除、隔离与并发 |
| B2 | 完成 | `TextNormalizer`、`Indexer`、四类文档处理 | TXT/MD/PDF/DOCX、回滚、去重、重索引 |
| B3 | 完成 | `kb_id` 隔离、VectorIndex、Retriever Adapter | Top-K 前隔离、Hybrid、Chroma where、合同转换 |
| B4 | 完成 | `NaiveRAGStrategy` | Citation、Trace、阈值、上下文预算、拒答 |
| B5 | 完成 | `AdvancedRAGStrategy` | Rewrite、Multi-Query、Hybrid、Rerank、Compression |
| B6 | 完成 | `VectorSearchTool` | Tool 合同、Registry/Executor、离线隔离集成 |

## 已完成能力

### B1：Knowledge 契约、Repository、状态机与 Service

- **实现文件**：`src/models/knowledge.py`、`src/knowledge/repositories.py`、`src/knowledge/state_machine.py`、`src/knowledge/service.py`。
- **核心接口**：`KnowledgeRepository`、`DocumentRepository`、`ChunkRepository`、三个 In-Memory 实现，以及 `KnowledgeService` 的知识库和文档生命周期方法。
- **测试证据**：`tests/unit/knowledge/test_repositories.py`、`test_state_machine.py`、`test_service.py`。
- **验收结论**：共享模型、CRUD、合法状态转换、同库 checksum 去重、级联删除、向量清理失败保护和线程安全写入均已覆盖。

### B2：Normalizer、Indexer 与四类文档

- **实现文件**：`src/ingestion/normalizer.py`、`loaders.py`、`splitter.py`、`indexer.py`。
- **核心接口**：`TextNormalizer.normalize_text()`、`LoaderFactory.load()`、`split_document()`、`Indexer.index_document()`、`Indexer.reindex_document()`。
- **测试证据**：`tests/unit/ingestion/test_normalizer.py`、`test_indexer.py`，以及既有 Loader/Splitter 单元测试。
- **验收结论**：TXT、Markdown、PDF、DOCX 的真实离线样本处理、原始字节 checksum、稳定 Chunk ID、Embedding 校验、失败回滚与重索引已覆盖。

### B3：向量检索与知识库隔离

- **实现文件**：`src/retrieval/embeddings.py`、`vector_index.py`、`retriever.py`、`adapter.py`、`hybrid.py`。
- **核心接口**：`InMemoryVectorIndex`、`ChromaVectorIndex`、`VectorRetriever.search()`、`VectorRetrieverAdapter.retrieve()`/`search()`。
- **测试证据**：`tests/unit/retrieval/test_vector_index_isolation.py`、`test_vector_retriever.py`、`test_hybrid_isolation.py`、`test_retriever_adapter.py`。
- **验收结论**：`kb_id` 与 metadata 在排名和 Top-K 前过滤；Adapter 可同时满足 C 的 `retrieve()` 注入面和 B Strategy 的 `search()` 调用面，且不重复执行 Embedding。

### B4：Naive RAG

- **实现文件**：`src/rag/strategies/naive.py`、`src/rag/naive_support.py`、`src/rag/c_contract.py`。
- **核心接口**：`async NaiveRAGStrategy.run(request, context) -> RAGResult`。
- **测试证据**：`tests/unit/rag/test_naive_strategy.py`、`tests/integration/test_naive_rag.py`。
- **验收结论**：单次隔离检索、阈值、有界上下文、生成、Citation、Trace 和无依据拒答均已覆盖；无有效依据时不调用 Generator。

### B5：Advanced RAG

- **实现文件**：`src/rag/strategies/advanced.py`、`src/retrieval/query_rewriter.py`、`multi_query.py`、`hybrid.py`、`reranker.py`、`compressor.py`。
- **核心接口**：`async AdvancedRAGStrategy.run(request, context) -> RAGResult`，以及五个可独立配置的高级模块开关。
- **测试证据**：`tests/unit/rag/test_advanced_strategy.py`、Query Rewrite/Multi-Query/Hybrid 单元测试、`tests/integration/test_advanced_rag.py`。
- **验收结论**：高级模块的正常路径、独立关闭、异常降级、稳定去重、重排追踪、压缩回退和最小消融均已覆盖。测试只证明流程差异，不宣称质量必然提升。

### B6：Vector Search Tool

- **实现文件**：`src/retrieval/vector_tool.py`。
- **核心接口**：`VectorSearchTool.execute()`、`invoke()`、`ainvoke()`、`CToolResultAdapter`、`register_vector_search_tool()`。
- **测试证据**：`tests/unit/agent/test_vector_tool.py`、`tests/contract/test_vector_tool_contract.py`、`tests/integration/test_vector_tool.py`。
- **验收结论**：`vector_search` 参数校验、一次检索边界、JSON 安全结果、结构化错误、Registry/Executor 调用和跨知识库隔离均已覆盖。

## 提交历史

| 阶段 | 提交 | 信息 |
| --- | --- | --- |
| Step 0 | `1d3dd69c8737de018106372c04100c815765980c` | `fix(b): clean embedding adapters and optional dependencies` |
| B1 | `a2e15cca88fe99551ff50fa93809b43df731498d` | `feat(b1): implement knowledge repositories and service` |
| B2 | `e503c1c89f38ab943a2e3707e87c714b42a01dc8` | `feat(b2): implement normalization and indexing pipeline` |
| B3 | `3a276be633a771379f63f03f109c676bdb4b1af2` | `feat(b3): enforce kb scoped vector retrieval` |
| B4 | `095d5601d86be26a773b6a33ed745fb4a26c409a` | `feat(b4): implement naive rag strategy` |
| B5 | `bed6263ba82de5572197eed0569f3ad701dad4d0` | `feat(b5): implement advanced rag strategy` |
| B6 | `1a71a2b1da66793e069a810a64219f8455422ee9` | `feat(b6): add vector search agent tool` |
| 合同 Adapter 修复 | `8d5c611f2dd2b823a4cbe7847b3b04c0acf0bddd` | `fix(b): align adapters with c contracts` |

本次文档提交在编写本文时尚未创建，因此不预先填写提交哈希；应以 Git 历史中的 `docs: update readme and project progress` 为准。

## 质量证据

| 检查 | 实际结果 |
| --- | --- |
| `python -m compileall src` | 通过，退出码 0 |
| `python -m pytest --collect-only -q` | 199 tests collected |
| `python -m pytest tests/unit -q` | 191 passed |
| `python -m pytest tests/integration -q` | 6 passed |
| `python -m pytest tests/contract -q` | 2 passed |
| `python -m pytest -q` | 199 passed |
| Git 冲突标记 | 当前基线未发现 |
| 工作区 | 文档修改前干净；最终状态以文档提交后的 Git 验证为准 |

## 待办事项

### 后续集成

- 与 C 的真实共享合同运行跨分支集成测试。
- 根据团队流程创建 PR。
- 合入主分支前处理冲突并验证 CI。

### 未实现功能

- Agent Router。
- Agent Workflow。
- Agentic RAG。
- Modular RAG。
- GraphRAG。
- 完整 FastAPI 应用。
- 前端。
- 正式质量评估基准。

### 非阻塞环境问题

- Windows 上旧 pytest 临时目录可能在退出清理阶段产生权限提示；当前测试命令退出码仍为 0。

## 风险说明

- B 侧合同 Adapter 已存在并有回归测试，但仍需在真实合并环境中复测共享模型、注册和执行链。
- HuggingFace/BGE 与 OpenAI Embedder 需要对应可选依赖；在线调用还需要安全提供凭据和网络环境。
- Chroma 的生产持久化需要部署方配置路径、容量、备份和生命周期策略。
- 当前测试重点是功能、合同、错误降级与知识库隔离正确性，不是生产回答质量或性能评估。
- `HashEmbedder` 用于确定性离线验证，不应作为生产语义质量结论的依据。
