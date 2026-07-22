# Knowledge Base Document Storage

本文档说明知识库上传文件的持久化约束，避免后续再次出现“前端选了知识库但 RAG/Graph 没命中、显示 0 切片、同名文件互相覆盖”的问题。

## 核心规则

- 文档元数据保存在 `data/knowledge_bases.json`。
- 新上传的真实文件保存在 `documents/<kb_id>/<doc_id>__<filename>`。
- 文档记录必须保存 `storage_path`，它是 `documents/` 下的相对路径。
- `filename` 只用于前端展示、引用来源、citation，不应该再被当成唯一文件路径。
- 旧记录如果没有 `storage_path`，后端会兼容读取 `documents/<filename>`。

## 读取入口

所有读取真实文件的后端入口都必须通过统一 resolver：

- `src.api.document_store.resolve_document_path`
- `src.api.document_store.apply_document_metadata`

当前已接入的入口：

- `src/api/routes/knowledge.py`: 上传、预览、删除、chunk count
- `src/api/unified_rag_service.py`: `/api/rag/query` 五种 RAG mode 初始化索引
- `src/api/routes/graph.py`: `/api/graphs/{kb_id}` 构图

## 删除约束

删除文档或知识库时，只能删除当前记录独占的真实文件。若另一个文档记录仍解析到同一个路径，文件必须保留，防止旧数据或共享 fallback 文件被误删。

## 防回归测试

修改知识库上传、RAG 初始化、Graph 构图、文档删除逻辑时，至少运行：

```bash
pytest tests/test_knowledge_document_store.py tests/test_graph_model.py tests/integration/test_unified_rag_modes.py
```

这些测试覆盖：

- 两个知识库上传同名文件时，必须生成不同 `storage_path`。
- 预览必须读到各自知识库的真实文件内容。
- 删除一个文档或知识库不能删除另一个知识库的同名文件。
- Unified RAG 文档加载必须优先使用 `storage_path` 并保留真实 `kb_id`。
- Graph chunk loader 必须使用同一套路径解析和来源 metadata。
