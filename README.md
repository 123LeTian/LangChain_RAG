# LangChain RAG + Chat Platform

这是一个基于 FastAPI + Vue 3 的多范式 RAG 实验平台，并融合了 Chat Platform 的会话、模型、预设和流式对话能力。

项目目标是在不破坏原有 RAG 核心能力的前提下，支持：

- Naive RAG
- Advanced RAG
- Modular RAG
- GraphRAG
- Agentic RAG
- 多轮 Chat Session
- SSE 流式输出
- 模型配置与会话内模型切换
- 内置预设与个人预设 CRUD
- 历史消息搜索
- Token 统计
- Markdown 导出
- dev / test / prod 分层配置

## 当前状态

当前分支已经完成 Chat Platform 主要融合能力，前端 Chat 页面已接入真实会话、历史、模型配置、预设配置和 session-aware stream。

RAG 侧的核心代码仍保留在独立模块中：

```text
src/rag/
src/retrieval/
src/graph/
src/agent/
src/evaluation/
```

近期 Chat / 模型 / 预设 / UI 改动没有改动这些 RAG 核心目录。

`/api/rag/query` 与 `/api/rag/query/stream` 已通过 `UnifiedRAGApiService` 按 `request.mode` 真正分发到 `src/rag/strategies/` 下对应的五种 Strategy（naive/advanced/modular/graph/agentic）。依赖注入（retriever、LLM、graph、reranker、compressor）由 `RAGService` + `RAGContext` 统一管理，不再走 `RealRAGService` 的单一模块管线。`RealRAGService` 保留作为底层文档/embedding/检索组件的初始化参考。

## 主要能力

### RAG 能力

| 模式 | 当前状态 | 说明 |
| --- | --- | --- |
| Naive RAG | ✅ 已接入 | `POST /api/rag/query` 按 mode=naive 分发到 NaiveRAGStrategy |
| Advanced RAG | ✅ 已接入 | `POST /api/rag/query` 按 mode=advanced 分发到 AdvancedRAGStrategy（rewrite/rerank/compress） |
| Modular RAG | ✅ 已接入 | `POST /api/rag/query` 按 mode=modular 分发到 ModularRAGStrategy（模块开关、pipeline_config） |
| GraphRAG | ✅ 已接入 | `POST /api/rag/query` 按 mode=graph 分发到 GraphRAGStrategy（local/global graph search）；图谱为空时返回 refusal + warning |
| Agentic RAG | ✅ 已接入 | `POST /api/rag/query` 按 mode=agentic 分发到 AgenticRAGStrategy（AgentWorkflow + tool use）；graph 不可用时自动降级 |

### Chat Platform 能力

- 创建、加载、重命名、删除会话
- 会话消息自动保存
- 用户消息和助手消息持久化
- 多轮上下文拼接
- Session-aware RAG 流式接口
- 自动标题
- 模型配置管理
- 用户自定义模型
- API Key 本地保存，不返回前端
- 模型连接测试
- 会话内模型切换
- 内置预设和个人预设 CRUD
- 会话内预设切换
- 历史消息搜索
- Token 统计
- Markdown 导出
- 停止生成后显示“对话已停止”

## 后端接口

旧 RAG 接口保持兼容：

```text
POST /api/rag/query
POST /api/rag/query/stream
POST /api/rag/compare/stream
```

Chat Session 接口：

```text
POST   /api/chat/sessions
GET    /api/chat/sessions
GET    /api/chat/sessions/{session_id}
PATCH  /api/chat/sessions/{session_id}
DELETE /api/chat/sessions/{session_id}
GET    /api/chat/sessions/{session_id}/messages
POST   /api/chat/sessions/{session_id}/messages
POST   /api/chat/sessions/{session_id}/stream
PATCH  /api/chat/sessions/{session_id}/model
PATCH  /api/chat/sessions/{session_id}/preset
```

模型接口：

```text
GET    /api/chat/models
GET    /api/chat/models/manage
POST   /api/chat/models
PATCH  /api/chat/models/{model_id}
DELETE /api/chat/models/{model_id}
PATCH  /api/chat/models/default
POST   /api/chat/models/{model_id}/test
POST   /api/chat/models/discover
```

预设接口：

```text
GET    /api/chat/presets
POST   /api/chat/presets
PATCH  /api/chat/presets/{preset_id}
DELETE /api/chat/presets/{preset_id}
```

搜索、统计、导出：

```text
GET /api/chat/search
GET /api/chat/stats
GET /api/chat/sessions/{session_id}/stats
GET /api/chat/sessions/{session_id}/export
```

运行时配置：

```text
GET /api/config/runtime
```

## 默认模型

仓库默认只内置一个本地 mock 模型：

```text
mock-chat
```

默认模型固定回答：

```text
您好，我是本地默认模型，不会回答任何问题，只用于测试，请在模型配置页面添加您的模型
```

真实模型请在前端“模型配置”页面添加。API Key 只会写入本地环境文件，模型配置中只保存 `api_key_env`，接口响应不会返回真实密钥。

## 内置预设

当前只保留三个内置预设：

- 默认助手
- RAG 证据专家
- 工程代码助手

内置预设为只读。用户可以在“提示词预设”页面创建、修改、删除个人预设，并为当前会话选择预设。

## 数据和密钥

不要提交以下内容：

```text
.env
.env.dev
.env.test
.env.prod
data/
logs/
frontend/dist/
导出的 Markdown 文件
运行时数据库
```

仓库只保留 `.env.*.example` 这类占位示例。

## 安装

后端依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-openai.txt
pip install -r requirements-bge.txt
```

前端依赖：

```bash
cd frontend
npm install
```

## 启动

后端：

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend
npm run dev
```

默认地址：

```text
Frontend: http://localhost:5173
Backend:  http://127.0.0.1:8000
Health:   http://127.0.0.1:8000/health
```

## 测试

推荐先运行重点测试：

```bash
pytest tests/test_chat_sessions.py tests/test_chat_stream.py tests/test_chat_models.py tests/test_chat_presets.py tests/test_chat_preset_prompt.py
pytest tests/unit/rag tests/unit/retrieval tests/unit/graph tests/unit/agent tests/integration/test_naive_rag.py tests/integration/test_advanced_rag.py
```

前端构建：

```bash
cd frontend
npm run build
```

Windows 下 pytest 退出阶段偶尔会出现临时目录清理 `WinError 5` 提示；只要 pytest 主结果和退出码通过，通常不是功能失败。

## 常见排查

### 选择知识库后 RAG 没有命中文档

先在“知识库管理”页面确认选中的知识库 `Chunk 数` 大于 0。上传文档后接口会动态计算切片数；如果页面仍显示 0，可以点击“重建索引”或重启后端。

统一 RAG 服务会根据 `data/knowledge_bases.json` 中的文档记录保留真实 `kb_id`，向量检索和 GraphRAG 都按当前选择的知识库隔离检索。若直接把文件复制到 `documents/` 但没有对应知识库文档记录，它只会作为 fallback 的 `default` 知识库文档参与索引。

### 修改后前端看起来没有变化

确认只运行一个后端进程，并重启：

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

浏览器页面使用的是 `http://localhost:5173`，实际 API 请求走 `http://127.0.0.1:8000`。如果有多个 uvicorn 同时占用 8000，可能会连到旧进程。

## 已知边界

- Graph 页面当前仍有演示数据，需要进一步接真实图谱仓储。
- Evaluation 页面当前仍有 mock 结果，需要进一步接真实 evaluation runner。
- GraphRAG 模式依赖文档图谱构建（`GraphIndexBuilder`）；若 `documents/` 为空或图谱构建失败，GraphRAG 会在 warnings 中返回结构化提示并降级返回 refusal answer。
- Agentic 模式依赖 AgentWorkflow + tool registry；若 graph retriever 不可用，Agent 会自动降级到 vector_search（trace/warnings 会记录降级路径）。
- 真实模型调用依赖用户本地添加模型和 API Key。

