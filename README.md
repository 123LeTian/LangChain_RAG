# LangChain RAG 多范式实验平台

本项目是一个面向学习、课程展示和实验验证的多范式 RAG 系统。它把文档上传、知识库管理、五种 RAG 模式问答、执行链路追踪、知识图谱展示、模式评测对比整合在同一个 Web 工作台中，便于对不同 RAG 技术路线进行直观比较。

项目后端基于 FastAPI，前端基于 Vue 3 + Vite，默认提供无需密钥的 `Mock Chat` 模型用于本地流程验证；配置 OpenAI-compatible 模型后，可以接入 DeepSeek、OpenAI、通义千问兼容接口或其他兼容 Chat Completions 的模型服务。

## 项目亮点

- 支持五种 RAG 模式：Naive RAG、Advanced RAG、Modular RAG、GraphRAG、Agentic RAG。
- 支持 TXT、Markdown、PDF、DOCX 文档上传、切分、索引和知识库持久化。
- 支持向量检索、关键词检索、混合检索、重排序、上下文压缩、查询改写和答案校验。
- 支持 GraphRAG：从文档中抽取实体关系，构建知识图谱，并支持 Local / Global 图检索思路。
- 支持 Agentic RAG：通过意图识别选择检索工具，支持 Vector Tool / Graph Tool 的最小可验收智能编排。
- 支持 Modular RAG 模块消融实验：可开关 rewrite / retrieve / rerank / compress / verify 等模块，非法组合会被前后端拦截。
- 支持 RAG 模式评测页面：同一知识库、同一模型、同一样本集下并发运行五种模式，输出 Hit@3、MRR、答案覆盖率、延迟、Token 等指标。
- 支持 Trace 追踪：展示每次问答的 preset、model、intent、retrieve、rerank、compress、generate、verify、complete 等阶段。
- 支持模型管理、提示词预设、历史对话、会话导出和聊天记录搜索。

## 功能模块说明

### 1. 问答实验台

问答实验台是主要交互页面。用户可以选择知识库、模型、提示词预设和 RAG 模式，然后向系统提问。

支持的能力包括：

- 直接问答：不使用知识库，只调用模型回答。
- Naive RAG：基础检索增强生成，适合简单事实问答。
- Advanced RAG：在基础检索上加入查询改写、重排序、压缩和验证等增强步骤。
- Modular RAG：将 RAG 流水线拆成可独立开关的模块，用于做消融实验。
- GraphRAG：使用知识图谱实体、关系和社区报告进行图增强问答。
- Agentic RAG：先判断问题意图，再选择向量检索、图检索等工具完成回答。

问答结果包含：

- 答案正文
- 引用片段
- 执行 Trace
- Token 使用量
- 模型与预设信息
- RAG 模式标签

### 2. 知识库管理

知识库页面用于创建知识库、上传文档和构建索引。

支持格式：

| 类型 | 后缀 |
| --- | --- |
| 文本文档 | `.txt` |
| Markdown | `.md`, `.markdown` |
| PDF | `.pdf` |
| Word | `.docx` |

文档处理流程：

```text
上传文档
-> LoaderFactory 读取 TXT / MD / PDF / DOCX
-> 文本归一化
-> RecursiveCharacterTextSplitter 切分
-> Embedding 向量化
-> 写入向量索引
-> 保存知识库、文档、chunk 元数据
```

默认切分参数：

- `chunk_size = 1500`
- `chunk_overlap = 300`

### 3. 五种 RAG 模式

#### Naive RAG

基础 RAG 流程：

```text
用户问题 -> 检索 Top-K 文档片段 -> 拼接上下文 -> LLM 生成答案 -> 返回引用
```

适合验证：

- 简单事实题
- 明确出现在文档中的短答案
- 基础检索能力

#### Advanced RAG

Advanced RAG 在 Naive RAG 的基础上增加增强步骤：

```text
用户问题
-> 查询改写 / 多查询扩展
-> 混合检索
-> 重排序
-> 上下文压缩
-> LLM 生成
-> 可选答案校验
```

适合验证：

- 需要整合多个片段的问题
- 需要归纳和解释的问题
- 对上下文噪声更敏感的问题

#### Modular RAG

Modular RAG 将流水线拆为可配置模块：

| 模块 | 作用 | 默认 |
| --- | --- | --- |
| `rewrite` | 查询改写与扩展 | 开 |
| `retrieve` | 向量 + 关键词混合检索 | 开 |
| `rerank` | 相关性重排序 | 开 |
| `compress` | 上下文压缩 | 开 |
| `verify` | 答案校验 | 关 |
| `generate` | 答案生成 | 固定开启 |

非法组合校验：

- `rerank` 依赖 `retrieve`
- `compress` 依赖 `retrieve`
- Agentic RAG 至少需要开启一个检索工具

当用户关闭某些模块时，系统会在 Trace 中显示 skipped 状态，方便观察模块消融对答案质量和延迟的影响。

#### GraphRAG

GraphRAG 会从 chunk 中抽取实体和关系，构建图谱结构：

```text
Chunk -> 实体抽取 -> 关系抽取 -> NetworkX 图索引 -> 社区报告 -> 图检索 -> 答案生成
```

支持两类典型问题：

- Local Graph Search：实体关系类问题，例如“i 茅台和直销渠道是什么关系？”
- Global Graph Search：整体概括类问题，例如“请总体说明贵州茅台长期竞争优势由哪些方面构成。”

图谱页面可以展示：

- 节点
- 边
- 社区
- 命中路径
- 图谱构建状态

#### Agentic RAG

Agentic RAG 是最小可验收的智能体式 RAG：

```text
用户问题
-> 意图识别
-> 选择工具
   -> vector_search
   -> graph_search
-> 工具调用
-> 汇总证据
-> 生成答案
-> 校验
```

目前最小版本重点验证：

- 能独立走 Agentic 后端链路
- 能使用向量检索回答事实题
- 能使用图谱信息回答关系题
- 关闭所有检索工具时返回中文配置错误
- Trace 中可看到 intent、tool_call、generate、verify、complete 等阶段

### 4. RAG 模式评测页面

RAG 模式评测页面用于比较五种模式在同一条件下的表现。

评测输入：

- 知识库
- 公共 LLM 模型
- 样本数量：1 / 3 / 5 / 10 / 全部
- 固定模式：Naive、Advanced、Modular、Graph、Agentic

评测逻辑：

```text
选择知识库和模型
-> 从当前知识库 chunk 动态构造评测样本
-> 五种 RAG 模式并发运行
-> 汇总检索指标、答案指标、系统指标
-> 表格和图表展示结果
```

主要指标：

| 指标 | 含义 |
| --- | --- |
| Hit@3 | 期望来源是否出现在 Top-3 检索结果中 |
| MRR | 第一个正确来源排名的倒数均值 |
| 答案覆盖率 | 答案是否覆盖期望事实 |
| P50 延迟 | 首 token / 响应延迟的中位表现 |
| 总耗时 | 每种模式总运行时间 |
| Token 总数 | 输入与输出 token 估算总量 |

评测页面不是只针对某一个示例知识库写死，而是从当前选中的知识库 chunk 动态生成样本，因此可以用于任意上传的知识库。

## 技术架构

```text
frontend/ Vue 3 + Vite
    |
    | REST / SSE
    v
src/api/ FastAPI
    |
    +-- chat service / model registry / preset service
    +-- knowledge service / document service
    +-- RAGService facade
            |
            +-- ingestion
            +-- retrieval
            +-- rag strategies
            +-- graph
            +-- agent
            +-- evaluation
```

后端核心目录：

```text
src/
  api/              FastAPI 应用、路由和依赖注入
  agent/            Agentic RAG 的路由、工具、状态和工作流
  chat/             会话、模型配置、提示词预设、流式输出
  chat_storage/     SQLite 聊天记录持久化
  config/           环境和运行时配置
  evaluation/       评测数据集、指标、报告和五模式 runner
  graph/            GraphRAG 图构建、检索、社区报告
  ingestion/        文档加载、归一化、切分、索引
  knowledge/        知识库、文档、chunk 仓储与状态机
  models/           Pydantic 数据模型
  rag/              Naive / Advanced / Modular / Graph / Agentic 策略
  retrieval/        embedding、向量索引、混合检索、rerank、compress
```

前端核心目录：

```text
frontend/
  src/
    App.vue                     全局布局、侧边栏、模型和预设管理入口
    api/client.ts               REST / SSE API 客户端
    components/CustomSelect.vue 自定义下拉组件
    views/ChatView.vue          问答实验台
    views/KnowledgeBaseView.vue 知识库管理
    views/GraphView.vue         知识图谱
    views/EvaluationView.vue    RAG 模式评测
    views/TraceView.vue         Trace 查询
```

## 环境要求

建议环境：

- Python 3.10 或以上
- Node.js 18 或以上
- npm 9 或以上
- Windows / macOS / Linux 均可运行

说明：

- 默认 `Mock Chat` 不需要 API Key，可以用于验证页面和流程。
- 如果使用真实 LLM，需要配置 OpenAI-compatible 模型和 API Key。
- 如果使用本地 BGE embedding，首次运行会下载 HuggingFace 模型。

## 安装依赖

### 1. 创建 Python 虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. 安装后端核心依赖

```bash
pip install -r requirements.txt
```

核心依赖可以启动 API、运行基础测试和查看页面。若要上传文档后进行真实向量检索，请继续安装 BGE embedding 依赖。

### 3. 安装开发测试依赖

```bash
pip install -r requirements-dev.txt
```

### 4. 安装真实 LLM 依赖

如果要接入 DeepSeek、OpenAI 或其他 OpenAI-compatible 模型：

```bash
pip install -r requirements-openai.txt
```

### 5. 安装 BGE embedding 依赖

如果要使用 `BAAI/bge-small-zh-v1.5`：

```bash
pip install -r requirements-bge.txt
```

### 6. 安装 Chroma 可选依赖

如果要使用 Chroma 持久化向量库：

```bash
pip install -r requirements-chroma.txt
```

如果希望一次性安装开发、LLM、BGE 和 Chroma 的全部依赖，可以使用：

```bash
pip install -r requirements-all.txt
```

### 7. 安装前端依赖

```bash
cd frontend
npm install
```

## 配置说明

### 环境配置

仓库提供：

- `.env.dev.example`
- `.env.test.example`
- `.env.prod.example`

可以复制为本地 `.env`：

```powershell
Copy-Item .env.dev.example .env
```

示例：

```env
APP_ENV=dev
DEEPSEEK_API_KEY=your-dev-deepseek-key
OPENAI_API_KEY=your-dev-openai-key
DASHSCOPE_API_KEY=your-dev-dashscope-key
```

说明：

- 启动命令中仍建议显式设置 `APP_ENV=dev`，用于选择 `config.dev.yaml`。
- 根目录 `.env` 会被真实 RAG 服务读取，可用于放模型 API Key。
- `.env`、`.env.dev`、`.env.prod` 等真实密钥文件已被 `.gitignore` 忽略，请不要提交。

### 模型配置

默认模型配置在 `config/models.yaml`：

```yaml
models:
  - id: mock-chat
    provider: mock
    display_name: Mock Chat
    model_name: mock-chat
    enabled: true
    is_default: true
```

启动后可以在前端“模型配置”页面添加真实模型。模型配置支持：

- provider
- display name
- model name
- base url
- api key 或 api key 环境变量
- 是否支持 streaming
- 是否启用
- 是否设为默认模型

### 运行时配置

配置文件：

- `config.dev.yaml`
- `config.test.yaml`
- `config.prod.yaml`
- `config.yaml`

主要控制：

- SQLite 聊天记录路径
- 知识库数据路径
- 向量数据路径
- 导出文件路径
- 日志路径
- 模型配置文件路径

## 启动项目

### 1. 启动后端

Windows PowerShell：

```powershell
$env:APP_ENV = "dev"
$env:PYTHONPATH = "."
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

macOS / Linux：

```bash
export APP_ENV=dev
export PYTHONPATH=.
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```text
http://127.0.0.1:8000/health
```

期望返回：

```json
{
  "status": "ok",
  "version": "0.1.0",
  "service": "LangChain RAG API"
}
```

### 2. 启动前端

新开一个终端：

```bash
cd frontend
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

## 推荐演示流程

老师或助教可以按下面流程快速验收：

### 步骤 1：启动项目

1. 安装依赖。
2. 启动后端 `127.0.0.1:8000`。
3. 启动前端 `127.0.0.1:5173`。
4. 打开页面确认侧边栏包含：问答实验台、知识库管理、知识图谱、RAG模式评测。

### 步骤 2：创建知识库并上传文档

1. 进入“知识库管理”。
2. 创建一个知识库。
3. 上传 PDF / TXT / MD / DOCX。
4. 等待文档完成索引。

### 步骤 3：验证五种 RAG 问答

进入“问答实验台”，选择知识库和模型，分别切换：

- Naive RAG
- Advanced RAG
- Modular RAG
- GraphRAG
- Agentic RAG

建议测试问题：

```text
这份文档的主要内容是什么？
文档中提到的核心流程有哪些？
请根据文档总结关键优势。
只根据文档，是否提到了某个不存在的业务？
```

验收重点：

- 答案是否基于引用片段。
- 无依据问题是否拒答或提示知识库中未找到充分依据。
- Trace 是否显示对应模式的执行阶段。
- 引用来源是否能展开查看。

### 步骤 4：验证 Modular RAG

建议配置：

- Retrieve 开
- Rerank 开
- Compress 开
- Verify 可选

再测试非法组合：

- Retrieve 关
- Rerank 开

期望：

- 前端或后端提示中文配置错误。
- 不应该继续执行无效链路。

### 步骤 5：验证 GraphRAG

适合问题：

```text
文档中的 A 和 B 是什么关系？
请根据图谱整体概括核心主题。
图谱中是否提到了某个不存在的业务？
```

验收重点：

- 关系类问题能体现实体和边。
- 总结类问题能体现整体图谱信息。
- 不存在的信息不应被编造。

### 步骤 6：验证 Agentic RAG

适合问题：

```text
文档的主要业务或主题是什么？
文档中两个概念之间是什么关系？
只根据文档，是否提到了某个不存在的事项？
```

验收重点：

- Trace 中出现 intent 和 tool_call。
- 事实题能走 vector_search。
- 关系题可以利用 graph_search。
- 关闭 Vector Tool 和 Graph Tool 时返回中文配置错误。

### 步骤 7：验证 RAG 模式评测

进入“RAG模式评测”：

1. 选择知识库。
2. 选择公共模型。
3. 样本数量先选 `1 条样本`。
4. 点击“运行评测”。

期望：

- 五种模式并发执行。
- 表格出现 Hit@3、MRR、答案覆盖率、P50 延迟、总耗时、Token 总数。
- 下方出现三张图表：Hit@3 命中率、答案覆盖率、总耗时对比。
- 最优指标会高亮。

## 常用 API

| Endpoint | Method | 说明 |
| --- | --- | --- |
| `/health` | GET | 健康检查 |
| `/api/knowledge-bases` | GET / POST | 知识库列表 / 创建 |
| `/api/knowledge-bases/{kb_id}/documents` | GET / POST | 文档列表 / 上传 |
| `/api/knowledge-bases/{kb_id}/index` | POST | 重建索引 |
| `/api/rag/query` | POST | 非流式 RAG 问答 |
| `/api/rag/query/stream` | POST | SSE 流式 RAG 问答 |
| `/api/rag/compare/stream` | POST | 两组 Modular 配置对比 |
| `/api/graphs/{kb_id}` | GET | 获取知识图谱 |
| `/api/evaluations/run` | POST | 启动五模式评测 |
| `/api/evaluations/{run_id}` | GET | 获取评测结果 |
| `/api/traces/{trace_id}` | GET | 获取 Trace |
| `/api/chat/sessions` | GET / POST | 会话列表 / 创建 |
| `/api/chat/models` | GET / POST | 模型列表 / 添加 |
| `/api/chat/presets` | GET / POST | 预设列表 / 添加 |

## 测试与构建

### 后端单元测试

```bash
python -m pytest -q
```

评测相关快速测试：

```bash
python -m pytest tests/unit/api/test_evaluation_routes.py tests/unit/evaluation/test_multi_mode_runner.py tests/unit/evaluation/test_final_report.py -q
```

### 前端构建

```bash
cd frontend
npm run build
```

### 评测脚本

```bash
$env:PYTHONPATH = "."
python eval_ablation.py
```

该脚本会运行固定评测样本，输出检索指标并生成 `eval_results.json`。该文件为运行产物，默认不建议提交。

## 依赖文件说明

| 文件 | 用途 |
| --- | --- |
| `requirements.txt` | 后端核心依赖，包含 FastAPI、文档解析、LangChain Core、图谱、基础数值计算 |
| `requirements-dev.txt` | 开发和测试依赖 |
| `requirements-openai.txt` | OpenAI-compatible LLM 接入依赖 |
| `requirements-bge.txt` | HuggingFace / BGE embedding 依赖 |
| `requirements-chroma.txt` | Chroma 向量库依赖 |
| `requirements-all.txt` | 一次性安装开发、LLM、BGE、Chroma 的完整依赖 |
| `frontend/package.json` | 前端 Vue 3 / Vite 依赖 |

## HuggingFace 离线模式

如果已经下载过 `BAAI/bge-small-zh-v1.5`，可以开启离线模式避免网络或证书问题。

Windows PowerShell：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

macOS / Linux：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

首次下载：

```bash
huggingface-cli download BAAI/bge-small-zh-v1.5
```

## 数据与文件持久化

运行时数据默认不提交到 Git：

| 路径 | 说明 |
| --- | --- |
| `documents/` | 用户上传的原始文档 |
| `data/` | 知识库、SQLite、向量数据、导出文件 |
| `logs/` | 运行日志 |
| `.env*` | 本地密钥和环境变量 |
| `frontend/dist/` | 前端构建产物 |

这些路径已在 `.gitignore` 中忽略。

## 常见问题

### 1. 前端能打开，但请求失败

检查后端是否启动：

```text
http://127.0.0.1:8000/health
```

如果后端端口不是 8000，需要同步调整 Vite 代理或前端 API base。

### 2. 上传 PDF 后没有内容

本项目使用 `pypdf` 抽取文本。扫描版 PDF 可能只有图片，没有可抽取文本，需要先 OCR。

### 3. 真实模型无法调用

检查：

- 模型是否启用。
- API Key 是否配置。
- Base URL 是否符合 OpenAI-compatible 格式。
- 模型名是否正确。
- 网络是否可以访问模型服务。

### 4. BGE 模型下载失败

可以先使用 `Mock Chat` 和 HashEmbedder 走基础流程；如需真实 embedding，建议提前下载 HuggingFace 模型，或配置离线缓存。

### 5. RAG 模式评测运行较慢

评测会并发调用五种模式。真实 LLM 下即使选择 1 条样本，也可能需要等待多个模型调用完成。建议课堂演示时：

- 先选择 `1 条样本`。
- 使用较快模型。
- 确保知识库 chunk 数量适中。
- 避免一次选择“全部样本”。

## 当前完成度

已完成：

- 文档上传与知识库管理
- Naive RAG
- Advanced RAG
- Modular RAG 与模块消融
- GraphRAG 基础图谱构建、关系检索和全局概括
- Agentic RAG 最小可验收版本
- Trace 展示
- 模型配置与提示词预设
- 历史会话、搜索和导出
- 五模式 RAG 评测页面
- 动态样本生成与指标统计
- 前端现代化评测页面

后续可继续增强：

- 更强的 Cross-Encoder Reranker
- 更完整的 Agentic 多步规划
- 更稳定的 LLM-as-a-Judge 评测
- 图谱可视化交互优化
- Docker Compose 一键部署
- CI 自动测试与构建

## 课程汇报建议

可以从以下角度介绍本项目：

1. 为什么需要 RAG：解决大模型无法直接访问私有文档、容易幻觉的问题。
2. 为什么做多范式：不同 RAG 技术路线适合不同问题类型。
3. Naive 到 Advanced：从基础检索到查询改写、重排序、压缩和校验。
4. Modular 的价值：通过开关模块做消融实验，观察每个模块的影响。
5. GraphRAG 的价值：用实体关系补充纯向量检索难以表达的结构信息。
6. Agentic RAG 的价值：让系统根据问题选择检索工具，而不是固定流水线。
7. 模式评测的价值：用统一样本、统一模型、统一知识库对比不同模式的效果和成本。
