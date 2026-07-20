# LangChain RAG

本仓库包含文档加载、文本切分、向量检索、混合检索、重排、上下文压缩和命令行 RAG 演示。基础测试使用确定性的 `HashEmbedder`，不访问网络，也不需要真实模型或 API Key。

## 基础安装

建议使用 Python 3.9 或更高版本，并在虚拟环境中安装基础依赖：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

基础依赖只包含 In-Memory 检索、Loader 和 Splitter 所需的软件包。导入 `src.retrieval` 不会强制安装或加载 BGE、OpenAI 或 Chroma。

## BGE Embedding（可选）

本地 BGE 向量模型由 `HuggingFaceEmbedder` 提供：

```bash
python -m pip install -r requirements-bge.txt
```

默认模型为 `BAAI/bge-small-zh-v1.5`。`sentence-transformers` 采用延迟导入，模型也只在第一次向量化非空文本时创建。首次使用默认模型可能需要联网下载；自动化测试不会创建或下载真实模型。

可以配置运行设备和批量大小：

```python
from src.retrieval import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu",
    batch_size=32,
)
```

## Chroma（可选）

需要持久化 Chroma 向量索引时安装：

```bash
python -m pip install -r requirements-chroma.txt
```

`chromadb` 只在创建 `ChromaVectorIndex` 时导入。Chroma 本地目录和数据库文件已加入 `.gitignore`，不得提交到仓库。

## OpenAI 兼容接口与 DeepSeek CLI（可选）

`OpenAIEmbedder` 和 `rag_chat.py` 使用 OpenAI 兼容客户端：

```bash
python -m pip install -r requirements-openai.txt
```

运行使用 BGE + DeepSeek 的 CLI 演示，需要同时安装 BGE 依赖：

```bash
python -m pip install -r requirements-bge.txt -r requirements-openai.txt
python rag_chat.py
```

CLI 从本地 `.env` 读取 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。`.env` 已被忽略；不要提交真实 API Key。离线测试不会调用 DeepSeek、OpenAI 或其他真实网络服务。

## 离线验证

```bash
python -m compileall src
python -m pytest --collect-only -q
python -m pytest tests -k "embedding or retrieval" -q
python -m pytest -q
```

这些命令只使用本地测试替身和 `HashEmbedder`，不需要安装 BGE/Chroma/OpenAI 可选依赖。
