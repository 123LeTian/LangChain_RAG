# LangChain RAG

A multi-paradigm RAG experimental platform for learning, validation, and demo.
Implements five RAG modes (Naive / Advanced / Modular / Graph / Agentic) with
a FastAPI backend and Vue 3 frontend.

## Features

### RAG Pipeline
- **Document Loading**: TXT / MD / PDF / DOCX via `LoaderFactory`
- **Text Splitting**: Recursive character splitter (chunk_size=1500, overlap=300)
- **Embeddings**: HuggingFace `BAAI/bge-small-zh-v1.5` (512-dim, Chinese-optimized)
- **Retrieval**: `HybridRetriever` (vector + keyword, alpha=0.3)
- **Reranker**: `SimpleReranker` with Chinese trigram tokenization + phrase boost
- **Query Rewriting**: LLM-powered query expansion when a chat model is configured
- **Context Compression**: Token-budget-aware truncation
- **Answer Verification**: LLM factuality check (optional)
- **LLM**: User-configured OpenAI-compatible chat model; clean clones default to `Mock Chat`

### 5-Module Pipeline Switches (Modular RAG)
Each module can be independently toggled on/off:

| Module | Description | Default |
|--------|-------------|---------|
| `rewrite` | Query rewriting / expansion | ON |
| `retrieve` | Vector + keyword hybrid retrieval | ON |
| `rerank` | Chinese-aware relevance re-scoring | ON |
| `compress` | Context truncation for token budget | ON |
| `verify` | Answer factuality check | OFF |
| `generate` | LLM answer generation (always ON) | ON |

**Illegal combination validation**: `rerank` and `compress` require `retrieve` to be ON.
The backend rejects invalid configs and the frontend blocks sending.

### Ablation Comparison
- Side-by-side comparison of two pipeline configs on the same query
- SSE streaming endpoint: `POST /api/rag/compare/stream`
- Frontend shows both answers, traces, and retrieval details

### Frontend
- Vue 3 + Vite 5
- 5 module toggle switches with real-time validation
- Pipeline config tag bar (green=ON, red=OFF)
- Retrieval detail panel: original query vs rewritten queries, pre-rerank Top-10 vs post-rerank Top-5
- Verify result panel: pass/fail + issues
- Trace timeline: skipped modules shown greyed out
- Knowledge base management: create, upload, persist
- Compare mode: two config columns, side-by-side results

## Architecture

```
Upload Document -> Save to documents/ -> Load TXT/MD/PDF/DOCX
-> Text Splitting -> HuggingFace Embedding -> Hybrid Retrieval
-> [Optional] Query Rewrite -> Merge & Deduplicate
-> [Optional] Rerank -> [Optional] Compress
-> LLM Generation -> Citation + Trace
-> [Optional] Verify
```

## Installation

### Core dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### RAG pipeline dependencies

```bash
# DeepSeek LLM (OpenAI-compatible)
pip install -r requirements-openai.txt

# HuggingFace embeddings (bge-small-zh-v1.5)
pip install -r requirements-bge.txt
```

### API server dependencies

```bash
pip install fastapi uvicorn sse-starlette python-multipart
```

### Optional dependencies

```bash
# Chroma vector database (for persistent storage)
pip install -r requirements-chroma.txt
```

### Frontend dependencies

```bash
cd frontend
npm install
```

## Model Configuration

Clean clones only include `Mock Chat`, which requires no API key and is intended
for local UI/testing flows. To use a real LLM, add a model in the frontend
model settings or create a `.env` file in the project root and reference that
environment variable from your model config.

Example DeepSeek environment variables:

```bash
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

**Never commit `.env`** - it is in `.gitignore`.

## HuggingFace Offline Mode

The embedding model (`BAAI/bge-small-zh-v1.5`) is downloaded from HuggingFace
on first run and cached locally at `~/.cache/huggingface/`. If you experience
SSL errors (common behind VPNs), set these environment variables to skip
network checks and use the cached model:

```bash
# Linux/Mac
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

To download the model for the first time, either:
- Run once without offline mode (auto-downloads to cache)
- Or manually download: `huggingface-cli download BAAI/bge-small-zh-v1.5`

## Running

### Backend

```bash
# Set environment (Windows PowerShell)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = "."
cd E:\rag\langchain-rag

python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Verify: open `http://127.0.0.1:8000/health` - should return `{"status":"ok"}`

### Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`

### Evaluation Script

```bash
$env:PYTHONPATH = "."
cd E:\rag\langchain-rag
python eval_ablation.py
```

Runs 10 fixed questions through 4 pipeline configs (Naive / Advanced /
Rewrite-only / Rerank-only), computes Hit@1/3/5/10 and MRR, outputs
comparison table and saves `eval_results.json`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/rag/query` | POST | Non-streaming RAG query |
| `/api/rag/query/stream` | POST | SSE streaming RAG query (trace + detail + chunks) |
| `/api/rag/compare/stream` | POST | SSE streaming comparison of two configs |
| `/api/knowledge-bases` | GET/POST | List / create knowledge bases |
| `/api/knowledge-bases/{id}/documents` | GET/POST | List / upload documents |
| `/api/knowledge-bases/{id}/index` | POST | Rebuild index |
| `/api/evaluations/*` | GET/POST | Evaluation endpoints |
| `/api/graphs/{kb_id}` | GET | Graph data |
| `/api/traces/{trace_id}` | GET | Trace query |

## Project Structure

```
src/
  api/              # FastAPI routes, RAG service, API models
    routes/         # chat, knowledge, evaluation, graph, trace
    real_rag_service.py    # Real RAG pipeline (DeepSeek + HF embeddings)
    llm_query_rewriter.py  # LLM-powered query rewriting
  ingestion/        # Loaders (TXT/MD/PDF/DOCX), splitter, normalizer
  retrieval/        # Embeddings, vector index, hybrid retriever, reranker
    reranker.py     # Chinese trigram reranker
    hybrid.py       # Vector + keyword hybrid retrieval
  rag/              # RAG strategies (naive, advanced, modular, agentic, graph)
    strategies/     # Strategy implementations
    service.py      # Unified RAG orchestration service
    trace.py        # TraceRecorder
  models/           # Shared data contracts (RAGRequest, RAGResult, etc.)
  graph/            # GraphRAG (NetworkX-based)
frontend/
  src/
    views/          # ChatView, KnowledgeBaseView, EvaluationView, GraphView
    api/            # API client (REST + SSE)
documents/          # Uploaded documents (gitignored)
data/               # Runtime data: knowledge_bases.json (gitignored)
eval_ablation.py    # Ablation evaluation script
```

## Documents

Place documents in `documents/` directory. Supported formats:
- PDF (`.pdf`)
- Word (`.docx`)
- Text (`.txt`)
- Markdown (`.md`)

Or upload via the frontend Knowledge Base page - files are saved to `documents/`
and the RAG index rebuilds on the next query.

## Notes

- `.env`, `data/`, `documents/`, `__pycache__/`, `eval_results.json` are gitignored
- Vite is pinned to v5 (v8 crashes on Windows due to rolldown native binding)
- Knowledge bases persist to `data/knowledge_bases.json` (survives restarts)
- The embedding model loads from HuggingFace cache after first download
