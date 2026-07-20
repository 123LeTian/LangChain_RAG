# RAG Architecture

> Owner: C (主责), 全组评审

## Pipeline Overview

```
Documents → Ingestion (B) → Retrieval (B) → Augmentation (C) → Generation (C)
                                       ↓
                              Graph (D) → Community Reports
                                       ↓
                              Agent (C) → Multi-step Reasoning
                                       ↓
                              Evaluation (D) → Metrics & Reports
```

## Module Ownership

| Module | Owner | Responsibility |
|--------|-------|---------------|
| `src/api/` | A | REST API layer, no RAG logic |
| `src/ingestion/` | B | Document loading, splitting, indexing |
| `src/retrieval/` | B | Embeddings, vector search, hybrid retrieval |
| `src/rag/` | C | Core RAG orchestration, strategy registry |
| `src/graph/` | D | Knowledge graph extraction & retrieval |
| `src/agent/` | C | Agentic RAG workflows |
| `src/evaluation/` | D | Metrics, datasets, evaluation runner |

## Strategy Taxonomy

| Strategy | File | Owner | Description |
|----------|------|-------|-------------|
| Naive | `strategies/naive.py` | B | Simple retrieve-then-generate |
| Advanced | `strategies/advanced.py` | B | Rerank + rewrite + compress |
| Modular | `strategies/modular.py` | C | Composable pipeline |
| Graph RAG | `strategies/graph_rag.py` | D | Knowledge-graph-enhanced |
| Agentic | `strategies/agentic.py` | C | Agent-driven multi-step |
