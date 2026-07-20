# API Contract

> Owner: A (API 层), 全组评审

## Overview

This document defines the REST API contract for the langchain-rag service.

## Endpoints

### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send a chat message |
| GET | `/api/chat/{session_id}` | Get chat history |

### Knowledge Base

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/knowledge/upload` | Upload documents |
| GET | `/api/knowledge/list` | List documents |
| DELETE | `/api/knowledge/{doc_id}` | Delete a document |

### Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/entities` | Query entities |
| GET | `/api/graph/communities` | List communities |

### Trace

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/trace/{run_id}` | Get trace details |
| GET | `/api/trace/list` | List recent traces |

### Evaluation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/evaluation/run` | Start an evaluation run |
| GET | `/api/evaluation/{run_id}` | Get evaluation results |
