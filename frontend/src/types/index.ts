/**
 * 前端类型定义 — 与后端 src/models/ 中的 Pydantic 模型对应
 * 后续可从 OpenAPI 自动生成，目前手工维护
 */

// ========== 枚举 ==========

/** RAG 运行模式 */
export type RAGMode = 'naive' | 'advanced' | 'modular' | 'graph' | 'agentic'

/** Trace 执行阶段 */
export type TraceStage =
  | 'intent' | 'rewrite' | 'retrieve' | 'rerank' | 'compress'
  | 'graph_search' | 'tool_call' | 'generate' | 'verify' | 'complete' | 'error'

// ========== 检索与引用 ==========

/** 统一召回结果 */
export interface RetrievalHit {
  chunk_id: string
  text: string
  score: number
  rank: number
  retriever: string
  metadata: Record<string, any>
}

/** 最终引用 */
export interface Citation {
  document_id: string
  chunk_id: string
  filename: string
  page?: number
  quote: string
  score: number
}

// ========== RAG 请求/结果 ==========

/** 统一 RAG 请求 */
export interface RAGRequest {
  query: string
  kb_id: string
  mode: RAGMode
  session_id?: string
  options: Record<string, any>
}

/** 统一 RAG 结果 */
export interface RAGResult {
  answer: string
  citations: Citation[]
  hits: RetrievalHit[]
  trace: TraceEvent[]
  usage: Record<string, any>
  warnings: string[]
  mode: RAGMode
}

// ========== Trace ==========

/** 单次 Trace 事件 */
export interface TraceEvent {
  trace_id: string
  stage: TraceStage
  started_at: string
  duration_ms: number
  input_summary: string
  output_summary: string
  metadata: Record<string, any>
}

// ========== 知识库 ==========

export type KBStatus = 'creating' | 'ready' | 'indexing' | 'error'
export type DocStatus = 'uploaded' | 'parsing' | 'chunking' | 'indexed' | 'error'
export type DocType = 'txt' | 'md' | 'pdf' | 'docx'

export interface KnowledgeBase {
  id: string
  owner_id: string
  name: string
  description: string
  status: KBStatus
  embedding_model: string
  doc_count: number
  chunk_count: number
  created_at: string
}

export interface DocumentRecord {
  id: string
  kb_id: string
  filename: string
  type: DocType
  checksum: string
  status: DocStatus
  chunk_count: number
  size_bytes: number
}

// ========== 图谱 ==========

export interface GraphNode {
  id: string
  name: string
  type: 'entity' | 'community'
  chunk_id: string
  label: string
}

export interface GraphLink {
  source: string
  target: string
  relation: string
  chunk_id: string
}

export interface Community {
  id: string
  name: string
  summary: string
  entity_count: number
  relation_count: number
}

export interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
  communities: Community[]
  hit_path: string[]
}

// ========== 评测 ==========

export interface RetrievalMetrics {
  hit_at_1: number
  hit_at_3: number
  hit_at_5: number
  hit_at_10: number
  mrr: number
}

export interface SystemMetrics {
  first_token_latency_ms: number
  total_latency_ms: number
  total_tokens: number
  estimated_cost_usd: number
}

export interface EvaluationResult {
  run_id: string
  mode: RAGMode
  metrics: {
    retrieval: RetrievalMetrics
    system: SystemMetrics
  }
  sample_count: number
  latency_ms: number
}

// ========== SSE 事件 ==========

/** SSE chunk 事件承载的 token 数据 */
export interface SSEDelta {
  delta: string
}

/** SSE done 事件承载的最终数据 */
export interface SSEDone {
  trace_id: string
  citations: Citation[]
  usage: Record<string, any>
}
