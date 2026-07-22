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

// ========== Chat Session ==========

export interface ChatSession {
  id: string
  title: string
  model_id?: string | null
  preset_id?: string | null
  rag_mode?: string | null
  knowledge_base_id?: string | null
  total_prompt_tokens: number
  total_completion_tokens: number
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  citations?: any[] | null
  trace?: any[] | null
  prompt_tokens: number
  completion_tokens: number
  latency_ms?: number | null
  created_at: string
}

export interface ChatStreamRequest {
  question: string
  rag_mode?: string | null
  knowledge_base_id?: string | null
  model_id?: string | null
  preset_id?: string | null
  top_k?: number | null
  rerank_top_k?: number | null
  score_threshold?: number | null
  temperature?: number | null
  rewrite_enabled?: boolean | null
  retrieve_enabled?: boolean | null
  rerank_enabled?: boolean | null
  compress_enabled?: boolean | null
  verify_enabled?: boolean | null
}

export interface ChatPreset {
  id: string
  name: string
  description: string
  owner_type: 'system' | 'user'
  is_default: boolean
  system_prompt?: string
  rag_prompt_hint?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface ChatPresetsResponse {
  presets: ChatPreset[]
  default_preset_id: string
}

export interface ChatSearchItem {
  session_id: string
  session_title: string
  message_id: string
  role: 'user' | 'assistant' | 'system'
  snippet: string
  created_at: string
}

export interface ChatSearchResponse {
  items: ChatSearchItem[]
  total: number
}

export interface ChatStats {
  sessions_count: number
  messages_count: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatSessionStats {
  session_id: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  messages_count: number
}

export interface ChatExportResponse {
  filename: string
  content: string
}

export interface ChatModel {
  id: string
  provider: string
  display_name: string
  model_name: string
  base_url?: string | null
  description?: string
  supports_stream: boolean
  supports_tools: boolean
  supports_vision: boolean
  enabled: boolean
  is_default: boolean
  is_builtin: boolean
  key_required: boolean
  key_configured: boolean
  key_scope: 'independent' | 'not_required'
}

export interface ChatModelsResponse {
  models: ChatModel[]
  default_model_id: string
  key_metrics?: {
    configured: number
    required: number
    missing: number
    independent: number
  }
}

export interface ChatModelPayload {
  provider: string
  display_name: string
  model_name: string
  base_url?: string | null
  api_key_env?: string | null
  description?: string
  supports_stream?: boolean
  supports_tools?: boolean
  supports_vision?: boolean
  enabled?: boolean
}

export interface ChatModelConnectionResult {
  ok: boolean
  message: string
  model_id: string
}

export type ChatStreamEvent =
  | { type: 'chunk'; content: string }
  | { type: 'trace'; trace: any[] }
  | { type: 'citation'; citations: any[] }
  | {
      type: 'done'
      message_id: string
      session_id: string
      content: string
      citations: any[]
      trace: any[]
      latency_ms: number
      prompt_tokens: number
      completion_tokens: number
      model_id?: string | null
      request_id?: string | null
      retry_count?: number
    }
  | { type: 'error'; message: string; request_id?: string | null; retry_count?: number }
