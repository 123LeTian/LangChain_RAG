/**
 * API 客户端 — 封装所有后端 REST 与 SSE 端点
 * 使用传入的 baseURL 以支持开发/生产环境切换
 */

import type {
  RAGRequest,
  RAGResult,
  KnowledgeBase,
  DocumentRecord,
  GraphData,
  EvaluationResult,
  TraceEvent,
  SSEDelta,
  SSEDone,
  ChatSession,
  ChatMessage,
  ChatStreamEvent,
  ChatStreamRequest,
  ChatModelsResponse,
  ChatPreset,
  ChatPresetsResponse,
  ChatSearchResponse,
  ChatSessionStats,
  ChatStats,
  ChatExportResponse,
} from '../types'

// ========== 默认基地址 ==========
const BASE = ''

// ========== 通用请求工具 ==========

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error?.message || `请求失败: ${res.status}`)
  }
  return res.json()
}

// ========== 健康检查 ==========

export async function healthCheck(): Promise<{ status: string; version: string }> {
  return request('/health')
}

// ========== 知识库 API ==========

/** 创建知识库 */
export async function createKnowledgeBase(name: string, description: string = ''): Promise<KnowledgeBase> {
  const form = new FormData()
  form.append('name', name)
  form.append('description', description)
  const res = await fetch(`${BASE}/api/knowledge-bases`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('创建知识库失败')
  return res.json()
}

/** 获取知识库列表 */
export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  return request('/api/knowledge-bases')
}

/** 获取单个知识库 */
export async function getKnowledgeBase(kbId: string): Promise<KnowledgeBase> {
  return request(`/api/knowledge-bases/${kbId}`)
}

/** 删除知识库 */
export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  await fetch(`${BASE}/api/knowledge-bases/${kbId}`, { method: 'DELETE' })
}

/** 上传文档 */
export async function uploadDocument(kbId: string, file: File): Promise<DocumentRecord> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/knowledge-bases/${kbId}/documents`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('上传失败')
  return res.json()
}

/** 获取文档列表 */
export async function listDocuments(kbId: string): Promise<DocumentRecord[]> {
  return request(`/api/knowledge-bases/${kbId}/documents`)
}

/** 删除文档 */
export async function deleteDocument(kbId: string, docId: string): Promise<void> {
  await fetch(`${BASE}/api/knowledge-bases/${kbId}/documents/${docId}`, { method: 'DELETE' })
}

/** 创建/重建索引 */
export async function createIndex(kbId: string): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE}/api/knowledge-bases/${kbId}/index`, { method: 'POST' })
  return res.json()
}

/** 查询索引进度 */
export async function getJobStatus(jobId: string): Promise<{ job_id: string; status: string; progress: number }> {
  return request(`/api/knowledge-bases/jobs/${jobId}`)
}

// ========== RAG 查询 API ==========

/** 非流式 RAG 查询 */
export async function ragQuery(req: RAGRequest): Promise<RAGResult> {
  return request('/api/rag/query', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

/** 流式 RAG 查询（SSE），回调分别处理 trace/chunk/done 事件 */
export async function ragQueryStream(
  req: RAGRequest,
  onTrace: (event: TraceEvent) => void,
  onChunk: (delta: string) => void,
  onDone: (data: SSEDone) => void,
  onError: (err: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await fetch(`${BASE}/api/rag/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal,
    })

    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body?.error?.message || `SSE 连接失败: ${res.status}`)
    }

    const reader = res.body?.getReader()
    if (!reader) throw new Error('无法读取响应流')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // 保留未完成行

      let eventType = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          const data = line.slice(6)
          try {
            const parsed = JSON.parse(data)
            if (eventType === 'trace') onTrace(parsed as TraceEvent)
            else if (eventType === 'chunk') onChunk((parsed as SSEDelta).delta)
            else if (eventType === 'done') onDone(parsed as SSEDone)
          } catch { /* 忽略解析失败的行 */ }
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') return // 正常取消
    onError(err)
  }
}

// ========== Chat Session API ==========

export async function createChatSession(payload: Partial<ChatSession> = {}): Promise<ChatSession> {
  return request('/api/chat/sessions', {
    method: 'POST',
    body: Object.keys(payload).length ? JSON.stringify(payload) : undefined,
  })
}

export async function listChatSessions(): Promise<ChatSession[]> {
  return request('/api/chat/sessions')
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  return request(`/api/chat/sessions/${sessionId}`)
}

export async function updateChatSession(
  sessionId: string,
  payload: Partial<Pick<ChatSession, 'title' | 'model_id' | 'preset_id' | 'rag_mode' | 'knowledge_base_id'>>,
): Promise<ChatSession> {
  return request(`/api/chat/sessions/${sessionId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  })
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/chat/sessions/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error?.message || `删除会话失败: ${res.status}`)
  }
}

export async function listChatMessages(sessionId: string): Promise<ChatMessage[]> {
  return request(`/api/chat/sessions/${sessionId}/messages`)
}

export async function listChatModels(): Promise<ChatModelsResponse> {
  return request('/api/chat/models')
}

export async function updateChatSessionModel(sessionId: string, modelId: string): Promise<ChatSession> {
  return request(`/api/chat/sessions/${sessionId}/model`, {
    method: 'PATCH',
    body: JSON.stringify({ model_id: modelId }),
  })
}

export async function listChatPresets(): Promise<ChatPresetsResponse> {
  return request('/api/chat/presets')
}

export async function createChatPreset(payload: {
  name: string
  description?: string
  system_prompt: string
  rag_prompt_hint?: string | null
}): Promise<ChatPreset> {
  return request('/api/chat/presets', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function updateChatPreset(
  presetId: string,
  payload: Partial<Pick<ChatPreset, 'name' | 'description' | 'system_prompt' | 'rag_prompt_hint'>>,
): Promise<ChatPreset> {
  return request(`/api/chat/presets/${presetId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  })
}

export async function deleteChatPreset(presetId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/chat/presets/${presetId}`, { method: 'DELETE' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error?.message || `删除预设失败: ${res.status}`)
  }
}

export async function updateChatSessionPreset(sessionId: string, presetId: string): Promise<ChatSession> {
  return request(`/api/chat/sessions/${sessionId}/preset`, {
    method: 'PATCH',
    body: JSON.stringify({ preset_id: presetId }),
  })
}

export async function searchChatMessages(params: {
  q: string
  session_id?: string
  role?: 'user' | 'assistant' | 'system'
  limit?: number
  offset?: number
}): Promise<ChatSearchResponse> {
  const query = new URLSearchParams()
  query.set('q', params.q)
  if (params.session_id) query.set('session_id', params.session_id)
  if (params.role) query.set('role', params.role)
  if (params.limit) query.set('limit', String(params.limit))
  if (params.offset) query.set('offset', String(params.offset))
  return request(`/api/chat/search?${query.toString()}`)
}

export async function getChatStats(): Promise<ChatStats> {
  return request('/api/chat/stats')
}

export async function getChatSessionStats(sessionId: string): Promise<ChatSessionStats> {
  return request(`/api/chat/sessions/${sessionId}/stats`)
}

export async function exportChatSession(sessionId: string): Promise<ChatExportResponse> {
  return request(`/api/chat/sessions/${sessionId}/export`)
}

export async function chatSessionStream(
  sessionId: string,
  req: ChatStreamRequest,
  onEvent: (event: ChatStreamEvent) => void,
  onError: (err: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await fetch(`${BASE}/api/chat/sessions/${sessionId}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal,
    })

    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body?.error?.message || `SSE 连接失败: ${res.status}`)
    }

    const reader = res.body?.getReader()
    if (!reader) throw new Error('无法读取响应流')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          onEvent(JSON.parse(line.slice(6)) as ChatStreamEvent)
        } catch {
          // Ignore malformed SSE rows.
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') return
    onError(err)
  }
}

// ========== Trace API ==========

export async function getTrace(traceId: string): Promise<TraceEvent[]> {
  return request(`/api/traces/${traceId}`)
}

// ========== Graph API ==========

export async function getGraphData(kbId: string): Promise<GraphData> {
  return request(`/api/graphs/${kbId}`)
}

// ========== Evaluation API ==========

export async function runEvaluation(kbId: string, modes: string[]): Promise<{ run_id: string }> {
  return request('/api/evaluations/run', {
    method: 'POST',
    body: JSON.stringify({ kb_id: kbId, modes }),
  })
}

export async function getEvaluationResult(runId: string): Promise<EvaluationResult[]> {
  return request(`/api/evaluations/${runId}`)
}
