/**
 * API client - all backend REST and SSE endpoints
 */

import type {
  RAGRequest, RAGResult, KnowledgeBase, DocumentRecord,
  GraphData, EvaluationResult, TraceEvent, SSEDelta, SSEDone,
} from '../types'

const BASE = ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.error?.message || `request failed: ${res.status}`)
  }
  return res.json()
}

export async function healthCheck(): Promise<{ status: string; version: string }> {
  return request('/health')
}

export async function createKnowledgeBase(name: string, description: string = ''): Promise<KnowledgeBase> {
  const form = new FormData()
  form.append('name', name)
  form.append('description', description)
  const res = await fetch(`${BASE}/api/knowledge-bases`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('create KB failed')
  return res.json()
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  return request('/api/knowledge-bases')
}

export async function getKnowledgeBase(kbId: string): Promise<KnowledgeBase> {
  return request(`/api/knowledge-bases/${kbId}`)
}

export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  await fetch(`${BASE}/api/knowledge-bases/${kbId}`, { method: 'DELETE' })
}

export async function uploadDocument(kbId: string, file: File): Promise<DocumentRecord> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/knowledge-bases/${kbId}/documents`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('upload failed')
  return res.json()
}

export async function listDocuments(kbId: string): Promise<DocumentRecord[]> {
  return request(`/api/knowledge-bases/${kbId}/documents`)
}

export async function deleteDocument(kbId: string, docId: string): Promise<void> {
  await fetch(`${BASE}/api/knowledge-bases/${kbId}/documents/${docId}`, { method: 'DELETE' })
}

export async function createIndex(kbId: string): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE}/api/knowledge-bases/${kbId}/index`, { method: 'POST' })
  return res.json()
}

export async function getJobStatus(jobId: string): Promise<{ job_id: string; status: string; progress: number }> {
  return request(`/api/knowledge-bases/jobs/${jobId}`)
}

export async function ragQuery(req: RAGRequest): Promise<RAGResult> {
  return request('/api/rag/query', { method: 'POST', body: JSON.stringify(req) })
}

/** SSE streaming with trace, detail, chunk, and done callbacks */
export async function ragQueryStream(
  req: RAGRequest,
  onTrace: (event: any) => void,
  onChunk: (delta: string) => void,
  onDone: (data: SSEDone) => void,
  onError: (err: Error) => void,
  signal?: AbortSignal,
  onDetail?: (detail: any) => void,
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
      throw new Error(body?.error?.message || `SSE failed: ${res.status}`)
    }
    const reader = res.body?.getReader()
    if (!reader) throw new Error('cannot read response stream')
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      let eventType = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          try {
            const parsed = JSON.parse(line.slice(6))
            if (eventType === 'trace') onTrace(parsed)
            else if (eventType === 'detail' && onDetail) onDetail(parsed)
            else if (eventType === 'chunk') onChunk(parsed.delta)
            else if (eventType === 'done') onDone(parsed)
          } catch { /* ignore */ }
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') return
    onError(err)
  }
}

/** SSE streaming comparison - runs same query with two configs */
export async function ragCompareStream(
  req: { query: string; kb_id: string; mode: string; config_a: any; config_b: any },
  signal: AbortSignal,
  onConfigA: (data: any) => void,
  onConfigB: (data: any) => void,
): Promise<void> {
  try {
    const res = await fetch(`${BASE}/api/rag/compare/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal,
    })
    if (!res.ok) throw new Error(`Compare SSE failed: ${res.status}`)
    const reader = res.body?.getReader()
    if (!reader) throw new Error('cannot read response stream')
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      let eventType = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          try {
            const parsed = JSON.parse(line.slice(6))
            if (eventType === 'config_a') onConfigA(parsed)
            else if (eventType === 'config_b') onConfigB(parsed)
          } catch { /* ignore */ }
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') return
    throw err
  }
}

export async function getTrace(traceId: string): Promise<TraceEvent[]> {
  return request(`/api/traces/${traceId}`)
}

export async function getGraphData(kbId: string): Promise<GraphData> {
  return request(`/api/graphs/${kbId}`)
}

export async function runEvaluation(kbId: string, modes: string[]): Promise<{ run_id: string }> {
  return request('/api/evaluations/run', { method: 'POST', body: JSON.stringify({ kb_id: kbId, modes }) })
}

export async function getEvaluationResult(runId: string): Promise<EvaluationResult[]> {
  return request(`/api/evaluations/${runId}`)
}