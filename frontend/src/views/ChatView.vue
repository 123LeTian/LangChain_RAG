<template>
  <section class="chat-view" :class="{ 'drawer-open': drawerOpen }">
    <main class="chat-main-panel">
      <div ref="msgContainer" class="chat-scroll">
        <div class="message-stack">
          <div v-if="messages.length === 0" class="welcome-panel">
            <h1>问答实验台</h1>
            <p>在底部选择模型和 RAG 模式后开始提问。</p>
          </div>
          <div v-if="errorText" class="chat-error">{{ errorText }}</div>

          <article v-for="(msg, index) in messages" :key="msg.id || index" :class="['message', msg.role]">
            <template v-if="msg.role === 'user'">
              <div class="message-label">你</div>
              <div class="message-body">{{ msg.content }}</div>
            </template>

            <div v-else class="ai-message-card">
              <header class="ai-message-header">
                <span class="ai-avatar">AI</span>
                <strong>{{ msg.modelLabel || currentModelLabel }} · {{ msg.modeLabel || currentModeLabel }}</strong>
              </header>

              <button
                v-if="msg.thinking || msg.trace?.length || msg.citations?.length"
                class="minimal-summary-bar"
                type="button"
                @click="msg.summaryOpen = !msg.summaryOpen"
              >
                <span :class="['thinking-pulse', { active: msg.thinking }]" />
                <span>{{ summaryText(msg) }}</span>
                <strong v-if="msg.citations?.length">{{ msg.summaryOpen ? '收起片段' : '查看片段' }}</strong>
              </button>

              <div v-if="msg.summaryOpen && msg.citations?.length" class="citation-preview-list">
                <article v-for="(citation, citationIndex) in msg.citations.slice(0, 3)" :key="citation.chunk_id || citationIndex">
                  <strong>{{ citation.filename || '未知来源' }}</strong>
                  <p>{{ citation.quote || citation.text || '暂无片段内容' }}</p>
                </article>
              </div>

              <div v-if="msg.stopped" class="stopped-notice">对话已停止</div>
              <div v-if="msg.content" class="message-body markdown-body" v-html="renderMarkdown(msg.content)" />

              <footer v-if="msg.content" class="ai-action-footer">
                <div class="rag-actions">
                  <button
                    v-if="msg.citations?.length"
                    class="citation-action"
                    type="button"
                    @click="openDrawer(msg, 'citations')"
                  >
                    <span>📄</span>
                    引用来源
                    <strong>{{ msg.citations.length }}</strong>
                  </button>

                  <button
                    v-if="msg.trace?.length"
                    class="trace-action"
                    type="button"
                    @click="openDrawer(msg, 'trace')"
                  >
                    <span>⚡</span>
                    执行轨迹
                    <small>{{ totalDuration(msg.trace) }}ms · {{ msg.trace.length }} 阶段</small>
                  </button>
                  <span v-if="messageTokenTotal(msg)" class="message-token-chip">
                    🪙 {{ messageTokenLabel(msg) }}
                  </span>
                </div>

                <div class="quick-actions">
                  <button type="button" title="复制内容" @click="copyMessage(msg)">📋</button>
                  <button type="button" title="重新生成" @click="regenerateFrom(index)">🔄</button>
                </div>
              </footer>
            </div>
          </article>
        </div>
      </div>

      <div class="composer-shell">
        <div class="control-pill-bar" aria-label="生成与 RAG 配置">
          <div class="popover-control control-slot control-slot-preset">
            <button class="control-pill" type="button" @click="toggleMenu('preset')">
              <span>🎯 预设</span>
              <strong>{{ currentPresetLabel }}</strong>
            </button>
            <div v-if="openMenu === 'preset'" class="option-popover">
              <button
                v-for="item in presetOptions"
                :key="item.value"
                :class="['option-item', { selected: generationSettings.preset === item.value }]"
                type="button"
                @click="choosePreset(item.value)"
              >
                <span>{{ item.label }}</span>
                <strong v-if="generationSettings.preset === item.value">✓</strong>
              </button>
            </div>
          </div>

          <div class="popover-control control-slot control-slot-model">
            <button class="control-pill" type="button" @click="toggleMenu('model')">
              <span>🤖 模型</span>
              <strong>{{ currentModelLabel }}</strong>
            </button>
            <div v-if="openMenu === 'model'" class="option-popover">
              <button
                v-for="item in modelOptions"
                :key="item.value"
                :class="['option-item', { selected: generationSettings.model === item.value }]"
                type="button"
                @click="chooseModel(item.value)"
              >
                <span>{{ item.label }}</span>
                <strong v-if="generationSettings.model === item.value">✓</strong>
              </button>
            </div>
          </div>

          <div class="popover-control control-slot control-slot-rag">
            <button class="control-pill rag-mode-pill" type="button" @click="toggleMenu('rag')">
              <span>⚡ RAG 模式</span>
              <strong>{{ currentModeLabel }}</strong>
            </button>
            <div v-if="openMenu === 'rag'" class="rag-popover">
              <div class="rag-mode-list">
                <button
                  v-for="mode in ragModes"
                  :key="mode.value"
                  :class="['option-item', { selected: props.chatSettings.mode === mode.value }]"
                  type="button"
                  @click="chooseRagMode(mode.value)"
                >
                  <span>{{ mode.label }}</span>
                  <strong v-if="props.chatSettings.mode === mode.value">✓</strong>
                </button>
              </div>

              <div class="rag-params-panel">
                <header>
                  <strong>{{ currentModeLabel }}</strong>
                  <button type="button" title="关闭" @click="openMenu = ''">×</button>
                </header>

                <p v-if="props.chatSettings.mode === 'direct'" class="direct-mode-note">
                  直接调用当前模型，不检索知识库。
                </p>

                <template v-else>
                  <label class="slider-field">
                    <span>Top-K：{{ props.chatSettings.topK }}</span>
                    <input v-model.number="props.chatSettings.topK" type="range" min="1" max="20" />
                  </label>

                  <div v-if="props.chatSettings.mode === 'graph'" class="segmented-field">
                    <span>Graph Scope</span>
                    <div>
                      <button
                        v-for="scope in graphScopeOptions"
                        :key="scope.value"
                        :class="{ active: props.chatSettings.graphScope === scope.value }"
                        type="button"
                        @click="props.chatSettings.graphScope = scope.value"
                      >
                        {{ scope.label }}
                      </button>
                    </div>
                  </div>

                  <label v-if="props.chatSettings.mode === 'naive'" class="slider-field">
                    <span>Score Threshold：{{ props.chatSettings.scoreThreshold.toFixed(2) }}</span>
                    <input v-model.number="props.chatSettings.scoreThreshold" type="range" min="0" max="1" step="0.05" />
                  </label>

                      <template v-if="props.chatSettings.mode === 'advanced'">
                        <label class="switch-field">
                          <span>Query Rewrite</span>
                          <input type="checkbox" checked disabled />
                        </label>
                        <label class="switch-field">
                          <span>Retrieve</span>
                          <input type="checkbox" checked disabled />
                        </label>
                        <label class="switch-field">
                          <span>Cross-Encoder Rerank</span>
                          <input type="checkbox" checked disabled />
                        </label>
                        <label class="switch-field">
                          <span>Compress</span>
                          <input type="checkbox" checked disabled />
                        </label>
                        <label class="switch-field">
                          <span>Verify</span>
                          <input v-model="props.chatSettings.verifyEnabled" type="checkbox" />
                        </label>
                        <label class="slider-field">
                          <span>Score Threshold：{{ props.chatSettings.scoreThreshold.toFixed(2) }}</span>
                          <input v-model.number="props.chatSettings.scoreThreshold" type="range" min="0" max="1" step="0.05" />
                        </label>
                        <label class="slider-field">
                          <span>Rerank Top-K：{{ props.chatSettings.rerankTopK }}</span>
                          <input v-model.number="props.chatSettings.rerankTopK" type="range" min="1" max="20" />
                        </label>
                      </template>

                      <template v-else-if="props.chatSettings.mode === 'modular'">
                        <label class="switch-field">
                          <span>Query Rewrite</span>
                          <input v-model="props.chatSettings.rewriteEnabled" type="checkbox" />
                        </label>
                        <label class="switch-field">
                          <span>Retrieve</span>
                          <input v-model="props.chatSettings.retrieveEnabled" type="checkbox" />
                        </label>
                        <label class="switch-field">
                          <span>Cross-Encoder Rerank</span>
                          <input v-model="props.chatSettings.rerankEnabled" type="checkbox" />
                        </label>
                        <label class="switch-field">
                          <span>Compress</span>
                          <input v-model="props.chatSettings.compressEnabled" type="checkbox" />
                        </label>
                        <label class="switch-field">
                          <span>Verify</span>
                          <input v-model="props.chatSettings.verifyEnabled" type="checkbox" />
                        </label>
                        <label class="slider-field">
                          <span>Score Threshold：{{ props.chatSettings.scoreThreshold.toFixed(2) }}</span>
                          <input v-model.number="props.chatSettings.scoreThreshold" type="range" min="0" max="1" step="0.05" />
                        </label>
                        <label class="slider-field">
                          <span>Rerank Top-K：{{ props.chatSettings.rerankTopK }}</span>
                          <input v-model.number="props.chatSettings.rerankTopK" type="range" min="1" max="20" />
                        </label>
                      </template>

                      <template v-if="props.chatSettings.mode === 'agentic'">
                        <label class="slider-field">
                          <span>Max Steps：{{ props.chatSettings.maxSteps }}</span>
                          <input v-model.number="props.chatSettings.maxSteps" type="range" min="4" max="12" />
                        </label>
                        <label class="switch-field">
                          <span>Vector Tool</span>
                          <input v-model="agentTools.vector" type="checkbox" />
                        </label>
                        <label class="switch-field">
                          <span>Graph Tool</span>
                          <input v-model="agentTools.graph" type="checkbox" />
                        </label>
                      </template>
                </template>
              </div>
            </div>
          </div>

          <div v-if="props.chatSettings.mode !== 'direct'" class="popover-control control-slot control-slot-kb">
            <button class="control-pill kb-pill" type="button" @click="toggleMenu('kb')">
              <span>📚 知识库</span>
              <strong>{{ currentKbLabel }}</strong>
            </button>
            <div v-if="openMenu === 'kb'" class="option-popover option-popover-right">
              <button
                v-for="item in kbOptions"
                :key="item.value"
                :class="['option-item', { selected: props.chatSettings.kbId === item.value }]"
                type="button"
                @click="chooseKnowledgeBase(item.value)"
              >
                <span>{{ item.label }}</span>
                <strong v-if="props.chatSettings.kbId === item.value">✓</strong>
              </button>
            </div>
          </div>
        </div>

        <form class="composer" @submit.prevent="sendQuery">
          <textarea
            v-model="query"
            :disabled="streaming"
            rows="1"
            placeholder="输入消息..."
            @keydown.enter.exact.prevent="sendQuery"
          />
          <button v-if="streaming" class="stop-button" type="button" @click="cancelQuery">停止</button>
          <button class="send-button" type="submit" :disabled="!query.trim() || streaming" title="发送">↑</button>
        </form>
      </div>
    </main>

    <aside class="debug-drawer" :class="{ open: drawerOpen }" aria-label="调试详情">
      <div class="debug-drawer-inner">
        <div class="drawer-header">
          <div>
            <strong>调试详情</strong>
            <span>{{ activeMessage ? totalDuration(activeMessage.trace || []) : 0 }}ms</span>
          </div>
          <button class="icon-button" type="button" title="关闭" @click="drawerOpen = false">×</button>
        </div>

        <div class="drawer-tabs">
          <button :class="{ active: activeTab === 'citations' }" type="button" @click="activeTab = 'citations'">引用溯源</button>
          <button :class="{ active: activeTab === 'trace' }" type="button" @click="activeTab = 'trace'">执行轨迹</button>
          <button :class="{ active: activeTab === 'raw' }" type="button" @click="activeTab = 'raw'">Raw JSON</button>
        </div>

        <div class="drawer-content">
          <div v-if="activeTab === 'citations'" class="citation-list">
            <article v-for="(citation, index) in activeMessage?.citations || []" :key="citation.chunk_id || index">
              <div class="chunk-meta">
                <strong>{{ citation.filename || '未知来源' }}</strong>
                <span>{{ citation.page ? `p.${citation.page}` : '片段' }}</span>
              </div>
              <p>{{ citation.quote || citation.text || '暂无片段内容' }}</p>
              <small>Chunk {{ citation.chunk_id || index + 1 }} · score {{ formatScore(citation.score) }}</small>
            </article>
            <p v-if="!(activeMessage?.citations || []).length" class="drawer-empty">暂无引用来源。</p>
          </div>

          <div v-else-if="activeTab === 'trace'" class="trace-list">
            <article v-for="(trace, index) in activeMessage?.trace || []" :key="`${trace.stage}-${index}`">
              <div class="trace-dot" />
              <div>
                <div class="trace-head">
                  <strong>{{ trace.stage || `阶段 ${index + 1}` }}</strong>
                  <span>{{ trace.duration_ms || 0 }}ms</span>
                </div>
                <p>{{ trace.output_summary || trace.input_summary || '无摘要' }}</p>
              </div>
            </article>
            <p v-if="!(activeMessage?.trace || []).length" class="drawer-empty">暂无执行轨迹。</p>
          </div>

          <pre v-else>{{ JSON.stringify(activeMessage || {}, null, 2) }}</pre>
        </div>
      </div>
    </aside>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import {
  chatSessionStream,
  createChatSession,
  getChatSession,
  listChatModels,
  listChatMessages,
  listChatPresets,
  updateChatSession,
  updateChatSessionModel,
  updateChatSessionPreset,
} from '../api/client'

const props = defineProps({
  chatSettings: {
    type: Object,
    required: true,
  },
  knowledgeBases: {
    type: Array,
    default: () => [],
  },
  modelOptions: {
    type: Array,
    default: () => [],
  },
  newChatSignal: {
    type: Number,
    default: 0,
  },
  currentSessionId: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['session-created', 'sessions-updated'])

const ragModes = [
  { value: 'direct', label: '无 RAG' },
  { value: 'naive', label: 'Naive RAG' },
  { value: 'advanced', label: 'Advanced RAG' },
  { value: 'modular', label: 'Modular RAG' },
  { value: 'graph', label: 'GraphRAG' },
  { value: 'agentic', label: 'Agentic RAG' },
]

const graphScopeOptions = [
  { value: 'auto', label: 'Auto' },
  { value: 'local', label: 'Local' },
  { value: 'global', label: 'Global' },
]

const presetOptions = reactive([
  { value: 'research', label: '严谨研报' },
  { value: 'summary', label: '极简摘要' },
  { value: 'structured', label: '代码/结构化' },
])

const generationSettings = ref({
  preset: 'default-assistant',
  model: 'mock-chat',
})
const agentTools = ref({
  vector: true,
  graph: true,
})

const query = ref('')
const messages = ref([])
const streaming = ref(false)
const msgContainer = ref(null)
const drawerOpen = ref(false)
const openMenu = ref('')
const activeTab = ref('citations')
const activeMessage = ref(null)
const errorText = ref('')
const activeStreamSessionId = ref('')
const modelOptions = ref([])
const defaultModelId = ref('mock-chat')
const defaultPresetId = ref('default-assistant')

let abortCtrl = null

function normalizeMessage(message) {
  return {
    id: message.id,
    session_id: message.session_id,
    role: message.role,
    content: message.content || '',
    citations: message.citations || [],
    trace: message.trace || [],
    latency_ms: message.latency_ms || 0,
    prompt_tokens: message.prompt_tokens || 0,
    completion_tokens: message.completion_tokens || 0,
    created_at: message.created_at,
    isStreaming: false,
    thinking: false,
    stopped: false,
    summaryOpen: false,
    mode: props.chatSettings.mode,
    modelLabel: currentModelLabel.value,
    modeLabel: currentModeLabel.value,
  }
}

async function loadSession(sessionId) {
  if (!sessionId) {
    messages.value = []
    activeMessage.value = null
    drawerOpen.value = false
    return
  }

  try {
    errorText.value = ''
    const [session, loadedMessages] = await Promise.all([
      getChatSession(sessionId),
      listChatMessages(sessionId),
    ])
    if (session.rag_mode) props.chatSettings.mode = session.rag_mode
    generationSettings.value.model = session.model_id || defaultModelId.value
    generationSettings.value.preset = session.preset_id || defaultPresetId.value
    props.chatSettings.modelId = generationSettings.value.model
    props.chatSettings.presetId = generationSettings.value.preset
    if (session.knowledge_base_id !== null && session.knowledge_base_id !== undefined) {
      props.chatSettings.kbId = session.knowledge_base_id || ''
    }
    messages.value = loadedMessages
      .filter(message => message.role === 'user' || message.role === 'assistant')
      .map(normalizeMessage)
    if (activeMessage.value) {
      activeMessage.value = messages.value.find(message => message.id === activeMessage.value?.id) || null
    }
    await scrollToBottom()
  } catch (err) {
    errorText.value = err?.message || '加载历史失败'
  }
}

async function ensureSession() {
  if (props.currentSessionId) return props.currentSessionId

  const session = await createChatSession({
    title: '新对话',
    model_id: generationSettings.value.model,
    preset_id: generationSettings.value.preset,
    rag_mode: props.chatSettings.mode,
    knowledge_base_id: props.chatSettings.kbId || null,
  })
  activeStreamSessionId.value = session.id
  emit('session-created', session)
  return session.id
}

async function sendQuery() {
  const q = query.value.trim()
  if (!q || streaming.value) return

  let sessionId = ''
  try {
    sessionId = await ensureSession()
  } catch (err) {
    errorText.value = err?.message || '创建会话失败'
    return
  }

  errorText.value = ''
  messages.value.push({
    id: `local_user_${Date.now()}`,
    role: 'user',
    content: q,
    citations: [],
    trace: [],
  })
  const assistantMessage = {
    id: `local_assistant_${Date.now()}`,
    role: 'assistant',
    content: '',
    citations: [],
    trace: [],
    latency_ms: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    request_id: '',
    retry_count: 0,
    isStreaming: true,
    thinking: true,
    stopped: false,
    summaryOpen: false,
    mode: props.chatSettings.mode,
    modelLabel: currentModelLabel.value,
    modeLabel: currentModeLabel.value,
  }
  messages.value.push(assistantMessage)
  query.value = ''
  streaming.value = true
  activeStreamSessionId.value = sessionId
  abortCtrl = new AbortController()
  await scrollToBottom()

  await chatSessionStream(
    sessionId,
    {
      question: q,
      rag_mode: props.chatSettings.mode,
      knowledge_base_id: props.chatSettings.kbId || null,
      model_id: generationSettings.value.model,
      preset_id: generationSettings.value.preset,
      top_k: props.chatSettings.topK,
      graph_scope: props.chatSettings.mode === 'graph' ? props.chatSettings.graphScope : null,
      max_steps: props.chatSettings.mode === 'agentic' ? props.chatSettings.maxSteps : null,
      agent_vector_enabled: props.chatSettings.mode === 'agentic' ? agentTools.value.vector : null,
      agent_graph_enabled: props.chatSettings.mode === 'agentic' ? agentTools.value.graph : null,
      rerank_top_k: props.chatSettings.mode === 'graph' ? null : props.chatSettings.rerankTopK,
      score_threshold: props.chatSettings.mode === 'graph' ? null : props.chatSettings.scoreThreshold,
      temperature: null,
      rewrite_enabled: props.chatSettings.mode === 'graph' ? null : (props.chatSettings.mode === 'advanced' ? true : props.chatSettings.rewriteEnabled),
      retrieve_enabled: props.chatSettings.mode === 'graph' ? null : (props.chatSettings.mode === 'advanced' ? true : props.chatSettings.retrieveEnabled),
      rerank_enabled: props.chatSettings.mode === 'graph' ? null : (props.chatSettings.mode === 'advanced' ? true : props.chatSettings.rerankEnabled),
      compress_enabled: props.chatSettings.mode === 'graph' ? null : (props.chatSettings.mode === 'advanced' ? true : props.chatSettings.compressEnabled),
      verify_enabled: props.chatSettings.mode === 'graph' ? null : props.chatSettings.verifyEnabled,
    },
    async (event) => {
      if (event.type === 'chunk') {
        assistantMessage.content += event.content || ''
      } else if (event.type === 'trace') {
        assistantMessage.trace = event.trace || []
      } else if (event.type === 'citation') {
        assistantMessage.citations = event.citations || []
      } else if (event.type === 'done') {
        assistantMessage.id = event.message_id || assistantMessage.id
        assistantMessage.session_id = event.session_id
        assistantMessage.content = event.content || assistantMessage.content
        assistantMessage.citations = event.citations || []
        assistantMessage.trace = event.trace || []
        assistantMessage.latency_ms = event.latency_ms || 0
        assistantMessage.prompt_tokens = event.prompt_tokens || 0
        assistantMessage.completion_tokens = event.completion_tokens || 0
        assistantMessage.request_id = event.request_id || ''
        assistantMessage.retry_count = event.retry_count || 0
        assistantMessage.isStreaming = false
        assistantMessage.thinking = false
        assistantMessage.summaryOpen = false
        streaming.value = false
        emit('sessions-updated')
      } else if (event.type === 'error') {
        const requestHint = event.request_id ? `（request_id: ${event.request_id}）` : ''
        errorText.value = `${event.message || '请求失败'}${requestHint}`
        assistantMessage.request_id = event.request_id || ''
        assistantMessage.retry_count = event.retry_count || 0
        assistantMessage.content = event.message ? `请求失败：${event.message}${requestHint}` : '请求失败'
        assistantMessage.isStreaming = false
        assistantMessage.thinking = false
        assistantMessage.summaryOpen = false
        streaming.value = false
      }
      await scrollToBottom()
    },
    (err) => {
      errorText.value = err?.message || '请求失败'
      assistantMessage.content = `请求失败：${err?.message || '未知错误'}`
      assistantMessage.isStreaming = false
      assistantMessage.thinking = false
      assistantMessage.summaryOpen = false
      streaming.value = false
    },
    abortCtrl.signal,
  )

  streaming.value = false
  activeStreamSessionId.value = ''
  assistantMessage.isStreaming = false
  assistantMessage.thinking = false
  assistantMessage.summaryOpen = false
  await scrollToBottom()
}

function cancelQuery() {
  abortCtrl?.abort()
  const currentAssistant = [...messages.value].reverse().find(message => message.role === 'assistant')
  if (currentAssistant) {
    currentAssistant.thinking = false
    currentAssistant.isStreaming = false
    currentAssistant.stopped = true
    currentAssistant.summaryOpen = false
  }
  streaming.value = false
  activeStreamSessionId.value = ''
}

function openDrawer(message, tab) {
  activeMessage.value = message
  activeTab.value = tab
  drawerOpen.value = true
}

function totalDuration(trace = []) {
  return trace.reduce((sum, item) => sum + Number(item.duration_ms || 0), 0)
}

function formatScore(score) {
  return typeof score === 'number' ? score.toFixed(2) : '-'
}

function summaryText(message) {
  if (isDirectMessage(message)) {
    if (message.thinking) return '🧠 正在思考...'
    return `🧠 思考完成 (${totalDuration(message.trace)}ms)`
  }
  if (message.thinking) return '🧠 正在思考与检索知识库...'
  const requestedCount = requestedRetrievalCount(message.trace)
  const candidateCount = retrievedCandidateCount(message.trace) || message.citations?.length || 0
  const compressedCount = compressedChunkCount(message.trace)
  const retrievalSummary = requestedCount
    ? `已请求 Top-${requestedCount}${candidateCount ? ` · 候选 ${candidateCount} 条` : ''}`
    : candidateCount
      ? `已检索候选 ${candidateCount} 条`
      : '已检索知识库'
  const compressionSummary = compressedCount ? ` · 压缩后 ${compressedCount} 条` : ''
  return `🧠 ${retrievalSummary}${compressionSummary} · 思考完成 (${totalDuration(message.trace)}ms)`
}

function isDirectMessage(message) {
  if (message?.mode === 'direct') return true
  if (String(message?.modeLabel || '').includes('无 RAG')) return true
  return (message?.trace || []).some(item => {
    const summary = `${item?.input_summary || ''} ${item?.output_summary || ''}`.toLowerCase()
    return summary.includes('direct chat') || summary.includes('no rag')
  })
}

function requestedRetrievalCount(trace = []) {
  const retrieveEvent = trace.find(item => String(item.stage || '').toLowerCase().includes('retrieve'))
  const text = `${retrieveEvent?.output_summary || ''} ${retrieveEvent?.input_summary || ''}`
  const match = text.match(/(?:requested\s+)?top[_-]?k[=:]\s*(\d+)/i)
  return match ? Number(match[1]) : 0
}

function retrievedCandidateCount(trace = []) {
  const retrieveEvent = trace.find(item => String(item.stage || '').toLowerCase().includes('retrieve'))
  const text = `${retrieveEvent?.output_summary || ''} ${retrieveEvent?.input_summary || ''}`
  const match = text.match(/pre-rerank candidates=(\d+)/i) || text.match(/pre-rerank Top-(\d+)/i)
  return match ? Number(match[1]) : 0
}

function compressedChunkCount(trace = []) {
  const compressEvent = trace.find(item => String(item.stage || '').toLowerCase().includes('compress'))
  const text = `${compressEvent?.output_summary || ''} ${compressEvent?.input_summary || ''}`
  const match = text.match(/compressed to (\d+) chunks/i)
  return match ? Number(match[1]) : 0
}

function renderMarkdown(value = '') {
  const lines = String(value || '').replace(/\r\n/g, '\n').split('\n')
  const blocks = []
  const paragraph = []
  const listItems = []

  const flushParagraph = () => {
    if (!paragraph.length) return
    blocks.push(`<p>${paragraph.join('<br>')}</p>`)
    paragraph.length = 0
  }

  const flushList = () => {
    if (!listItems.length) return
    blocks.push(`<ul class="md-list">${listItems.map(item => `<li>${item}</li>`).join('')}</ul>`)
    listItems.length = 0
  }

  for (const rawLine of lines) {
    const trimmedLine = rawLine.trim()
    if (!trimmedLine) {
      flushParagraph()
      flushList()
      continue
    }

    const headingMatch = trimmedLine.match(/^(#{1,3})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      flushList()
      const level = headingMatch[1].length
      blocks.push(`<h${level}>${formatMarkdownInline(headingMatch[2])}</h${level}>`)
      continue
    }

    const listMatch = trimmedLine.match(/^[-*•]\s+(.*)$/)
    if (listMatch) {
      flushParagraph()
      listItems.push(formatMarkdownInline(listMatch[1]))
      continue
    }

    flushList()
    paragraph.push(formatMarkdownInline(rawLine.trimEnd()))
  }

  flushParagraph()
  flushList()
  return blocks.join('')
}

function formatMarkdownInline(value = '') {
  return escapeHtml(value)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}
function escapeHtml(value = '') {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

async function copyMessage(message) {
  try {
    await navigator.clipboard?.writeText(message.content || '')
  } catch {
    // Clipboard access can be unavailable in some browser contexts.
  }
}

function regenerateFrom(index) {
  const previousUser = [...messages.value.slice(0, index)]
    .reverse()
    .find(message => message.role === 'user')
  if (!previousUser || streaming.value) return
  query.value = previousUser.content
  sendQuery()
}

const currentModeLabel = computed(() =>
  ragModes.find(mode => mode.value === props.chatSettings.mode)?.label || '无 RAG'
)

const currentPresetLabel = computed(() =>
  presetOptions.find(item => item.value === generationSettings.value.preset)?.label || '默认助手'
)

const currentModelLabel = computed(() =>
  modelOptions.value.find(item => item.value === generationSettings.value.model)?.label || 'Mock Chat'
)

const kbOptions = computed(() => [
  { value: '', label: '不使用知识库' },
  ...props.knowledgeBases.map(kb => ({ value: kb.id, label: kb.name })),
])

const currentKbLabel = computed(() =>
  kbOptions.value.find(item => item.value === props.chatSettings.kbId)?.label || '不使用知识库'
)

function toggleMenu(name) {
  openMenu.value = openMenu.value === name ? '' : name
}

async function choosePreset(value) {
  const previous = generationSettings.value.preset
  generationSettings.value.preset = value
  openMenu.value = ''
  try {
    const sessionId = await ensureSession()
    const updated = await updateChatSessionPreset(sessionId, value)
    generationSettings.value.preset = updated.preset_id || value
    props.chatSettings.presetId = generationSettings.value.preset
    emit('sessions-updated')
  } catch (err) {
    generationSettings.value.preset = previous
    props.chatSettings.presetId = previous
    errorText.value = err?.message || '预设切换失败'
  }
}

async function chooseModel(value) {
  const previous = generationSettings.value.model
  generationSettings.value.model = value
  openMenu.value = ''
  try {
    const sessionId = await ensureSession()
    const updated = await updateChatSessionModel(sessionId, value)
    generationSettings.value.model = updated.model_id || value
    props.chatSettings.modelId = generationSettings.value.model
    emit('sessions-updated')
  } catch (err) {
    generationSettings.value.model = previous
    props.chatSettings.modelId = previous
    errorText.value = err?.message || '模型切换失败'
  }
}

async function persistSessionConfig() {
  if (!props.currentSessionId) return
  try {
    await updateChatSession(props.currentSessionId, {
      rag_mode: props.chatSettings.mode,
      knowledge_base_id: props.chatSettings.kbId || null,
    })
    emit('sessions-updated')
  } catch (err) {
    errorText.value = err?.message || '会话配置保存失败'
  }
}

function chooseRagMode(value) {
  props.chatSettings.mode = value
  persistSessionConfig()
}

function chooseKnowledgeBase(value) {
  props.chatSettings.kbId = value
  openMenu.value = ''
  persistSessionConfig()
}

function formatTokenCount(value) {
  const count = Number(value || 0)
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}m`
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
  return String(count)
}

function messageTokenTotal(message) {
  return Number(message?.prompt_tokens || 0) + Number(message?.completion_tokens || 0)
}

function messageTokenLabel(message) {
  const prompt = Number(message?.prompt_tokens || 0)
  const completion = Number(message?.completion_tokens || 0)
  const total = prompt + completion
  if (!total) return ''
  if (prompt || completion) {
    return `Tokens: ${formatTokenCount(total)} (In: ${formatTokenCount(prompt)} / Out: ${formatTokenCount(completion)})`
  }
  return `${formatTokenCount(total)} Tokens`
}

async function scrollToBottom() {
  await nextTick()
  if (msgContainer.value) {
    msgContainer.value.scrollTop = msgContainer.value.scrollHeight
  }
}

async function loadModels() {
  try {
    const data = await listChatModels()
    applyModelOptions(data.models.map(model => ({
      value: model.id,
      label: model.display_name,
      provider: model.provider,
      model,
    })))
    defaultModelId.value = data.default_model_id || modelOptions.value[0]?.value || 'mock-chat'
    if (!modelOptions.value.some(item => item.value === generationSettings.value.model)) {
      generationSettings.value.model = defaultModelId.value
    }
  } catch (err) {
    errorText.value = err?.message || '模型列表加载失败'
    modelOptions.value = [{ value: 'mock-chat', label: 'Mock Chat' }]
    defaultModelId.value = 'mock-chat'
  }
}

function applyModelOptions(options) {
  const normalized = (options || [])
    .map(item => ({
      value: item.value || item.id,
      label: item.label || item.display_name || item.model_name || item.id,
      provider: item.provider,
      model: item.model || item,
    }))
    .filter(item => item.value)
  if (!normalized.length) return
  modelOptions.value = normalized
  if (!modelOptions.value.some(item => item.value === generationSettings.value.model)) {
    const preferred = props.chatSettings.modelId || defaultModelId.value || modelOptions.value[0]?.value
    generationSettings.value.model = modelOptions.value.some(item => item.value === preferred)
      ? preferred
      : modelOptions.value[0]?.value
  }
}

async function loadPresets() {
  try {
    const data = await listChatPresets()
    presetOptions.splice(
      0,
      presetOptions.length,
      ...data.presets.map(preset => ({
        value: preset.id,
        label: preset.name,
        description: preset.description,
        ownerType: preset.owner_type,
        preset,
      })),
    )
    defaultPresetId.value = data.default_preset_id || presetOptions[0]?.value || 'default-assistant'
    if (!presetOptions.some(item => item.value === generationSettings.value.preset)) {
      generationSettings.value.preset = defaultPresetId.value
    }
  } catch (err) {
    errorText.value = err?.message || '预设列表加载失败'
    presetOptions.splice(
      0,
      presetOptions.length,
      { value: 'default-assistant', label: '默认助手' },
    )
    defaultPresetId.value = 'default-assistant'
  }
}

onMounted(() => {
  loadModels()
  loadPresets()
})

watch(() => props.chatSettings.mode, () => {
  drawerOpen.value = false
})

watch(() => props.chatSettings.modelId, (modelId) => {
  if (modelId && modelId !== generationSettings.value.model) {
    generationSettings.value.model = modelId
  }
})

watch(() => props.modelOptions, (options) => {
  applyModelOptions(options)
}, { deep: true, immediate: true })

watch(() => props.chatSettings.presetId, (presetId) => {
  if (presetId && presetId !== generationSettings.value.preset) {
    generationSettings.value.preset = presetId
  }
})

watch(() => props.currentSessionId, (sessionId) => {
  if (streaming.value && sessionId === activeStreamSessionId.value) return
  loadSession(sessionId)
}, { immediate: true })

watch(() => props.newChatSignal, () => {
  abortCtrl?.abort()
  query.value = ''
  streaming.value = false
  activeStreamSessionId.value = ''
  drawerOpen.value = false
  activeMessage.value = null
  activeTab.value = 'citations'
  openMenu.value = ''
  errorText.value = ''
  if (!props.currentSessionId) {
    messages.value = []
  }
})
</script>

<style scoped>
.chat-view {
  position: relative;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-primary);
}

.chat-scroll {
  height: 100%;
  overflow-y: auto;
  padding: 56px 24px 128px;
}

.message-stack {
  width: min(800px, 100%);
  margin: 0 auto;
}

.welcome-panel {
  display: grid;
  min-height: 48vh;
  place-items: center;
  text-align: center;
  color: var(--text-muted);
}

.welcome-panel h1 {
  color: var(--text-primary);
  font-size: 28px;
  font-weight: 720;
}

.welcome-panel p {
  margin-top: 6px;
}

.chat-error {
  margin-bottom: 16px;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  background: #fff7ed;
  padding: 10px 12px;
  color: #c2410c;
  font-size: 13px;
}

.message {
  margin-bottom: 26px;
}

.message.user {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.message-label {
  margin-bottom: 7px;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 700;
}

.message-body {
  max-width: 100%;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-primary);
}

.message.user .message-body {
  max-width: min(620px, 90%);
  border-radius: 18px;
  background: var(--bg-secondary);
  padding: 10px 14px;
}

.message.assistant .message-body {
  font-size: 16px;
  line-height: 1.75;
}

.stopped-notice {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  gap: 6px;
  margin: 12px 0 8px;
  border: 1px solid rgba(245, 158, 11, 0.28);
  border-radius: 999px;
  background: #fffbeb;
  color: #92400e;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 750;
}

.stopped-notice::before {
  content: "■";
  color: #f59e0b;
  font-size: 8px;
}

.message-body.thinking {
  color: var(--text-muted);
}

.micro-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.micro-chips button {
  min-height: 28px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: var(--surface-strong);
  color: var(--text-secondary);
  padding: 0 10px;
  font-size: 12px;
}

.micro-chips button:hover {
  background: var(--accent-soft);
  color: var(--accent);
}

.composer {
  position: relative;
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  width: 100%;
  margin: 0;
  border: 1px solid var(--border-color);
  border-radius: 24px;
  background: var(--surface);
  padding: 10px 10px 10px 18px;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
}

.composer textarea {
  min-height: 34px;
  max-height: 150px;
  resize: none;
  border: 0;
  outline: 0;
  color: var(--text-primary);
  line-height: 34px;
}

.stop-button {
  min-width: 52px;
  border: 0;
  background: transparent;
  color: var(--text-muted);
}

.send-button {
  display: grid;
  width: 36px;
  height: 36px;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: #17212b;
  color: white;
  font-size: 20px;
  font-weight: 800;
}

.debug-drawer {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: min(380px, 100%);
  transform: translateX(100%);
  border-left: 1px solid var(--border-color);
  background: var(--surface);
  box-shadow: var(--shadow-md);
  transition: transform 0.22s ease;
  z-index: 5;
}

.debug-drawer.open {
  transform: translateX(0);
}

.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border-color);
}

.drawer-header div {
  display: flex;
  flex-direction: column;
}

.drawer-header span {
  color: var(--text-muted);
  font-size: 12px;
}

.drawer-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
  padding: 10px;
}

.drawer-tabs button {
  min-height: 34px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
}

.drawer-tabs button.active {
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-weight: 700;
}

.drawer-content {
  height: calc(100% - 108px);
  overflow: auto;
  padding: 10px 16px 20px;
}

.citation-list,
.trace-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.citation-list article {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--surface-strong);
  padding: 12px;
}

.chunk-meta,
.trace-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.chunk-meta span,
.citation-list small,
.trace-head span {
  color: var(--text-muted);
  font-size: 12px;
}

.citation-list p,
.trace-list p {
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 13px;
}

.trace-list article {
  display: grid;
  grid-template-columns: 12px 1fr;
  gap: 10px;
}

.trace-dot {
  width: 10px;
  height: 10px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--accent);
}

.drawer-content pre {
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-secondary);
  font-size: 12px;
}

.drawer-empty {
  color: var(--text-muted);
}

.composer-shell {
  display: grid;
  gap: 12px;
  z-index: 4;
}

.control-pill-bar {
  display: grid;
  grid-template-columns:
    minmax(170px, max-content)
    minmax(240px, 300px)
    minmax(300px, 360px)
    minmax(180px, 240px);
  align-items: center;
  gap: 14px;
  border: 1px solid var(--border-color);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.92);
  padding: 14px 16px;
  box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(14px);
}

.popover-control {
  position: relative;
  min-width: 0;
}

.control-slot {
  min-width: 0;
}

.control-pill {
  display: inline-flex;
  width: 100%;
  min-height: 48px;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: var(--surface-strong);
  padding: 0 14px;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1;
  min-width: 0;
}

.control-pill strong {
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 750;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.control-pill span {
  flex: 0 0 auto;
  color: var(--text-muted);
  font-size: 12px;
  white-space: nowrap;
}

.option-popover {
  position: absolute;
  left: 0;
  bottom: 40px;
  display: grid;
  min-width: 180px;
  gap: 2px;
  border: 1px solid rgba(226, 232, 240, 0.8);
  border-radius: 16px;
  background: white;
  padding: 8px;
  box-shadow: 0 20px 36px rgba(15, 23, 42, 0.16);
  z-index: 12;
}

.option-popover-right {
  right: 0;
  left: auto;
}

.option-item {
  display: flex;
  min-height: 34px;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  padding: 8px 12px;
  color: #334155;
  font-size: 12px;
  font-weight: 650;
  text-align: left;
  cursor: pointer;
}

.option-item:hover {
  background: rgba(241, 245, 249, 0.8);
}

.option-item.selected {
  color: #0f172a;
  font-weight: 800;
}

.option-item strong {
  color: #0f172a;
  font-size: 12px;
}

.control-pill:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.rag-popover {
  position: absolute;
  left: 0;
  bottom: 42px;
  display: grid;
  grid-template-columns: 190px minmax(260px, 320px);
  width: min(560px, calc(100vw - 36px));
  min-height: 318px;
  max-height: min(420px, calc(100vh - 190px));
  gap: 12px;
  border: 1px solid var(--border-color);
  border-radius: 14px;
  background: var(--surface);
  padding: 12px;
  box-shadow: var(--shadow-md);
}

.rag-mode-list {
  display: grid;
  align-content: start;
  gap: 4px;
  min-height: 0;
  border-right: 1px solid var(--border-color);
  padding-right: 10px;
}

.rag-params-panel {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
  padding-right: 2px;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.6) transparent;
}

.rag-params-panel::-webkit-scrollbar {
  width: 6px;
}

.rag-params-panel::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.55);
}

.rag-params-panel header {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--surface);
}

.rag-params-panel header strong {
  font-size: 13px;
}

.rag-params-panel header button {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: var(--bg-secondary);
  color: var(--text-secondary);
}

.direct-mode-note {
  display: grid;
  min-height: 214px;
  align-content: center;
  margin: 2px 0 0;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.6;
}

.slider-field,
.switch-field {
  display: grid;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
}

.segmented-field {
  display: grid;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.segmented-field > div {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 4px;
  border-radius: 8px;
  background: var(--bg-secondary);
  padding: 4px;
}

.segmented-field button {
  min-height: 30px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
}

.segmented-field button.active {
  background: var(--surface);
  color: var(--text-primary);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
}

.switch-field {
  grid-template-columns: 1fr auto;
  align-items: center;
}

.slider-field input[type="range"] {
  width: 100%;
  accent-color: #17212b;
}

.composer {
  position: relative;
  right: auto;
  bottom: auto;
  left: auto;
  width: 100%;
  margin: 0;
  border-radius: 22px;
}

@media (max-width: 1180px) {
  .control-pill-bar {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
}

@media (max-width: 700px) {
  .composer {
    width: 100%;
  }

  .control-pill-bar {
    align-items: stretch;
  }

  .control-pill-bar {
    grid-template-columns: 1fr;
    padding: 10px;
  }

  .control-pill,
  .kb-pill {
    width: 100%;
  }

  .rag-popover {
    left: 0;
    grid-template-columns: 1fr;
    width: 100%;
    min-height: 360px;
    max-height: min(520px, calc(100vh - 220px));
    overflow: hidden;
  }

  .rag-mode-list {
    max-height: 176px;
    overflow-y: auto;
    border-right: 0;
    border-bottom: 1px solid var(--border-color);
    padding-right: 0;
    padding-bottom: 10px;
  }
}

.chat-view {
  display: flex;
  width: 100%;
}

.chat-main-panel {
  --chat-canvas-width: min(1180px, calc(100% - 48px));

  position: relative;
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  transition: flex-basis 0.3s ease, width 0.3s ease;
}

.chat-scroll {
  flex: 1 1 auto;
  height: auto;
  min-height: 0;
  overflow-y: auto;
  padding: 56px 24px 36px;
}

.composer-shell {
  position: relative;
  right: auto;
  bottom: auto;
  left: auto;
  flex: 0 0 auto;
  width: var(--chat-canvas-width);
  margin: 0 auto;
  padding: 0 0 28px;
}

.debug-drawer {
  position: relative;
  top: auto;
  right: auto;
  bottom: auto;
  flex: 0 0 0;
  width: 0;
  height: 100%;
  overflow: hidden;
  border-left: 0;
  opacity: 0;
  transform: none;
  box-shadow: none;
  transition: flex-basis 0.3s ease, width 0.3s ease, opacity 0.2s ease, border-color 0.3s ease;
}

.debug-drawer.open {
  flex-basis: 360px;
  width: 360px;
  border-left: 1px solid var(--border-color);
  opacity: 1;
  transform: none;
}

.debug-drawer-inner {
  display: flex;
  width: 360px;
  height: 100%;
  flex-direction: column;
  background: var(--surface);
}

.drawer-content {
  flex: 1 1 auto;
  height: auto;
  min-height: 0;
}

@media (max-width: 900px) {
  .debug-drawer.open {
    flex-basis: 320px;
    width: 320px;
  }

  .debug-drawer-inner {
    width: 320px;
  }
}

.message-stack {
  display: flex;
  width: var(--chat-canvas-width);
  flex-direction: column;
  gap: 24px;
}

.control-pill-bar,
.composer {
  width: 100%;
}

.chat-view.drawer-open .chat-main-panel {
  --chat-canvas-width: min(1088px, calc(100% - 96px));
}

.chat-view.drawer-open .control-pill-bar {
  grid-template-columns:
    minmax(0, 0.9fr)
    minmax(0, 1.2fr)
    minmax(0, 1.35fr)
    minmax(0, 0.95fr);
}

@media (max-width: 1500px) {
  .chat-view.drawer-open .control-pill-bar {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
}

.message {
  margin-bottom: 0;
}

.ai-message-card {
  display: grid;
  gap: 12px;
}

.ai-message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.ai-avatar {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border-radius: 7px;
  background: #17212b;
  color: white;
  font-size: 10px;
  font-weight: 850;
}

.ai-message-header strong {
  color: var(--text-secondary);
  font-weight: 750;
}

.minimal-summary-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
  border: 0;
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.5);
  padding: 4px 8px;
  color: #94a3b8;
  font-size: 12px;
  text-align: left;
}

.minimal-summary-bar strong {
  color: #64748b;
  font-size: 11px;
  font-weight: 700;
}

.thinking-pulse {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: #94a3b8;
}

.thinking-pulse.active {
  animation: thinking-pulse 1.1s ease-in-out infinite;
  background: #6366f1;
}

@keyframes thinking-pulse {
  0%, 100% {
    opacity: 0.35;
    transform: scale(0.82);
  }

  50% {
    opacity: 1;
    transform: scale(1.12);
  }
}

.citation-preview-list {
  display: grid;
  gap: 8px;
  width: min(620px, 100%);
}

.citation-preview-list article {
  border: 1px solid #f1f5f9;
  border-radius: 10px;
  background: rgba(248, 250, 252, 0.56);
  padding: 9px 11px;
}

.citation-preview-list strong {
  color: #64748b;
  font-size: 12px;
}

.citation-preview-list p {
  margin-top: 4px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.55;
}

.markdown-body {
  line-height: 1.6;
}

.markdown-body :deep(p) {
  margin: 0 0 6px;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  margin: 8px 0 6px;
  color: var(--text-primary);
  line-height: 1.3;
}

.markdown-body :deep(code) {
  border-radius: 5px;
  background: #f1f5f9;
  padding: 2px 5px;
  color: #334155;
  font-size: 0.92em;
}

.markdown-body :deep(ul.md-list) {
  margin: 2px 0 6px;
  padding-left: 1.2em;
}

.markdown-body :deep(ul.md-list li) {
  margin: 2px 0;
}

.ai-action-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 2px;
  border-top: 1px solid #f1f5f9;
  padding-top: 8px;
  font-size: 12px;
}

.rag-actions,
.quick-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.citation-action,
.trace-action {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  gap: 5px;
  border: 0;
  border-radius: 8px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 750;
  transition: background 0.16s ease, color 0.16s ease;
}

.citation-action {
  background: #eef2ff;
  color: #4f46e5;
}

.citation-action:hover {
  background: #e0e7ff;
}

.citation-action strong {
  border-radius: 999px;
  background: rgba(199, 210, 254, 0.6);
  padding: 0 6px;
  color: #4338ca;
  font-size: 10px;
}

.trace-action {
  background: #fffbeb;
  color: #b45309;
}

.trace-action:hover {
  background: #fef3c7;
}

.trace-action small {
  color: #d97706;
  font-size: 10px;
}

.message-token-chip {
  display: inline-flex;
  min-height: 24px;
  align-items: center;
  gap: 4px;
  border: 1px solid rgba(251, 191, 36, 0.6);
  border-radius: 6px;
  background: #fffbeb;
  padding: 2px 8px;
  color: #b45309;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 11px;
  font-weight: 750;
  white-space: nowrap;
}

.quick-actions button {
  display: grid;
  width: 26px;
  height: 26px;
  place-items: center;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: #94a3b8;
}

.quick-actions button:hover {
  background: #f1f5f9;
  color: #475569;
}

@media (max-width: 1180px) {
  .chat-main-panel {
    --chat-canvas-width: min(920px, calc(100% - 40px));
  }
}

@media (max-width: 700px) {
  .chat-main-panel {
    --chat-canvas-width: calc(100% - 24px);
  }

  .chat-scroll {
    padding: 32px 12px 28px;
  }

  .composer-shell {
    padding: 12px 0 16px;
  }

  .chat-view.drawer-open .control-pill-bar {
    grid-template-columns: 1fr;
  }
}
</style>

