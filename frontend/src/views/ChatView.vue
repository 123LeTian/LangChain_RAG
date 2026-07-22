<template>
  <section class="chat-view">
    <main class="chat-main-panel">
      <div ref="msgContainer" class="chat-scroll">
        <div class="message-stack">
          <div v-if="messages.length === 0" class="welcome-panel">
            <h1>问答实验台</h1>
            <p>在底部选择模型、RAG 模式和知识库后开始提问。</p>
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
          <div class="toolbar-left">
            <div class="popover-control">
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

            <div class="popover-control">
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

            <span class="control-divider" aria-hidden="true" />

            <div class="rag-cluster">
              <div class="popover-control">
                <button class="control-pill rag-mode-pill" type="button" @click="toggleMenu('rag')">
                  <span>⚡ RAG 模式</span>
                  <strong>{{ currentModeLabel }}</strong>
                </button>
                <div v-if="openMenu === 'rag'" class="option-popover">
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
              </div>

              <div class="advanced-wrap">
                <button
                  class="advanced-pill"
                  type="button"
                  :class="{ active: advancedOpen }"
                  @click="toggleAdvanced"
                >
                  ⚙️ 高级参数
                </button>

                <div v-if="advancedOpen" class="advanced-popover">
                  <header>
                    <strong>{{ currentModeLabel }}</strong>
                    <button type="button" title="关闭" @click="advancedOpen = false">×</button>
                  </header>

                  <label class="slider-field">
                    <span>Top-K：{{ props.chatSettings.topK }}</span>
                    <input v-model.number="props.chatSettings.topK" type="range" min="1" max="20" />
                  </label>

                  <label v-if="props.chatSettings.mode === 'naive'" class="slider-field">
                    <span>Score Threshold：{{ props.chatSettings.scoreThreshold.toFixed(2) }}</span>
                    <input v-model.number="props.chatSettings.scoreThreshold" type="range" min="0" max="1" step="0.05" />
                  </label>

                  <template v-if="advancedModes.includes(props.chatSettings.mode)">
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
                      <span>Rerank Top-K：{{ props.chatSettings.rerankTopK }}</span>
                      <input v-model.number="props.chatSettings.rerankTopK" type="range" min="1" max="20" />
                    </label>
                  </template>

                  <template v-if="props.chatSettings.mode === 'agentic'">
                    <label class="slider-field">
                      <span>Max Steps：{{ props.chatSettings.maxSteps }}</span>
                      <input v-model.number="props.chatSettings.maxSteps" type="range" min="1" max="12" />
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
                </div>
              </div>
            </div>
          </div>

          <div class="toolbar-right">
            <div class="popover-control kb-control">
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
  { value: 'naive', label: 'Naive RAG' },
  { value: 'advanced', label: 'Advanced RAG' },
  { value: 'modular', label: 'Modular RAG' },
  { value: 'graph', label: 'GraphRAG' },
  { value: 'agentic', label: 'Agentic RAG' },
]

const presetOptions = reactive([
  { value: 'research', label: '严谨研报' },
  { value: 'summary', label: '极简摘要' },
  { value: 'structured', label: '代码/结构化' },
])

const advancedModes = ['advanced', 'modular']
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
const advancedOpen = ref(false)
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
    summaryOpen: false,
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
    summaryOpen: false,
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
      rerank_top_k: props.chatSettings.rerankTopK,
      score_threshold: props.chatSettings.scoreThreshold,
      temperature: null,
      rewrite_enabled: props.chatSettings.rewriteEnabled,
      retrieve_enabled: props.chatSettings.retrieveEnabled,
      rerank_enabled: props.chatSettings.rerankEnabled,
      compress_enabled: props.chatSettings.compressEnabled,
      verify_enabled: props.chatSettings.verifyEnabled,
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
  if (message.thinking) return '🧠 正在思考与检索知识库...'
  const chunksCount = message.citations?.length || retrievedChunkCount(message.trace)
  return `🧠 已检索 ${chunksCount} 条文档 · 思考完成 (${totalDuration(message.trace)}ms)`
}

function retrievedChunkCount(trace = []) {
  const retrieveEvent = trace.find(item => String(item.stage || '').toLowerCase().includes('retrieve'))
  const text = `${retrieveEvent?.output_summary || ''} ${retrieveEvent?.input_summary || ''}`
  const match = text.match(/(\d+)/)
  return match ? Number(match[1]) : 0
}

function renderMarkdown(value = '') {
  const escaped = escapeHtml(value)
  return escaped
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^\s*[-*]\s+(.*)$/gm, '<p class="md-list">• $1</p>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^(.+)$/s, '<p>$1</p>')
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
  ragModes.find(mode => mode.value === props.chatSettings.mode)?.label || 'RAG 参数'
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
  advancedOpen.value = false
  openMenu.value = openMenu.value === name ? '' : name
}

function toggleAdvanced() {
  openMenu.value = ''
  advancedOpen.value = !advancedOpen.value
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
  openMenu.value = ''
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
    modelOptions.value = data.models.map(model => ({
      value: model.id,
      label: model.display_name,
      provider: model.provider,
      model,
    }))
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
  advancedOpen.value = false
  openMenu.value = ''
})

watch(() => props.chatSettings.modelId, (modelId) => {
  if (modelId && modelId !== generationSettings.value.model) {
    generationSettings.value.model = modelId
  }
})

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
  advancedOpen.value = false
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
  position: absolute;
  right: 24px;
  bottom: 24px;
  left: 24px;
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  width: min(800px, calc(100% - 48px));
  margin: 0 auto;
  border: 1px solid var(--border-color);
  border-radius: 24px;
  background: var(--surface);
  padding: 10px 10px 10px 18px;
  box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
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

.chat-scroll {
  padding-bottom: 228px;
}

.composer-shell {
  position: absolute;
  right: 24px;
  bottom: 24px;
  left: 24px;
  display: grid;
  gap: 8px;
  width: min(800px, calc(100% - 48px));
  margin: 0 auto;
  z-index: 4;
}

.control-pill-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid var(--border-color);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  padding: 6px 8px;
  box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(14px);
}

.toolbar-left,
.toolbar-right {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.toolbar-left {
  min-width: 0;
  flex: 1;
}

.toolbar-right {
  flex: 0 0 auto;
  justify-content: flex-end;
}

.popover-control {
  position: relative;
}

.rag-cluster {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: rgba(248, 250, 252, 0.72);
  padding: 2px;
  overflow: visible;
}

.control-pill {
  display: inline-flex;
  min-height: 32px;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: var(--surface-strong);
  padding: 0 10px;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1;
}

.control-pill strong {
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 750;
  white-space: nowrap;
}

.rag-cluster .control-pill {
  border: 0;
  background: transparent;
}

.control-pill span {
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

.advanced-wrap {
  position: relative;
}

.advanced-pill {
  min-height: 32px;
  border: 0;
  border-left: 1px solid var(--border-color);
  border-radius: 0 999px 999px 0;
  background: transparent;
  padding: 0 11px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
}

.advanced-pill:hover,
.advanced-pill.active,
.control-pill:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.advanced-popover {
  position: absolute;
  right: 0;
  bottom: 42px;
  display: grid;
  width: 276px;
  gap: 12px;
  border: 1px solid var(--border-color);
  border-radius: 14px;
  background: var(--surface);
  padding: 12px;
  box-shadow: var(--shadow-md);
}

.advanced-popover header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.advanced-popover header strong {
  font-size: 13px;
}

.advanced-popover header button {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: var(--bg-secondary);
  color: var(--text-secondary);
}

.slider-field,
.switch-field {
  display: grid;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
}

.switch-field {
  grid-template-columns: 1fr auto;
  align-items: center;
}

.slider-field input[type="range"] {
  width: 100%;
  accent-color: #17212b;
}

.control-divider {
  width: 1px;
  height: 14px;
  margin: 0 4px;
  background: var(--border-color);
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

@media (max-width: 700px) {
  .chat-scroll {
    padding: 32px 16px 250px;
  }

  .composer-shell {
    right: 12px;
    bottom: 14px;
    left: 12px;
    width: calc(100% - 24px);
  }

  .composer {
    width: 100%;
  }

  .control-pill-bar,
  .toolbar-left,
  .toolbar-right {
    align-items: stretch;
  }

  .control-pill-bar {
    flex-direction: column;
  }

  .toolbar-left,
  .toolbar-right {
    width: 100%;
  }

  .control-pill,
  .rag-cluster,
  .kb-control,
  .kb-pill {
    width: 100%;
  }

  .rag-cluster {
    flex-wrap: wrap;
    border-radius: 16px;
  }

  .advanced-pill {
    border-left: 0;
    border-top: 1px solid var(--border-color);
    border-radius: 0 0 16px 16px;
    width: 100%;
  }
}

.chat-view {
  display: flex;
  width: 100%;
}

.chat-main-panel {
  position: relative;
  display: flex;
  min-width: 0;
  flex: 1 1 auto;
  flex-direction: column;
  height: 100%;
  transition: flex-basis 0.3s ease, width 0.3s ease;
}

.chat-scroll {
  flex: 1 1 auto;
  height: auto;
  min-height: 0;
  padding: 56px 24px 24px;
}

.composer-shell {
  position: relative;
  right: auto;
  bottom: auto;
  left: auto;
  flex: 0 0 auto;
  width: min(800px, calc(100% - 48px));
  margin: 0 auto;
  padding: 0 0 24px;
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
  width: min(768px, 100%);
  flex-direction: column;
  gap: 24px;
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
  line-height: 1.75;
}

.markdown-body :deep(p) {
  margin: 0 0 10px;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  margin: 10px 0 8px;
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

.markdown-body :deep(.md-list) {
  margin: 4px 0;
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

.composer-shell {
  width: min(768px, calc(100% - 48px));
}
</style>
