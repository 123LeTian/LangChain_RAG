<template>
  <div id="app-root" :class="{ collapsed: sidebarCollapsed }">
    <aside class="global-sidebar">
      <div class="sidebar-top">
        <div class="brand-mark" aria-hidden="true">R</div>
        <div class="brand-copy">
          <strong>LangChain RAG</strong>
          <span>实验工作台</span>
        </div>
        <button class="icon-button" type="button" title="折叠侧边栏" @click="sidebarCollapsed = !sidebarCollapsed">
          {{ sidebarCollapsed ? '>' : '<' }}
        </button>
      </div>

      <button class="new-chat" type="button" @click="startNewChat">
        <span class="new-chat-main">
          <span class="new-chat-icon">➕</span>
          <strong>新建对话</strong>
        </span>
      </button>

      <nav class="sidebar-nav" aria-label="主导航">
        <RouterLink v-for="item in navItems" :key="item.to" :to="item.to" active-class="active">
          <span class="nav-icon">{{ item.icon }}</span>
          <span class="nav-label">{{ item.label }}</span>
        </RouterLink>
      </nav>

      <section class="history-list" aria-label="历史对话">
        <div class="section-title">历史对话</div>
        <button class="sidebar-search-button" type="button" @click="openSearchModal">
          <div class="sidebar-search-copy">
            <span>🔍</span>
            <span>搜索历史对话...</span>
          </div>
        </button>
        <div v-if="sessionLoadError" class="history-empty">{{ sessionLoadError }}</div>
        <div v-else-if="!chatSessions.length" class="history-empty">暂无历史</div>
        <article
          v-for="session in chatSessions"
          :key="session.id"
          :class="['history-item', { active: session.id === currentSessionId }]"
        >
          <form
            v-if="editingSessionId === session.id"
            class="history-edit"
            @submit.prevent="saveSessionTitle(session.id)"
          >
            <input
              v-model="editingTitle"
              type="text"
              aria-label="会话标题"
              @keydown.esc.prevent="cancelRename"
            />
            <button type="submit" title="保存">✓</button>
            <button type="button" title="取消" @click="cancelRename">×</button>
          </form>
          <template v-else>
            <button class="history-select" type="button" @click="selectSession(session.id)">
              <span>{{ session.title }}</span>
              <small>{{ formatSessionTime(session.updated_at) }}</small>
            </button>
            <div class="history-actions">
              <button type="button" title="重命名" @click.stop="startRename(session)">✎</button>
              <button type="button" title="删除" @click.stop="removeSession(session.id)">×</button>
            </div>
          </template>
        </article>
      </section>

      <section class="sidebar-config-dock" aria-label="会话快捷配置">
        <div class="sidebar-config-control">
          <button
            :class="['sidebar-config-button', { active: activeSidebarConfig === 'model' }]"
            type="button"
            @click="toggleSidebarConfig('model')"
          >
            <span class="config-icon">◎</span>
            <span>
              <strong>模型配置</strong>
              <small>{{ currentModelLabel }}</small>
            </span>
          </button>
          <div v-if="activeSidebarConfig === 'model'" class="sidebar-config-panel">
            <button
              v-for="model in sidebarModelOptions"
              :key="model.value"
              :class="['sidebar-option', { selected: chatSettings.modelId === model.value }]"
              type="button"
              @click="chooseSidebarModel(model.value)"
            >
              <span>
                <strong>{{ model.label }}</strong>
                <small>{{ model.provider }}</small>
              </span>
              <b v-if="chatSettings.modelId === model.value">✓</b>
            </button>
          </div>
        </div>

        <div class="sidebar-config-control">
          <button
            :class="['sidebar-config-button', { active: activeSidebarConfig === 'session' }]"
            type="button"
            @click="toggleSidebarConfig('session')"
          >
            <span class="config-icon">⚙</span>
            <span>
              <strong>会话配置</strong>
              <small>{{ currentSessionConfigLabel }}</small>
            </span>
          </button>
          <div v-if="activeSidebarConfig === 'session'" class="sidebar-config-panel">
            <div class="sidebar-option-group">
              <span>RAG 模式</span>
              <button
                v-for="mode in ragModeOptions"
                :key="mode.value"
                :class="['sidebar-option', { selected: chatSettings.mode === mode.value }]"
                type="button"
                @click="chooseSidebarRagMode(mode.value)"
              >
                <span>{{ mode.label }}</span>
                <b v-if="chatSettings.mode === mode.value">✓</b>
              </button>
            </div>
            <div class="sidebar-option-group">
              <span>知识库</span>
              <button
                v-for="kb in sidebarKnowledgeBaseOptions"
                :key="kb.value"
                :class="['sidebar-option', { selected: chatSettings.kbId === kb.value }]"
                type="button"
                @click="chooseSidebarKnowledgeBase(kb.value)"
              >
                <span>{{ kb.label }}</span>
                <b v-if="chatSettings.kbId === kb.value">✓</b>
              </button>
            </div>
          </div>
        </div>
      </section>

      <div class="sidebar-status">
        <span class="status-dot" :class="healthStatus" />
        <span>{{ healthText }}</span>
      </div>
    </aside>

    <main class="app-canvas">
      <RouterView v-slot="{ Component }">
        <component
          :is="Component"
          :chat-settings="chatSettings"
          :knowledge-bases="knowledgeBases"
          :current-session-id="currentSessionId"
          :new-chat-signal="newChatSignal"
          @session-created="handleSessionCreated"
          @sessions-updated="loadChatSessions"
          @knowledge-bases-updated="loadKnowledgeBases"
        />
      </RouterView>
    </main>

    <teleport to="body">
      <div
        v-if="searchModalOpen"
        class="search-modal-backdrop"
        role="presentation"
        @mousedown.self="closeSearchModal"
      >
        <section class="search-modal" role="dialog" aria-modal="true" aria-label="搜索历史对话">
          <header class="search-modal-input-row">
            <span aria-hidden="true">🔍</span>
            <input
              ref="searchInputRef"
              v-model="historySearch"
              type="text"
              placeholder="搜索历史对话..."
              autocomplete="off"
              @input="scheduleHistorySearch"
              @keydown.down.prevent="moveSearchSelection(1)"
              @keydown.up.prevent="moveSearchSelection(-1)"
              @keydown.enter.prevent="confirmSearchSelection"
              @keydown.esc.prevent="closeSearchModal"
            />
            <button class="search-modal-close" type="button" aria-label="关闭搜索" @click="closeSearchModal">❌</button>
          </header>

          <div class="search-modal-body">
            <div v-if="historySearchLoading" class="search-modal-state">搜索中...</div>
            <div v-else-if="historySearch.trim() && !historySearchResults.length" class="search-modal-state">
              没有找到匹配消息
            </div>
            <div v-else-if="!historySearch.trim()" class="search-modal-state">
              输入关键词搜索所有历史消息
            </div>

            <button
              v-for="(item, index) in historySearchResults"
              :key="item.message_id"
              :class="['search-result-item', { active: index === selectedSearchIndex }]"
              type="button"
              @mouseenter="selectedSearchIndex = index"
              @click="openSearchResult(item.session_id)"
            >
              <div>
                <strong>{{ item.session_title }}</strong>
                <span>{{ item.role === 'user' ? '你' : item.role === 'assistant' ? 'AI' : 'System' }}</span>
              </div>
              <p>{{ item.snippet }}</p>
            </button>
          </div>
        </section>
      </div>
    </teleport>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import {
  createChatSession,
  deleteChatSession,
  healthCheck,
  listChatModels,
  listChatSessions,
  listKnowledgeBases,
  searchChatMessages,
  updateChatSession,
  updateChatSessionModel,
} from './api/client'

const router = useRouter()
const sidebarCollapsed = ref(false)
const healthStatus = ref('checking')
const healthText = ref('连接中...')
const knowledgeBases = ref([])
const newChatSignal = ref(0)
const chatSessions = ref([])
const currentSessionId = ref('')
const sessionLoadError = ref('')
const editingSessionId = ref('')
const editingTitle = ref('')
const historySearch = ref('')
const historySearchResults = ref([])
const historySearchLoading = ref(false)
const searchModalOpen = ref(false)
const searchInputRef = ref(null)
const selectedSearchIndex = ref(0)
const activeSidebarConfig = ref('')
const sidebarModelOptions = ref([])
const defaultModelId = ref('deepseek-chat')
let historySearchTimer = null

const chatSettings = reactive({
  modelId: 'deepseek-chat',
  presetId: '',
  mode: 'naive',
  kbId: '',
  topK: 5,
  rewriteEnabled: false,
  rerankEnabled: false,
  scoreThreshold: 0.2,
  maxSteps: 5,
})

const navItems = [
  { to: '/chat', icon: '💬', label: '问答实验台' },
  { to: '/knowledge', icon: '📚', label: '知识库管理' },
  { to: '/graph', icon: '🕸️', label: '知识图谱' },
  { to: '/evaluation', icon: '📊', label: '模式评测' },
]

const ragModeOptions = [
  { value: 'naive', label: 'Naive RAG' },
  { value: 'advanced', label: 'Advanced RAG' },
  { value: 'modular', label: 'Modular RAG' },
  { value: 'graph', label: 'GraphRAG' },
  { value: 'agentic', label: 'Agentic RAG' },
]

const currentSession = computed(() =>
  chatSessions.value.find(session => session.id === currentSessionId.value) || null
)

const sidebarKnowledgeBaseOptions = computed(() => [
  { value: '', label: '不使用知识库' },
  ...knowledgeBases.value.map(kb => ({ value: kb.id, label: kb.name })),
])

const currentModelLabel = computed(() =>
  sidebarModelOptions.value.find(model => model.value === chatSettings.modelId)?.label || 'DeepSeek Chat'
)

const currentRagModeLabel = computed(() =>
  ragModeOptions.find(mode => mode.value === chatSettings.mode)?.label || 'Naive RAG'
)

const currentKnowledgeBaseLabel = computed(() =>
  sidebarKnowledgeBaseOptions.value.find(kb => kb.value === chatSettings.kbId)?.label || '不使用知识库'
)

const currentSessionConfigLabel = computed(() =>
  `${currentRagModeLabel.value} · ${currentKnowledgeBaseLabel.value}`
)

function applySessionSettings(session) {
  if (!session) return
  chatSettings.modelId = session.model_id || defaultModelId.value
  chatSettings.presetId = session.preset_id || ''
  if (session.rag_mode) chatSettings.mode = session.rag_mode
  if (session.knowledge_base_id !== null && session.knowledge_base_id !== undefined) {
    chatSettings.kbId = session.knowledge_base_id || ''
  }
}

async function loadSidebarModels() {
  try {
    const data = await listChatModels()
    sidebarModelOptions.value = data.models.map(model => ({
      value: model.id,
      label: model.display_name,
      provider: model.provider,
    }))
    defaultModelId.value = data.default_model_id || sidebarModelOptions.value[0]?.value || 'deepseek-chat'
    if (!sidebarModelOptions.value.some(model => model.value === chatSettings.modelId)) {
      chatSettings.modelId = defaultModelId.value
    }
  } catch {
    sidebarModelOptions.value = [{ value: 'deepseek-chat', label: 'DeepSeek Chat', provider: 'deepseek' }]
    defaultModelId.value = 'deepseek-chat'
  }
}

async function loadKnowledgeBases() {
  try {
    knowledgeBases.value = await listKnowledgeBases()
    if (!chatSettings.kbId && knowledgeBases.value.length > 0) {
      chatSettings.kbId = knowledgeBases.value[0].id
    }
  } catch {
    knowledgeBases.value = []
  }
}

async function loadChatSessions() {
  try {
    sessionLoadError.value = ''
    chatSessions.value = await listChatSessions()
    if (!currentSessionId.value && chatSessions.value.length > 0) {
      currentSessionId.value = chatSessions.value[0].id
    }
    if (
      currentSessionId.value
      && !chatSessions.value.some(session => session.id === currentSessionId.value)
    ) {
      currentSessionId.value = chatSessions.value[0]?.id || ''
    }
    applySessionSettings(currentSession.value)
  } catch {
    sessionLoadError.value = '历史加载失败'
  }
}

async function startNewChat() {
  try {
    const session = await createChatSession({
      title: '新对话',
      model_id: chatSettings.modelId || null,
      preset_id: chatSettings.presetId || null,
      rag_mode: chatSettings.mode,
      knowledge_base_id: chatSettings.kbId || null,
    })
    currentSessionId.value = session.id
    newChatSignal.value += 1
    await loadChatSessions()
  } catch {
    sessionLoadError.value = '新建失败'
  }
  router.push('/chat')
}

function selectSession(sessionId) {
  currentSessionId.value = sessionId
  applySessionSettings(chatSessions.value.find(session => session.id === sessionId))
  router.push('/chat')
}

function toggleSidebarConfig(name) {
  activeSidebarConfig.value = activeSidebarConfig.value === name ? '' : name
}

async function ensureSidebarSession() {
  if (currentSessionId.value) return currentSessionId.value
  await startNewChat()
  return currentSessionId.value
}

async function chooseSidebarModel(modelId) {
  const previous = chatSettings.modelId
  chatSettings.modelId = modelId
  try {
    const sessionId = await ensureSidebarSession()
    const updated = await updateChatSessionModel(sessionId, modelId)
    chatSessions.value = chatSessions.value.map(session => session.id === updated.id ? updated : session)
    applySessionSettings(updated)
  } catch (err) {
    chatSettings.modelId = previous
    sessionLoadError.value = err?.message || '模型切换失败'
  }
}

async function persistSidebarSessionConfig() {
  if (!currentSessionId.value) return
  try {
    const updated = await updateChatSession(currentSessionId.value, {
      rag_mode: chatSettings.mode,
      knowledge_base_id: chatSettings.kbId || null,
    })
    chatSessions.value = chatSessions.value.map(session => session.id === updated.id ? updated : session)
  } catch (err) {
    sessionLoadError.value = err?.message || '会话配置保存失败'
  }
}

function chooseSidebarRagMode(value) {
  chatSettings.mode = value
  persistSidebarSessionConfig()
}

function chooseSidebarKnowledgeBase(value) {
  chatSettings.kbId = value
  persistSidebarSessionConfig()
}

async function runHistorySearch() {
  const q = historySearch.value.trim()
  if (!q) {
    historySearchResults.value = []
    historySearchLoading.value = false
    return
  }
  try {
    historySearchLoading.value = true
    const result = await searchChatMessages({ q, limit: 8 })
    if (historySearch.value.trim() === q) {
      historySearchResults.value = result.items
    }
  } catch {
    historySearchResults.value = []
  } finally {
    historySearchLoading.value = false
  }
}

function scheduleHistorySearch() {
  window.clearTimeout(historySearchTimer)
  historySearchTimer = window.setTimeout(runHistorySearch, 220)
}

async function openSearchModal() {
  searchModalOpen.value = true
  selectedSearchIndex.value = 0
  await nextTick()
  searchInputRef.value?.focus()
}

function closeSearchModal() {
  searchModalOpen.value = false
  historySearch.value = ''
  historySearchResults.value = []
  historySearchLoading.value = false
  selectedSearchIndex.value = 0
  window.clearTimeout(historySearchTimer)
}

function moveSearchSelection(delta) {
  if (!historySearchResults.value.length) return
  const total = historySearchResults.value.length
  selectedSearchIndex.value = (selectedSearchIndex.value + delta + total) % total
}

function confirmSearchSelection() {
  const item = historySearchResults.value[selectedSearchIndex.value]
  if (item) openSearchResult(item.session_id)
}

function openSearchResult(sessionId) {
  closeSearchModal()
  selectSession(sessionId)
}

function handleGlobalKeydown(event) {
  const isSearchShortcut = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k'
  if (!isSearchShortcut) return
  event.preventDefault()
  if (searchModalOpen.value) {
    closeSearchModal()
  } else {
    openSearchModal()
  }
}

function startRename(session) {
  editingSessionId.value = session.id
  editingTitle.value = session.title
}

function cancelRename() {
  editingSessionId.value = ''
  editingTitle.value = ''
}

async function saveSessionTitle(sessionId) {
  const title = editingTitle.value.trim()
  const current = chatSessions.value.find(session => session.id === sessionId)
  if (!title || title === current?.title) {
    cancelRename()
    return
  }
  try {
    const updated = await updateChatSession(sessionId, { title })
    chatSessions.value = chatSessions.value.map(item => item.id === updated.id ? updated : item)
    cancelRename()
  } catch {
    sessionLoadError.value = '重命名失败'
  }
}

async function removeSession(sessionId) {
  try {
    await deleteChatSession(sessionId)
    if (editingSessionId.value === sessionId) cancelRename()
    await loadChatSessions()
    if (currentSessionId.value === sessionId) {
      currentSessionId.value = chatSessions.value[0]?.id || ''
      newChatSignal.value += 1
    }
  } catch {
    sessionLoadError.value = '删除失败'
  }
}

async function handleSessionCreated(session) {
  currentSessionId.value = session.id
  applySessionSettings(session)
  await loadChatSessions()
}

function formatSessionTime(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const now = new Date()
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

watch(historySearchResults, (items) => {
  if (!items.length) {
    selectedSearchIndex.value = 0
    return
  }
  selectedSearchIndex.value = Math.min(selectedSearchIndex.value, items.length - 1)
})

onMounted(async () => {
  window.addEventListener('keydown', handleGlobalKeydown)
  await loadSidebarModels()
  await loadKnowledgeBases()
  await loadChatSessions()
  try {
    const res = await healthCheck()
    healthStatus.value = 'ok'
    healthText.value = `API v${res.version}`
  } catch {
    healthStatus.value = 'error'
    healthText.value = 'API 离线'
  }
})

watch(currentSession, (session) => {
  applySessionSettings(session)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeydown)
  window.clearTimeout(historySearchTimer)
})
</script>

<style scoped>
.sidebar-search-button {
  display: flex;
  width: 100%;
  min-height: 36px;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 2px 0 4px;
  border: 1px solid transparent;
  border-radius: 12px;
  background: rgba(241, 245, 249, 0.72);
  padding: 8px 12px;
  color: #94a3b8;
  font-size: 12px;
  transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease;
}

.sidebar-search-button:hover {
  border-color: #e2e8f0;
  background: rgba(226, 232, 240, 0.72);
  color: #475569;
}

.sidebar-search-copy {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.sidebar-search-copy span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.search-modal-backdrop {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  background: rgba(15, 23, 42, 0.24);
  padding: min(14vh, 112px) 18px 24px;
  backdrop-filter: blur(8px);
  z-index: 80;
}

.search-modal {
  width: min(640px, 100%);
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 28px 80px rgba(15, 23, 42, 0.26);
}

.search-modal-input-row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 12px;
  align-items: center;
  min-height: 58px;
  border-bottom: 1px solid #eef2f7;
  padding: 0 16px;
}

.search-modal-input-row input {
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: #0f172a;
  font-size: 15px;
}

.search-modal-close {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #94a3b8;
  font-size: 13px;
  line-height: 1;
}

.search-modal-close:hover {
  background: #f1f5f9;
  color: #475569;
}

.search-modal-body {
  display: grid;
  max-height: min(58vh, 520px);
  gap: 4px;
  overflow: auto;
  padding: 8px;
}

.search-modal-state {
  padding: 24px 16px;
  color: #94a3b8;
  font-size: 13px;
  text-align: center;
}

.search-result-item {
  display: grid;
  gap: 6px;
  border: 1px solid transparent;
  border-radius: 12px;
  background: transparent;
  padding: 11px 12px;
  color: #475569;
  text-align: left;
}

.search-result-item:hover,
.search-result-item.active {
  border-color: #e2e8f0;
  background: #f8fafc;
}

.search-result-item div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.search-result-item strong {
  overflow: hidden;
  color: #0f172a;
  font-size: 13px;
  font-weight: 750;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.search-result-item span {
  flex: 0 0 auto;
  color: #94a3b8;
  font-size: 11px;
}

.search-result-item p {
  display: -webkit-box;
  overflow: hidden;
  color: #64748b;
  font-size: 12px;
  line-height: 1.5;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.sidebar-config-dock {
  display: grid;
  gap: 8px;
  padding-top: 2px;
}

.sidebar-config-control {
  position: relative;
  display: grid;
  gap: 6px;
}

.sidebar-config-button {
  display: grid;
  grid-template-columns: 28px 1fr;
  width: 100%;
  min-height: 48px;
  align-items: center;
  gap: 8px;
  border: 1px solid rgba(226, 232, 240, 0.84);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.9);
  color: #475569;
  padding: 8px 10px;
  text-align: left;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease;
}

.sidebar-config-button:hover,
.sidebar-config-button.active {
  border-color: #cbd5e1;
  background: #f8fafc;
  color: #0f172a;
}

.config-icon {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 9px;
  background: #f1f5f9;
  color: #64748b;
  font-size: 13px;
}

.sidebar-config-button > span:last-child {
  display: grid;
  min-width: 0;
  gap: 1px;
}

.sidebar-config-button strong,
.sidebar-config-button small,
.sidebar-option strong,
.sidebar-option small,
.sidebar-option > span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sidebar-config-button strong {
  font-size: 12px;
  font-weight: 750;
}

.sidebar-config-button small {
  color: #94a3b8;
  font-size: 11px;
}

.sidebar-config-panel {
  display: grid;
  max-height: min(42vh, 340px);
  gap: 5px;
  overflow: auto;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.98);
  padding: 6px;
  box-shadow: 0 16px 42px rgba(15, 23, 42, 0.14);
}

.sidebar-option-group {
  display: grid;
  gap: 5px;
}

.sidebar-option-group + .sidebar-option-group {
  margin-top: 6px;
  border-top: 1px solid #eef2f7;
  padding-top: 8px;
}

.sidebar-option-group > span {
  padding: 0 4px;
  color: #94a3b8;
  font-size: 11px;
  font-weight: 750;
}

.sidebar-option {
  display: flex;
  width: 100%;
  min-height: 34px;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: transparent;
  color: #475569;
  padding: 7px 8px;
  text-align: left;
}

.sidebar-option:hover,
.sidebar-option.selected {
  border-color: #e2e8f0;
  background: #f8fafc;
  color: #0f172a;
}

.sidebar-option span {
  display: grid;
  min-width: 0;
  gap: 1px;
}

.sidebar-option strong {
  font-size: 12px;
  font-weight: 700;
}

.sidebar-option small {
  color: #94a3b8;
  font-size: 10px;
  text-transform: uppercase;
}

.sidebar-option b {
  flex: 0 0 auto;
  color: #16a34a;
  font-size: 12px;
}

#app-root.collapsed .sidebar-config-dock {
  display: none;
}
</style>
