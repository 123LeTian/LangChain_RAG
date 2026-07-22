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
          <kbd>Ctrl+K</kbd>
        </button>

        <div class="history-scroll">
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
                <button type="button" title="导出为 Markdown" @click.stop="downloadSessionMarkdown(session)">📥</button>
                <button type="button" title="重命名" @click.stop="startRename(session)">✎</button>
                <button type="button" title="删除" @click.stop="removeSession(session.id)">×</button>
              </div>
            </template>
          </article>
        </div>
      </section>

      <section class="sidebar-config-dock" aria-label="会话快捷配置">
        <div class="sidebar-config-bar">
          <button
            class="sidebar-config-pill"
            type="button"
            title="点击配置模型"
            @click="openConfigModal('model')"
          >
            <span>🤖</span>
            <strong>模型配置</strong>
          </button>

          <button
            class="sidebar-config-pill"
            type="button"
            title="点击配置预设"
            @click="openConfigModal('preset')"
          >
            <span>🎭</span>
            <strong>提示词预设</strong>
          </button>
        </div>
      </section>
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

    <teleport to="body">
      <div
        v-if="configModalOpen"
        class="config-page-backdrop"
        role="presentation"
        @mousedown.self="closeConfigModal"
      >
        <section class="config-page" role="dialog" aria-modal="true">
          <header class="config-page-header">
            <button class="config-back-button" type="button" @click="closeConfigModal">← 返回对话</button>
            <div class="config-page-title">
              <strong>{{ configModalOpen === 'model' ? '模型配置' : '提示词预设' }}</strong>
              <span>
                {{ configModalOpen === 'model'
                  ? '模型列表、自定义模型、默认模型、连接测试和密钥状态'
                  : '内置预设、个人预设 CRUD 和当前会话预设选择' }}
              </span>
            </div>
            <button class="config-primary-button" type="button" @click="openItemDialog(configModalOpen)">
              {{ configModalOpen === 'model' ? '+ 添加模型' : '+ 新建预设' }}
            </button>
          </header>

          <div v-if="configMessage" class="config-message">{{ configMessage }}</div>

          <div v-if="configModalOpen === 'model'" class="config-card-grid">
            <article
              v-for="model in managedModels"
              :key="model.id"
              class="config-card"
            >
              <header>
                <div>
                  <strong>{{ model.display_name }}</strong>
                  <small>{{ model.provider }} · {{ model.model_name }}</small>
                </div>
                <span>{{ model.is_default ? '默认' : model.is_builtin ? '内置·只读' : '自定义' }}</span>
              </header>
              <p>{{ model.description || model.base_url || 'OpenAI-compatible chat completion model.' }}</p>
              <div class="config-card-meta">
                <span>{{ model.enabled ? '已启用' : '已停用' }}</span>
                <span>{{ model.key_required ? (model.key_configured ? '密钥已配置' : '缺少密钥') : '无需密钥' }}</span>
                <span>{{ model.supports_stream ? '流式' : '非流式' }}</span>
              </div>
              <footer>
                <div>
                  <button type="button" @click="testManagedModel(model.id)">测试连接</button>
                  <button type="button" @click="chooseManagedModel(model.id)">用于当前</button>
                  <button type="button" @click="setDefaultModel(model.id)">设默认</button>
                  <button v-if="!model.is_builtin" type="button" @click="editModelDraft(model)">编辑</button>
                </div>
                <button
                  v-if="!model.is_builtin"
                  class="text-danger"
                  type="button"
                  @click="removeManagedModel(model.id)"
                >
                  删除
                </button>
              </footer>
            </article>
          </div>

          <div v-else class="config-card-grid">
            <article
              v-for="preset in managedPresets"
              :key="preset.id"
              class="config-card"
            >
              <header>
                <div>
                  <strong>{{ preset.name }}</strong>
                  <small>{{ preset.owner_type === 'system' ? '内置预设' : '个人预设' }}</small>
                </div>
                <span>{{ preset.id === chatSettings.presetId ? '当前' : preset.owner_type === 'system' ? '内置·只读' : '自定义' }}</span>
              </header>
              <p>{{ preset.description || preset.system_prompt || '内置提示词预设，完整提示词保持隐藏。' }}</p>
              <div class="config-card-meta">
                <span>{{ preset.owner_type === 'system' ? '只读' : '可编辑' }}</span>
                <span>{{ preset.is_default ? '默认' : '可选' }}</span>
              </div>
              <footer>
                <div>
                  <button type="button" @click="chooseManagedPreset(preset.id)">用于当前</button>
                  <button v-if="preset.owner_type === 'user'" type="button" @click="editPresetDraft(preset)">编辑</button>
                </div>
                <button
                  v-if="preset.owner_type === 'user'"
                  class="text-danger"
                  type="button"
                  @click="removeManagedPreset(preset.id)"
                >
                  删除
                </button>
              </footer>
            </article>
          </div>
        </section>
      </div>
    </teleport>

    <teleport to="body">
      <div
        v-if="itemDialogOpen"
        class="item-dialog-backdrop"
        role="presentation"
        @mousedown.self="closeItemDialog"
      >
        <form
          v-if="itemDialogOpen === 'model'"
          class="item-dialog"
          role="dialog"
          aria-modal="true"
          @submit.prevent="saveModelDraft"
        >
          <header>
            <strong>{{ editingModelId ? '编辑模型' : '添加模型' }}</strong>
            <button type="button" aria-label="关闭" @click="closeItemDialog">×</button>
          </header>
          <label>
            <span>名称</span>
            <input v-model="modelDraft.display_name" type="text" placeholder="DeepSeek Pro" />
          </label>
          <label>
            <span>描述</span>
            <input v-model="modelDraft.description" type="text" placeholder="用途或备注" />
          </label>
          <label>
            <span>Provider</span>
            <input v-model="modelDraft.provider" type="text" placeholder="deepseek / openai / ollama" />
          </label>
          <label>
            <span>模型名</span>
            <input v-model="modelDraft.model_name" type="text" placeholder="deepseek-chat" />
          </label>
          <label>
            <span>Base URL</span>
            <input v-model="modelDraft.base_url" type="text" placeholder="https://api.example.com/v1" />
          </label>
          <label>
            <span>独立密钥环境变量</span>
            <input v-model="modelDraft.api_key_env" type="text" placeholder="MY_MODEL_API_KEY" />
          </label>
          <div class="item-dialog-checks">
            <label><input v-model="modelDraft.enabled" type="checkbox" /> 启用</label>
            <label><input v-model="modelDraft.supports_stream" type="checkbox" /> 流式</label>
            <label><input v-model="modelDraft.supports_tools" type="checkbox" /> Tools</label>
            <label><input v-model="modelDraft.supports_vision" type="checkbox" /> Vision</label>
          </div>
          <button class="item-dialog-save" type="submit">保存</button>
        </form>

        <form
          v-else
          class="item-dialog"
          role="dialog"
          aria-modal="true"
          @submit.prevent="savePresetDraft"
        >
          <header>
            <strong>{{ editingPresetId ? '编辑预设' : '新建预设' }}</strong>
            <button type="button" aria-label="关闭" @click="closeItemDialog">×</button>
          </header>
          <label>
            <span>名称</span>
            <input v-model="presetDraft.name" type="text" placeholder="代码审查助手" />
          </label>
          <label>
            <span>描述</span>
            <input v-model="presetDraft.description" type="text" placeholder="用于解释、审查或写作" />
          </label>
          <label>
            <span>System Prompt</span>
            <textarea v-model="presetDraft.system_prompt" rows="5" placeholder="你是一名..." />
          </label>
          <label>
            <span>RAG Prompt Hint</span>
            <textarea v-model="presetDraft.rag_prompt_hint" rows="3" placeholder="回答时优先..." />
          </label>
          <button class="item-dialog-save" type="submit">保存</button>
        </form>
      </div>
    </teleport>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import {
  createChatSession,
  createChatModel,
  createChatPreset,
  deleteChatSession,
  deleteChatModel,
  deleteChatPreset,
  exportChatSession,
  listChatModels,
  listChatPresets,
  listChatSessions,
  listKnowledgeBases,
  manageChatModels,
  searchChatMessages,
  setDefaultChatModel,
  testChatModelConnection,
  updateChatModel,
  updateChatPreset,
  updateChatSession,
  updateChatSessionModel,
  updateChatSessionPreset,
} from './api/client'

const router = useRouter()
const sidebarCollapsed = ref(false)
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
const sidebarModelOptions = ref([])
const defaultModelId = ref('deepseek-chat')
const configModalOpen = ref('')
const configMessage = ref('')
const itemDialogOpen = ref('')
const managedModels = ref([])
const modelKeyMetrics = ref({ configured: 0, required: 0, missing: 0, independent: 0 })
const selectedManagedModelId = ref('')
const editingModelId = ref('')
const managedPresets = ref([])
const defaultPresetId = ref('default-assistant')
const selectedManagedPresetId = ref('')
const editingPresetId = ref('')
let historySearchTimer = null

const emptyModelDraft = () => ({
  provider: 'openai',
  display_name: '',
  model_name: '',
  base_url: '',
  api_key_env: '',
  description: '',
  supports_stream: true,
  supports_tools: false,
  supports_vision: false,
  enabled: true,
})

const emptyPresetDraft = () => ({
  name: '',
  description: '',
  system_prompt: '',
  rag_prompt_hint: '',
})

const modelDraft = reactive(emptyModelDraft())
const presetDraft = reactive(emptyPresetDraft())

const chatSettings = reactive({
  modelId: 'deepseek-chat',
  presetId: 'default-assistant',
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

const currentPresetLabel = computed(() =>
  managedPresets.value.find(preset => preset.id === chatSettings.presetId)?.name || '默认预设'
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

const selectedManagedModel = computed(() =>
  managedModels.value.find(model => model.id === selectedManagedModelId.value) || null
)

const selectedManagedPreset = computed(() =>
  managedPresets.value.find(preset => preset.id === selectedManagedPresetId.value) || null
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

async function loadModelManagement() {
  const data = await manageChatModels()
  managedModels.value = data.models
  modelKeyMetrics.value = data.key_metrics || { configured: 0, required: 0, missing: 0, independent: 0 }
  defaultModelId.value = data.default_model_id || defaultModelId.value
  if (!selectedManagedModelId.value || !managedModels.value.some(model => model.id === selectedManagedModelId.value)) {
    selectedManagedModelId.value = chatSettings.modelId || data.default_model_id || managedModels.value[0]?.id || ''
  }
  sidebarModelOptions.value = data.models
    .filter(model => model.enabled)
    .map(model => ({ value: model.id, label: model.display_name, provider: model.provider }))
}

async function loadPresetManagement() {
  const data = await listChatPresets()
  managedPresets.value = data.presets
  defaultPresetId.value = data.default_preset_id || managedPresets.value[0]?.id || 'default-assistant'
  if (!chatSettings.presetId) chatSettings.presetId = defaultPresetId.value
  if (!selectedManagedPresetId.value || !managedPresets.value.some(preset => preset.id === selectedManagedPresetId.value)) {
    selectedManagedPresetId.value = chatSettings.presetId || defaultPresetId.value || managedPresets.value[0]?.id || ''
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

async function downloadSessionMarkdown(session) {
  if (!session?.id) return
  try {
    sessionLoadError.value = ''
    const data = await exportChatSession(session.id)
    const blob = new Blob([data.content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = data.filename || `${session.title || 'chat-session'}.md`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  } catch (err) {
    sessionLoadError.value = err?.message || '导出失败'
  }
}

async function openConfigModal(name) {
  configModalOpen.value = name
  configMessage.value = ''
  try {
    if (name === 'model') {
      await loadModelManagement()
    } else {
      await loadPresetManagement()
    }
  } catch (err) {
    configMessage.value = err?.message || '配置加载失败'
  }
}

function closeConfigModal() {
  configModalOpen.value = ''
  configMessage.value = ''
  closeItemDialog()
}

function openItemDialog(name) {
  configMessage.value = ''
  if (name === 'model') {
    resetModelDraft()
    itemDialogOpen.value = 'model'
  } else {
    resetPresetDraft()
    itemDialogOpen.value = 'preset'
  }
}

function closeItemDialog() {
  itemDialogOpen.value = ''
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

async function chooseManagedModel(modelId) {
  await chooseSidebarModel(modelId)
  selectedManagedModelId.value = modelId
  configMessage.value = '已用于当前会话'
  await loadModelManagement()
}

async function setDefaultModel(modelId) {
  try {
    const model = await setDefaultChatModel(modelId)
    defaultModelId.value = model.id
    configMessage.value = '默认模型已更新'
    await loadModelManagement()
    await loadSidebarModels()
  } catch (err) {
    configMessage.value = err?.message || '默认模型更新失败'
  }
}

async function testManagedModel(modelId) {
  try {
    const result = await testChatModelConnection(modelId)
    configMessage.value = result.message
  } catch (err) {
    configMessage.value = err?.message || '连接测试失败'
  }
}

function resetModelDraft() {
  Object.assign(modelDraft, emptyModelDraft())
  editingModelId.value = ''
}

function editModelDraft(model) {
  editingModelId.value = model.id
  itemDialogOpen.value = 'model'
  Object.assign(modelDraft, {
    provider: model.provider || 'openai',
    display_name: model.display_name || '',
    model_name: model.model_name || '',
    base_url: model.base_url || '',
    api_key_env: '',
    description: model.description || '',
    supports_stream: Boolean(model.supports_stream),
    supports_tools: Boolean(model.supports_tools),
    supports_vision: Boolean(model.supports_vision),
    enabled: Boolean(model.enabled),
  })
}

async function saveModelDraft() {
  const payload = {
    provider: modelDraft.provider,
    display_name: modelDraft.display_name,
    model_name: modelDraft.model_name,
    base_url: modelDraft.base_url || null,
    description: modelDraft.description || '',
    supports_stream: modelDraft.supports_stream,
    supports_tools: modelDraft.supports_tools,
    supports_vision: modelDraft.supports_vision,
    enabled: modelDraft.enabled,
  }
  if (!editingModelId.value || modelDraft.api_key_env) {
    payload.api_key_env = modelDraft.api_key_env || null
  }
  try {
    const model = editingModelId.value
      ? await updateChatModel(editingModelId.value, payload)
      : await createChatModel(payload)
    selectedManagedModelId.value = model.id
    configMessage.value = editingModelId.value ? '模型已保存' : '模型已新增'
    resetModelDraft()
    closeItemDialog()
    await loadModelManagement()
    await loadSidebarModels()
  } catch (err) {
    configMessage.value = err?.message || '模型保存失败'
  }
}

async function removeManagedModel(modelId) {
  try {
    await deleteChatModel(modelId)
    if (selectedManagedModelId.value === modelId) selectedManagedModelId.value = ''
    configMessage.value = '模型已删除'
    await loadModelManagement()
    await loadSidebarModels()
  } catch (err) {
    configMessage.value = err?.message || '模型删除失败'
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

async function chooseManagedPreset(presetId) {
  const previous = chatSettings.presetId
  chatSettings.presetId = presetId
  try {
    const sessionId = await ensureSidebarSession()
    const updated = await updateChatSessionPreset(sessionId, presetId)
    chatSessions.value = chatSessions.value.map(session => session.id === updated.id ? updated : session)
    applySessionSettings(updated)
    selectedManagedPresetId.value = presetId
    configMessage.value = '已用于当前会话'
  } catch (err) {
    chatSettings.presetId = previous
    configMessage.value = err?.message || '预设切换失败'
  }
}

function resetPresetDraft() {
  Object.assign(presetDraft, emptyPresetDraft())
  editingPresetId.value = ''
}

function editPresetDraft(preset) {
  editingPresetId.value = preset.id
  itemDialogOpen.value = 'preset'
  Object.assign(presetDraft, {
    name: preset.name || '',
    description: preset.description || '',
    system_prompt: preset.system_prompt || '',
    rag_prompt_hint: preset.rag_prompt_hint || '',
  })
}

async function savePresetDraft() {
  const payload = {
    name: presetDraft.name,
    description: presetDraft.description,
    system_prompt: presetDraft.system_prompt,
    rag_prompt_hint: presetDraft.rag_prompt_hint || null,
  }
  try {
    const preset = editingPresetId.value
      ? await updateChatPreset(editingPresetId.value, payload)
      : await createChatPreset(payload)
    selectedManagedPresetId.value = preset.id
    configMessage.value = editingPresetId.value ? '预设已保存' : '预设已新增'
    resetPresetDraft()
    closeItemDialog()
    await loadPresetManagement()
  } catch (err) {
    configMessage.value = err?.message || '预设保存失败'
  }
}

async function removeManagedPreset(presetId) {
  try {
    await deleteChatPreset(presetId)
    if (chatSettings.presetId === presetId) chatSettings.presetId = defaultPresetId.value
    selectedManagedPresetId.value = ''
    configMessage.value = '预设已删除'
    await loadPresetManagement()
    await loadChatSessions()
  } catch (err) {
    configMessage.value = err?.message || '预设删除失败'
  }
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
  await loadPresetManagement()
  await loadKnowledgeBases()
  await loadChatSessions()
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
  min-height: 32px;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 10px 0;
  border: 1px solid transparent;
  border-radius: 8px;
  background: rgba(226, 232, 240, 0.5);
  padding: 6px 12px;
  color: #64748b;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease;
}

.sidebar-search-button:hover {
  border-color: rgba(203, 213, 225, 0.5);
  background: rgba(226, 232, 240, 0.8);
  color: #334155;
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

.sidebar-search-button kbd {
  flex: 0 0 auto;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.6);
  padding: 2px 6px;
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  line-height: 1.2;
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

.config-modal-backdrop {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.26);
  padding: 24px;
  backdrop-filter: blur(8px);
  z-index: 90;
}

.config-modal {
  display: grid;
  width: min(980px, 100%);
  max-height: min(86vh, 760px);
  grid-template-rows: auto auto 1fr;
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.92);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 30px 90px rgba(15, 23, 42, 0.28);
}

.config-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid #eef2f7;
  padding: 16px 18px;
}

.config-modal-header div {
  display: grid;
  min-width: 0;
  gap: 2px;
}

.config-modal-header strong {
  color: #0f172a;
  font-size: 16px;
  font-weight: 800;
}

.config-modal-header span {
  color: #64748b;
  font-size: 12px;
}

.config-modal-header button {
  display: grid;
  width: 32px;
  height: 32px;
  place-items: center;
  border: 0;
  border-radius: 9px;
  background: transparent;
  color: #94a3b8;
}

.config-modal-header button:hover {
  background: #f1f5f9;
  color: #475569;
}

.config-message {
  margin: 10px 14px 0;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
  padding: 9px 12px;
  color: #475569;
  font-size: 12px;
}

.config-page-backdrop {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: stretch;
  justify-content: center;
  background: rgba(15, 23, 42, 0.18);
  padding: 20px;
  backdrop-filter: blur(8px);
  z-index: 90;
}

.config-page {
  display: flex;
  width: min(1180px, 100%);
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.88);
  border-radius: 24px;
  background: #f8fafc;
  box-shadow: 0 30px 90px rgba(15, 23, 42, 0.24);
}

.config-page-header {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 18px;
  align-items: center;
  border-bottom: 1px solid rgba(226, 232, 240, 0.8);
  background: rgba(255, 255, 255, 0.78);
  padding: 18px 20px;
}

.config-back-button {
  min-height: 34px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: transparent;
  color: #64748b;
  padding: 0 10px;
  font-size: 13px;
  font-weight: 650;
}

.config-back-button:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.config-page-title {
  display: grid;
  min-width: 0;
  gap: 3px;
  text-align: center;
}

.config-page-title strong {
  color: #0f172a;
  font-size: 20px;
  font-weight: 850;
}

.config-page-title span {
  overflow: hidden;
  color: #64748b;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.config-primary-button {
  min-height: 38px;
  border: 0;
  border-radius: 12px;
  background: #047857;
  color: #ffffff;
  padding: 0 16px;
  font-size: 14px;
  font-weight: 700;
}

.config-primary-button:hover {
  background: #065f46;
}

.config-card-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  min-height: 0;
  overflow: auto;
  padding: 24px;
  scrollbar-width: thin;
  scrollbar-color: transparent transparent;
}

.config-card-grid:hover {
  scrollbar-color: rgba(203, 213, 225, 0.55) transparent;
}

.config-card-grid::-webkit-scrollbar {
  width: 4px;
}

.config-card-grid::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: transparent;
}

.config-card-grid:hover::-webkit-scrollbar-thumb {
  background: rgba(203, 213, 225, 0.55);
}

.config-card {
  display: flex;
  min-height: 218px;
  flex-direction: column;
  justify-content: space-between;
  border: 1px solid rgba(226, 232, 240, 0.8);
  border-radius: 16px;
  background: #ffffff;
  padding: 20px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}

.config-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 16px 38px rgba(15, 23, 42, 0.10);
  transform: translateY(-1px);
}

.config-card header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.config-card header div {
  display: grid;
  min-width: 0;
  gap: 3px;
}

.config-card header strong,
.config-card header small,
.config-card p {
  overflow: hidden;
  text-overflow: ellipsis;
}

.config-card header strong {
  color: #0f172a;
  font-size: 15px;
  font-weight: 800;
  white-space: nowrap;
}

.config-card header small {
  color: #94a3b8;
  font-size: 11px;
  white-space: nowrap;
}

.config-card header span {
  flex: 0 0 auto;
  height: fit-content;
  border-radius: 999px;
  background: #f1f5f9;
  padding: 3px 8px;
  color: #64748b;
  font-size: 10px;
  font-weight: 750;
}

.config-card p {
  display: -webkit-box;
  margin: 14px 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.6;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.config-card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 14px;
}

.config-card-meta span {
  border-radius: 999px;
  background: #f8fafc;
  padding: 3px 8px;
  color: #64748b;
  font-size: 10px;
}

.config-card footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-top: 1px solid #f1f5f9;
  padding-top: 12px;
}

.config-card footer div {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.config-card footer button {
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #475569;
  padding: 5px 7px;
  font-size: 11px;
  font-weight: 700;
}

.config-card footer button:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.config-card footer .text-danger {
  color: #dc2626;
}

.item-dialog-backdrop {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.4);
  padding: 20px;
  backdrop-filter: blur(3px);
  z-index: 100;
}

.item-dialog {
  display: grid;
  width: min(448px, 100%);
  max-height: min(86vh, 720px);
  gap: 12px;
  overflow: auto;
  border-radius: 16px;
  background: #ffffff;
  padding: 24px;
  box-shadow: 0 28px 80px rgba(15, 23, 42, 0.32);
}

.item-dialog header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.item-dialog header strong {
  color: #0f172a;
  font-size: 16px;
  font-weight: 850;
}

.item-dialog header button {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border: 0;
  border-radius: 8px;
  background: #f8fafc;
  color: #64748b;
}

.item-dialog label {
  display: grid;
  gap: 5px;
  color: #64748b;
  font-size: 12px;
}

.item-dialog input[type="text"],
.item-dialog textarea {
  width: 100%;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
  padding: 9px 11px;
  color: #0f172a;
  font-size: 13px;
  outline: 0;
}

.item-dialog input[type="text"]:focus,
.item-dialog textarea:focus {
  border-color: #cbd5e1;
  background: #ffffff;
}

.item-dialog textarea {
  resize: vertical;
}

.item-dialog-checks {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.item-dialog-checks label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.item-dialog-save {
  min-height: 40px;
  border: 0;
  border-radius: 12px;
  background: #047857;
  color: white;
  padding: 0 16px;
  font-size: 14px;
  font-weight: 750;
}

.item-dialog-save:hover {
  background: #065f46;
}

@media (max-width: 1024px) {
  .config-card-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .config-page-backdrop {
    padding: 0;
  }

  .config-page {
    height: 100%;
    border-radius: 0;
  }

  .config-page-header {
    grid-template-columns: 1fr;
    gap: 10px;
    text-align: left;
  }

  .config-page-title {
    text-align: left;
  }

  .config-card-grid {
    grid-template-columns: 1fr;
    padding: 16px;
  }
}

.sidebar-config-dock {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: auto;
  border-top: 1px solid rgba(226, 232, 240, 0.6);
  padding: 8px 0 4px;
}

.sidebar-config-bar {
  order: 2;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  border-radius: 12px;
  background: rgba(226, 232, 240, 0.4);
  padding: 6px;
  color: #475569;
  font-size: 11px;
}

.sidebar-config-pill {
  display: flex;
  min-width: 0;
  min-height: 30px;
  align-items: center;
  gap: 6px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #475569;
  padding: 6px;
  transition: background 0.14s ease, color 0.14s ease, box-shadow 0.14s ease;
}

.sidebar-config-pill:hover,
.sidebar-config-pill.active {
  background: #ffffff;
  color: #0f172a;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.sidebar-config-pill span {
  flex: 0 0 auto;
}

.sidebar-config-pill strong,
.sidebar-option strong,
.sidebar-option small,
.sidebar-option > span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sidebar-config-pill strong {
  min-width: 0;
  font-size: 11px;
  font-weight: 650;
}

.sidebar-config-panel {
  order: 1;
  display: grid;
  max-height: min(40vh, 320px);
  gap: 5px;
  overflow: auto;
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.98);
  padding: 6px;
  box-shadow: 0 16px 42px rgba(15, 23, 42, 0.14);
  scrollbar-width: thin;
  scrollbar-color: transparent transparent;
}

.sidebar-config-panel:hover {
  scrollbar-color: rgba(203, 213, 225, 0.52) transparent;
}

.sidebar-config-panel::-webkit-scrollbar {
  width: 4px;
}

.sidebar-config-panel::-webkit-scrollbar-track {
  background: transparent;
}

.sidebar-config-panel::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: transparent;
}

.sidebar-config-panel:hover::-webkit-scrollbar-thumb {
  background: rgba(203, 213, 225, 0.52);
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
