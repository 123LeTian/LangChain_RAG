<template>
  <div class="chat-page">
    <aside class="chat-sidebar">
      <h3>RAG 模式</h3>
      <div class="mode-list">
        <button
          v-for="m in modes" :key="m.value"
          :class="['mode-btn', { active: selectedMode === m.value }]"
          @click="selectedMode = m.value"
        >
          {{ m.label }}
        </button>
      </div>

      <h3>知识库</h3>
      <select v-model="selectedKbId" class="kb-select">
        <option v-for="kb in kbList" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
      </select>

      <h3>模块开关</h3>
      <label class="option-row" v-for="mod in moduleList" :key="mod.key">
        <input type="checkbox" v-model="options[mod.key]" :disabled="mod.disabled" />
        <span :class="{ disabled: mod.disabled }">{{ mod.label }}</span>
      </label>

      <h3>参数</h3>
      <label class="option-row">
        <span>检索 Top-K</span>
        <input type="number" v-model.number="options.top_k" min="1" max="20" />
      </label>
      <label class="option-row">
        <span>重排 Top-K</span>
        <input type="number" v-model.number="options.rerank_top_k" min="1" max="20" />
      </label>

      <h3>对比模式</h3>
      <label class="option-row">
        <input type="checkbox" v-model="compareMode" />
        <span>启用配置对比</span>
      </label>
      <div v-if="compareMode" class="compare-configs">
        <div class="compare-config">
          <div class="compare-label">配置 A</div>
          <label v-for="mod in moduleList" :key="'a-'+mod.key" class="mini-row">
            <input type="checkbox" v-model="configA[mod.key]" :disabled="mod.disabled" />
            <span>{{ mod.short }}</span>
          </label>
        </div>
        <div class="compare-config">
          <div class="compare-label">配置 B</div>
          <label v-for="mod in moduleList" :key="'b-'+mod.key" class="mini-row">
            <input type="checkbox" v-model="configB[mod.key]" :disabled="mod.disabled" />
            <span>{{ mod.short }}</span>
          </label>
        </div>
      </div>

      <div v-if="configError" class="config-error">{{ configError }}</div>
    </aside>

    <main class="chat-main">
      <div class="chat-messages" ref="msgContainer">
        <div v-if="messages.length === 0" class="welcome">
          <h2>LangChain RAG 实验平台</h2>
          <p>选择左侧模式，输入问题开始测试</p>
          <p style="margin-top:12px;font-size:13px;">5个模块开关可自由组合，勾选对比模式可一键对比两套配置</p>
        </div>

        <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
          <div class="msg-role">{{ msg.role === 'user' ? '你' : 'AI' }}</div>
          <div class="msg-content" v-html="msg.content"></div>

          <!-- Pipeline 配置标签 -->
          <div v-if="msg.detail?.pipeline_config" class="pipeline-tags">
            <span class="pipe-label">Pipeline:</span>
            <span
              v-for="mod in ['rewrite','retrieve','rerank','compress','verify','generate']"
              :key="mod"
              :class="['pipe-tag', { on: msg.detail.pipeline_config[mod], off: !msg.detail.pipeline_config[mod] }]"
            >{{ mod }}</span>
          </div>

          <!-- 检索详情 -->
          <div v-if="msg.detail" class="detail-panel">
            <details open>
              <summary>📋 检索详情</summary>

              <div class="detail-section">
                <div class="detail-label">原始问题</div>
                <div class="detail-value original">{{ msg.detail.original_query }}</div>
                <template v-if="msg.detail.pipeline_config?.rewrite">
                  <div class="detail-label">改写问题 ({{ msg.detail.rewritten_queries?.length || 0 }})</div>
                  <div class="rewrite-list">
                    <div v-for="(rq, idx) in msg.detail.rewritten_queries" :key="idx" class="rewrite-item">{{ rq }}</div>
                  </div>
                </template>
                <span v-else class="skipped-tag">改写已关闭</span>
              </div>

              <div class="detail-section">
                <div class="hit-comparison">
                  <div class="hit-column">
                    <div class="hit-column-title">重排前 Top-{{ msg.detail.pre_rerank_top_k?.length || 0 }}</div>
                    <div v-for="h in msg.detail.pre_rerank_top_k" :key="h.chunk_id" class="hit-item">
                      <span class="hit-rank">#{{ h.rank }}</span>
                      <span class="hit-score">{{ h.score }}</span>
                      <span class="hit-text">{{ h.text }}</span>
                    </div>
                    <span v-if="msg.detail.pipeline_config && !msg.detail.pipeline_config.retrieve" class="skipped-tag">检索已关闭</span>
                  </div>
                  <div class="hit-column" v-if="msg.detail.pipeline_config?.rerank">
                    <div class="hit-column-title">重排后 Top-{{ msg.detail.post_rerank_top_k?.length || 0 }}</div>
                    <div v-for="h in msg.detail.post_rerank_top_k" :key="h.chunk_id" class="hit-item reranked">
                      <span class="hit-rank">#{{ h.rank }}</span>
                      <span class="hit-score">{{ h.score }}</span>
                      <span class="hit-text">{{ h.text }}</span>
                    </div>
                  </div>
                  <span v-else-if="msg.detail.pipeline_config" class="skipped-tag">重排已关闭</span>
                </div>
              </div>

              <!-- Verify 结果 -->
              <div v-if="msg.detail.verify_result" class="detail-section">
                <div class="detail-label">答案验证 (Verify)</div>
                <div :class="['verify-result', msg.detail.verify_result.passed ? 'pass' : 'fail']">
                  <span class="verify-status">{{ msg.detail.verify_result.passed ? '✓ 通过' : '✗ 未通过' }}</span>
                  <div v-if="msg.detail.verify_result.issues?.length" class="verify-issues">
                    <div v-for="(issue, idx) in msg.detail.verify_result.issues" :key="idx" class="verify-issue">• {{ issue }}</div>
                  </div>
                </div>
              </div>
            </details>
          </div>

          <div v-if="msg.citations?.length" class="citations">
            <details>
              <summary>📎 {{ msg.citations.length }} 条引用来源</summary>
              <ul>
                <li v-for="cit in msg.citations" :key="cit.chunk_id">
                  📄 {{ cit.filename }} {{ cit.page ? `p.${cit.page}` : '' }}
                  <blockquote>{{ cit.quote }}</blockquote>
                </li>
              </ul>
            </details>
          </div>

          <div v-if="msg.trace?.length" class="trace-mini">
            <details>
              <summary>🔍 执行轨迹 ({{ msg.trace.length }} 阶段)</summary>
              <div class="trace-timeline">
                <div v-for="(t, idx) in msg.trace" :key="idx" :class="['trace-node', { skipped: t.output_summary?.includes('skipped') }]">
                  <span class="trace-stage">{{ t.stage }}</span>
                  <span class="trace-duration">{{ t.duration_ms }}ms</span>
                  <span class="trace-summary">{{ t.output_summary }}</span>
                </div>
              </div>
              <div v-if="msg.detail?.active_modules" class="active-modules">
                活动模块顺序: {{ msg.detail.active_modules.join(' → ') }}
              </div>
            </details>
          </div>
        </div>

        <div v-if="streaming" class="message assistant streaming">
          <div class="msg-role">AI …</div>
          <div class="msg-content">{{ streamingText || '思考中...' }}</div>
        </div>
      </div>

      <div class="chat-input-bar">
        <textarea
          v-model="query"
          @keydown.enter.exact.prevent="sendQuery"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          rows="2"
          :disabled="streaming"
        />
        <button @click="sendQuery" :disabled="!query.trim() || streaming || !!configError" class="send-btn">
          {{ streaming ? '发送中...' : (compareMode ? '对比发送' : '发送') }}
        </button>
        <button v-if="streaming" @click="cancelQuery" class="cancel-btn">取消</button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import { ragQueryStream, ragCompareStream, listKnowledgeBases } from '../api/client'

const modes = [
  { value: 'naive', label: 'Naive' },
  { value: 'advanced', label: 'Advanced' },
  { value: 'modular', label: 'Modular' },
  { value: 'graph', label: 'GraphRAG' },
  { value: 'agentic', label: 'Agentic' },
]

const moduleList = [
  { key: 'rewrite_enabled',  label: '查询改写 (Rewrite)',  short: 'RW', disabled: false },
  { key: 'retrieve_enabled', label: '检索 (Retrieve)',     short: 'RT', disabled: false },
  { key: 'rerank_enabled',   label: '重排序 (Rerank)',     short: 'RR', disabled: false },
  { key: 'compress_enabled', label: '压缩 (Compress)',     short: 'CP', disabled: false },
  { key: 'verify_enabled',   label: '验证 (Verify)',       short: 'VF', disabled: false },
]

const selectedMode = ref('modular')
const selectedKbId = ref('')
const query = ref('')
const messages = ref([])
const streaming = ref(false)
const streamingText = ref('')
const msgContainer = ref(null)
const kbList = ref([])
const compareMode = ref(false)

const options = ref({
  rewrite_enabled: true,
  retrieve_enabled: true,
  rerank_enabled: true,
  compress_enabled: true,
  verify_enabled: false,
  top_k: 10,
  rerank_top_k: 5,
})

const configA = ref({
  rewrite_enabled: true, retrieve_enabled: true, rerank_enabled: true,
  compress_enabled: true, verify_enabled: false, top_k: 10, rerank_top_k: 5,
})
const configB = ref({
  rewrite_enabled: false, retrieve_enabled: true, rerank_enabled: false,
  compress_enabled: false, verify_enabled: false, top_k: 10, rerank_top_k: 5,
})

let abortCtrl = null

onMounted(async () => {
  try {
    kbList.value = await listKnowledgeBases()
    if (kbList.value.length > 0) selectedKbId.value = kbList.value[0].id
  } catch (e) { console.error('Failed to load KBs:', e) }
})

// Validate config combinations
const configError = computed(() => {
  const cfg = compareMode.value ? null : options.value
  if (!cfg) {
    // Validate both compare configs
    for (const [name, c] of [['A', configA.value], ['B', configB.value]]) {
      if (c.rerank_enabled && !c.retrieve_enabled)
        return `配置${name}: 开启 rerank 必须先开启 retrieve`
      if (c.compress_enabled && !c.retrieve_enabled)
        return `配置${name}: 开启 compress 必须先开启 retrieve`
    }
    return null
  }
  if (cfg.rerank_enabled && !cfg.retrieve_enabled)
    return '开启 rerank 必须先开启 retrieve'
  if (cfg.compress_enabled && !cfg.retrieve_enabled)
    return '开启 compress 必须先开启 retrieve'
  return null
})

async function sendQuery() {
  const q = query.value.trim()
  if (!q || streaming.value || configError.value) return

  messages.value.push({ role: 'user', content: q })
  query.value = ''
  streaming.value = true
  streamingText.value = ''

  if (compareMode.value) {
    await sendCompare(q)
  } else {
    await sendSingle(q)
  }

  await nextTick()
  scrollToBottom()
}

async function sendSingle(q) {
  const currentMsg = { role: 'assistant', content: '', citations: [], trace: [], detail: null }
  messages.value.push(currentMsg)
  abortCtrl = new AbortController()

  await ragQueryStream(
    { query: q, kb_id: selectedKbId.value, mode: selectedMode.value, options: options.value },
    (event) => { currentMsg.trace.push(event) },
    (delta) => { streamingText.value += delta; currentMsg.content += delta },
    (data) => { currentMsg.citations = data.citations; streaming.value = false; streamingText.value = '' },
    (err) => { currentMsg.content = `❌ 错误：${err.message}`; streaming.value = false },
    abortCtrl.signal,
    (detail) => { currentMsg.detail = detail },
  )
}

async function sendCompare(q) {
  const msgA = { role: 'assistant', content: '', citations: [], trace: [], detail: null }
  const msgB = { role: 'assistant', content: '', citations: [], trace: [], detail: null }

  // Insert separator
  messages.value.push({ role: 'assistant', content: '═══ 配置对比 ═══', citations: [], trace: [], detail: null, isSeparator: true })
  messages.value.push({ ...msgA, content: '' })
  messages.value.push({ ...msgB, content: '' })

  const idxA = messages.value.length - 2
  const idxB = messages.value.length - 1

  abortCtrl = new AbortController()

  try {
    await ragCompareStream(
      { query: q, kb_id: selectedKbId.value, mode: 'modular', config_a: configA.value, config_b: configB.value },
      abortCtrl.signal,
      (data) => {
        // config_a result
        messages.value[idxA].content = data.answer
        messages.value[idxA].citations = data.citations
        messages.value[idxA].trace = data.trace
        messages.value[idxA].detail = data.detail
      },
      (data) => {
        // config_b result
        messages.value[idxB].content = data.answer
        messages.value[idxB].citations = data.citations
        messages.value[idxB].trace = data.trace
        messages.value[idxB].detail = data.detail
      },
    )
  } catch (err) {
    messages.value[idxA].content = `❌ 错误：${err.message}`
  }
  streaming.value = false
}

function cancelQuery() {
  abortCtrl?.abort()
  streaming.value = false
}

function scrollToBottom() {
  if (msgContainer.value) msgContainer.value.scrollTop = msgContainer.value.scrollHeight
}
</script>

<style scoped>
.chat-page { display: flex; height: 100%; }
.chat-sidebar { width: 280px; padding: 16px; border-right: 1px solid var(--border-color); overflow-y: auto; flex-shrink: 0; }
.chat-sidebar h3 { margin: 16px 0 8px; font-size: 14px; color: var(--text-muted); }
.chat-sidebar h3:first-child { margin-top: 0; }
.mode-list { display: flex; flex-direction: column; gap: 4px; }
.mode-btn { padding: 8px 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-secondary); cursor: pointer; text-align: left; font-size: 13px; }
.mode-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
.kb-select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid var(--border-color); }
.option-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 13px; }
.option-row input[type="number"] { width: 50px; padding: 4px; }
.disabled { color: #999; }
.compare-configs { display: flex; gap: 8px; margin-top: 8px; }
.compare-config { flex: 1; padding: 6px; background: var(--bg-secondary); border-radius: 6px; }
.compare-label { font-size: 12px; font-weight: 700; margin-bottom: 4px; }
.mini-row { display: flex; align-items: center; gap: 4px; font-size: 11px; margin-top: 2px; }
.config-error { margin-top: 8px; padding: 6px; background: #ffebee; color: #c62828; border-radius: 4px; font-size: 12px; }

.chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.chat-messages { flex: 1; overflow-y: auto; padding: 20px; }
.welcome { text-align: center; margin-top: 120px; color: var(--text-muted); }
.welcome h2 { font-size: 24px; margin-bottom: 8px; }

.message { margin-bottom: 16px; max-width: 85%; }
.message.user { margin-left: auto; }
.message.user .msg-role { color: var(--accent); }
.message.assistant .msg-role { color: var(--success); }
.msg-role { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.msg-content { padding: 12px 16px; border-radius: 12px; line-height: 1.6; }
.message.user .msg-content { background: var(--accent); color: white; }
.message.assistant .msg-content { background: var(--bg-secondary); }

.pipeline-tags { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.pipe-label { font-size: 11px; color: #666; font-weight: 600; }
.pipe-tag { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 600; }
.pipe-tag.on { background: #c8e6c9; color: #2e7d32; }
.pipe-tag.off { background: #ffcdd2; color: #c62828; text-decoration: line-through; }

.detail-panel { margin-top: 8px; font-size: 13px; }
.detail-panel details { padding: 10px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0; }
.detail-panel summary { cursor: pointer; font-weight: 600; color: #333; }
.detail-section { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }
.detail-label { font-size: 12px; color: #666; font-weight: 600; margin-bottom: 4px; }
.detail-value.original { padding: 6px 10px; background: #e3f2fd; border-radius: 4px; margin-bottom: 6px; }
.rewrite-list { display: flex; flex-direction: column; gap: 4px; }
.rewrite-item { padding: 4px 10px; background: #fff3e0; border-radius: 4px; font-size: 12px; }
.skipped-tag { display: inline-block; padding: 2px 8px; background: #ffebee; color: #c62828; border-radius: 4px; font-size: 11px; margin-top: 4px; }

.hit-comparison { display: flex; gap: 12px; margin-top: 6px; }
.hit-column { flex: 1; }
.hit-column-title { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; padding-bottom: 4px; border-bottom: 2px solid #ddd; }
.hit-item { padding: 4px 6px; background: #fafafa; border-radius: 4px; margin-bottom: 3px; font-size: 11px; display: flex; gap: 4px; align-items: flex-start; }
.hit-item.reranked { background: #e8f5e9; }
.hit-rank { font-weight: 700; color: #1976d2; min-width: 28px; }
.hit-score { color: #e65100; font-weight: 600; min-width: 50px; }
.hit-text { color: #555; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.verify-result { padding: 8px; border-radius: 4px; }
.verify-result.pass { background: #e8f5e9; }
.verify-result.fail { background: #ffebee; }
.verify-status { font-weight: 700; }
.verify-issues { margin-top: 4px; }
.verify-issue { font-size: 11px; color: #c62828; margin-top: 2px; }

.citations { margin-top: 8px; font-size: 13px; }
.citations details { padding: 8px; background: var(--bg-tertiary); border-radius: 6px; }
.citations blockquote { border-left: 3px solid var(--accent); padding-left: 12px; margin: 4px 0; color: var(--text-muted); }

.trace-mini { margin-top: 4px; font-size: 12px; }
.trace-mini details { padding: 6px; background: #f0f6ff; border-radius: 6px; }
.trace-timeline { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.trace-node { padding: 4px 8px; background: #e8f0fe; border-radius: 4px; }
.trace-node.skipped { background: #f5f5f5; opacity: 0.6; }
.trace-stage { font-weight: 600; margin-right: 4px; }
.trace-duration { color: var(--text-muted); margin-right: 4px; }
.active-modules { margin-top: 6px; font-size: 11px; color: #666; font-style: italic; }

.chat-input-bar { display: flex; gap: 8px; padding: 16px; border-top: 1px solid var(--border-color); }
.chat-input-bar textarea { flex: 1; padding: 10px; border: 1px solid var(--border-color); border-radius: 8px; resize: none; font-family: inherit; }
.send-btn, .cancel-btn { padding: 8px 20px; border: none; border-radius: 8px; cursor: pointer; }
.send-btn { background: var(--accent); color: white; }
.cancel-btn { background: var(--danger); color: white; }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>