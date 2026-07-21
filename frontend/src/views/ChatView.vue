<template>
  <!--
    对话页面 — Chat + Trace
    五模式选择器、SSE 流式输出、引用抽屉、Trace 时间线
  -->
  <div class="chat-page">
    <!-- 侧边栏：模式选择 + 知识库选择 -->
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
        <option value="kb_001">产品技术手册</option>
        <option value="kb_002">公司内部制度</option>
        <option value="kb_003">行业白皮书合集</option>
      </select>

      <h3>高级选项</h3>
      <label class="option-row">
        <span>Top-K</span>
        <input type="number" v-model.number="options.top_k" min="1" max="20" />
      </label>
      <label class="option-row" v-if="selectedMode === 'advanced' || selectedMode === 'modular'">
        <input type="checkbox" v-model="options.rewrite_enabled" />
        <span>查询改写</span>
      </label>
      <label class="option-row" v-if="selectedMode === 'advanced' || selectedMode === 'modular'">
        <input type="checkbox" v-model="options.rerank_enabled" />
        <span>重排序</span>
      </label>
    </aside>

    <!-- 主体：消息区和输入框 -->
    <main class="chat-main">
      <div class="chat-messages" ref="msgContainer">
        <div v-if="messages.length === 0" class="welcome">
          <h2>LangChain RAG 实验平台</h2>
          <p>选择左侧模式，输入问题开始测试五种 RAG 范式</p>
        </div>

        <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
          <div class="msg-role">{{ msg.role === 'user' ? '你' : 'AI' }}</div>
          <div class="msg-content" v-html="msg.content"></div>

          <!-- 引用抽屉（仅 AI 消息显示） -->
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

          <!-- Trace 时间线缩略 -->
          <div v-if="msg.trace?.length" class="trace-mini">
            <details>
              <summary>🔍 执行轨迹 ({{ msg.trace.length }} 阶段)</summary>
              <div class="trace-timeline">
                <div v-for="t in msg.trace" :key="t.stage" class="trace-node">
                  <span class="trace-stage">{{ t.stage }}</span>
                  <span class="trace-duration">{{ t.duration_ms }}ms</span>
                  <span class="trace-summary">{{ t.output_summary }}</span>
                </div>
              </div>
            </details>
          </div>
        </div>

        <div v-if="streaming" class="message assistant streaming">
          <div class="msg-role">AI ⏳</div>
          <div class="msg-content">{{ streamingText || '思考中...' }}</div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="chat-input-bar">
        <textarea
          v-model="query"
          @keydown.enter.exact.prevent="sendQuery"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          rows="2"
          :disabled="streaming"
        />
        <button @click="sendQuery" :disabled="!query.trim() || streaming" class="send-btn">
          {{ streaming ? '发送中...' : '发送' }}
        </button>
        <button v-if="streaming" @click="cancelQuery" class="cancel-btn">取消</button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { ragQueryStream } from '../api/client'

// ===== 模式列表 =====
const modes = [
  { value: 'naive', label: 'Naive' },
  { value: 'advanced', label: 'Advanced' },
  { value: 'modular', label: 'Modular' },
  { value: 'graph', label: 'GraphRAG' },
  { value: 'agentic', label: 'Agentic' },
]

// ===== 响应式状态 =====
const selectedMode = ref('naive')
const selectedKbId = ref('kb_001')
const query = ref('')
const messages = ref([])
const streaming = ref(false)
const streamingText = ref('')
const msgContainer = ref(null)

// 高级选项
const options = ref({
  top_k: 5,
  rewrite_enabled: false,
  rerank_enabled: false,
})

// AbortController 用于取消 SSE
let abortCtrl = null

// ===== 发送查询 =====
async function sendQuery() {
  const q = query.value.trim()
  if (!q || streaming.value) return

  // 添加用户消息
  messages.value.push({ role: 'user', content: q })
  query.value = ''
  streaming.value = true
  streamingText.value = ''

  const currentMsg = { role: 'assistant', content: '', citations: [], trace: [] }
  messages.value.push(currentMsg)

  abortCtrl = new AbortController()

  await ragQueryStream(
    {
      query: q,
      kb_id: selectedKbId.value,
      mode: selectedMode.value,
      options: options.value,
    },
    // onTrace
    (event) => { currentMsg.trace.push(event) },
    // onChunk
    (delta) => {
      streamingText.value += delta
      currentMsg.content += delta
    },
    // onDone
    (data) => {
      currentMsg.citations = data.citations
      streaming.value = false
      streamingText.value = ''
    },
    // onError
    (err) => {
      currentMsg.content = `❌ 错误：${err.message}`
      streaming.value = false
    },
    abortCtrl.signal,
  )

  await nextTick()
  scrollToBottom()
}

// ===== 取消查询 =====
function cancelQuery() {
  abortCtrl?.abort()
  streaming.value = false
}

// ===== 滚动到底部 =====
function scrollToBottom() {
  if (msgContainer.value) {
    msgContainer.value.scrollTop = msgContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-page { display: flex; height: 100%; }
.chat-sidebar {
  width: 260px; padding: 16px; border-right: 1px solid var(--border-color);
  overflow-y: auto; flex-shrink: 0;
}
.chat-sidebar h3 { margin: 16px 0 8px; font-size: 14px; color: var(--text-muted); }
.chat-sidebar h3:first-child { margin-top: 0; }
.mode-list { display: flex; flex-direction: column; gap: 4px; }
.mode-btn {
  padding: 8px 12px; border: 1px solid var(--border-color); border-radius: 6px;
  background: var(--bg-secondary); cursor: pointer; text-align: left; font-size: 13px;
}
.mode-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
.kb-select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid var(--border-color); }
.option-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 13px; }
.option-row input[type="number"] { width: 60px; padding: 4px; }

.chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.chat-messages { flex: 1; overflow-y: auto; padding: 20px; }
.welcome { text-align: center; margin-top: 120px; color: var(--text-muted); }
.welcome h2 { font-size: 24px; margin-bottom: 8px; }

.message { margin-bottom: 16px; max-width: 80%; }
.message.user { margin-left: auto; }
.message.user .msg-role { color: var(--accent); }
.message.assistant .msg-role { color: var(--success); }
.msg-role { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.msg-content { padding: 12px 16px; border-radius: 12px; line-height: 1.6; }
.message.user .msg-content { background: var(--accent); color: white; }
.message.assistant .msg-content { background: var(--bg-secondary); }

.citations { margin-top: 8px; font-size: 13px; }
.citations details { padding: 8px; background: var(--bg-tertiary); border-radius: 6px; }
.citations blockquote { border-left: 3px solid var(--accent); padding-left: 12px; margin: 4px 0; color: var(--text-muted); }

.trace-mini { margin-top: 4px; font-size: 12px; }
.trace-mini details { padding: 6px; background: #f0f6ff; border-radius: 6px; }
.trace-timeline { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.trace-node { padding: 4px 8px; background: #e8f0fe; border-radius: 4px; }
.trace-stage { font-weight: 600; margin-right: 4px; }
.trace-duration { color: var(--text-muted); margin-right: 4px; }

.chat-input-bar { display: flex; gap: 8px; padding: 16px; border-top: 1px solid var(--border-color); }
.chat-input-bar textarea {
  flex: 1; padding: 10px; border: 1px solid var(--border-color); border-radius: 8px;
  resize: none; font-family: inherit;
}
.send-btn, .cancel-btn { padding: 8px 20px; border: none; border-radius: 8px; cursor: pointer; }
.send-btn { background: var(--accent); color: white; }
.cancel-btn { background: var(--danger); color: white; }
</style>
