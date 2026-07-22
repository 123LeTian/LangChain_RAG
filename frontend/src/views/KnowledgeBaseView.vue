<template>
  <section class="kb-workbench">
    <header class="kb-metrics">
      <article>
        <span>总知识库</span>
        <strong>{{ kbs.length }}</strong>
      </article>
      <article>
        <span>总文档</span>
        <strong>{{ totalDocs }}</strong>
      </article>
      <article>
        <span>总 Chunk</span>
        <strong>{{ totalChunks }}</strong>
      </article>
      <article>
        <span>向量库状态</span>
        <strong class="health">正常</strong>
      </article>
    </header>

    <div class="kb-layout">
      <aside class="kb-rail">
        <button class="btn-primary create-button" type="button" @click="showCreateForm = !showCreateForm">+ 创建知识库</button>

        <form v-if="showCreateForm" class="create-form" @submit.prevent="createKB">
          <input v-model="newKB.name" placeholder="知识库名称" required />
          <input v-model="newKB.description" placeholder="描述" />
          <div>
            <button class="btn-primary" type="submit">确认</button>
            <button class="btn-ghost" type="button" @click="showCreateForm = false">取消</button>
          </div>
        </form>

        <input v-model="searchText" class="kb-search" placeholder="搜索知识库" />

        <div class="kb-list">
          <button
            v-for="kb in filteredKBs"
            :key="kb.id"
            :class="['kb-card', { active: selectedKB?.id === kb.id }]"
            type="button"
            @click="selectKB(kb.id)"
          >
            <span class="kb-card-title">{{ kb.name }}</span>
            <small>{{ kb.description || '暂无描述' }}</small>
            <span>{{ kb.doc_count || 0 }} 文档 · {{ kb.chunk_count || 0 }} 切片</span>
          </button>
        </div>
      </aside>

      <main class="kb-detail">
        <div v-if="selectedKB" class="kb-detail-inner">
          <header class="detail-head">
            <div>
              <h1>{{ selectedKB.name }}</h1>
              <p>{{ selectedKB.description || '文档与索引管理' }}</p>
            </div>
            <div class="detail-actions">
              <button class="btn-secondary" type="button" @click="triggerUpload">上传文档</button>
              <button class="btn-secondary" type="button" @click="rebuildIndex(selectedKB.id)">重建索引</button>
              <button class="btn-danger" type="button" @click="removeKB(selectedKB.id)">删除知识库</button>
            </div>
          </header>

          <div class="tabs">
            <button :class="{ active: activeTab === 'docs' }" type="button" @click="activeTab = 'docs'">文档列表</button>
            <button :class="{ active: activeTab === 'chunks' }" type="button" @click="activeTab = 'chunks'">切片预览</button>
            <button :class="{ active: activeTab === 'config' }" type="button" @click="activeTab = 'config'">索引配置</button>
          </div>

          <div v-if="activeTab === 'docs'" class="table-shell">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>文件名</th>
                  <th>类型</th>
                  <th>大小</th>
                  <th>Chunk 数</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="doc in docs" :key="doc.id">
                  <td>{{ doc.filename }}</td>
                  <td><span class="file-type">{{ doc.type || fileType(doc.filename) }}</span></td>
                  <td>{{ formatSize(doc.size_bytes || 0) }}</td>
                  <td>{{ doc.chunk_count || 0 }}</td>
                  <td><span :class="['status-badge', doc.status || 'uploaded']">{{ doc.status || 'uploaded' }}</span></td>
                  <td class="row-actions">
                    <button class="btn-ghost" type="button">预览</button>
                    <button class="btn-danger" type="button" @click="removeDoc(doc.id)">删除</button>
                  </td>
                </tr>
              </tbody>
            </table>
            <div v-if="!docs.length" class="empty-state">当前知识库还没有文档。</div>
          </div>

          <div v-else-if="activeTab === 'chunks'" class="chunk-grid">
            <article v-for="chunk in previewChunks" :key="chunk.id">
              <span>{{ chunk.id }}</span>
              <p>{{ chunk.text }}</p>
              <small>vector_id: {{ chunk.vectorId }}</small>
            </article>
          </div>

          <div v-else class="index-config">
            <label>
              <span>切片大小</span>
              <input value="800 tokens" readonly />
            </label>
            <label>
              <span>重叠窗口</span>
              <input value="120 tokens" readonly />
            </label>
            <label>
              <span>Embedding</span>
              <input value="BGE / OpenAI Adapter" readonly />
            </label>
          </div>

          <button class="upload-zone" type="button" @click="triggerUpload">
            <strong>拖拽或点击上传 PDF / Word / TXT</strong>
            <span>上传后自动解析、切片并写入索引。</span>
          </button>
        </div>

        <div v-else class="empty-state">创建或选择一个知识库。</div>
      </main>
    </div>

    <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md" hidden @change="handleUpload" />
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import {
  createIndex,
  createKnowledgeBase,
  deleteDocument,
  deleteKnowledgeBase,
  listDocuments,
  listKnowledgeBases,
  uploadDocument,
} from '../api/client'

const props = defineProps({
  knowledgeBases: {
    type: Array,
    default: () => [],
  },
})

const emit = defineEmits(['knowledge-bases-updated'])

const kbs = ref([])
const docs = ref([])
const selectedKB = ref(null)
const showCreateForm = ref(false)
const searchText = ref('')
const activeTab = ref('docs')
const fileInput = ref(null)
const newKB = ref({ name: '', description: '' })

const filteredKBs = computed(() => {
  if (!searchText.value) return kbs.value
  return kbs.value.filter(kb => kb.name?.includes(searchText.value))
})

const totalDocs = computed(() => kbs.value.reduce((sum, kb) => sum + Number(kb.doc_count || 0), 0))
const totalChunks = computed(() => kbs.value.reduce((sum, kb) => sum + Number(kb.chunk_count || 0), 0))
const previewChunks = computed(() => {
  const source = docs.value.slice(0, 6)
  if (!source.length) {
    return [
      { id: 'Chunk #1', text: '星河计划的项目负责人是李明，预算为 128 万元。', vectorId: 'vec_demo_001' },
      { id: 'Chunk #2', text: '项目里程碑包含需求评审、原型验证、上线验收三个阶段。', vectorId: 'vec_demo_002' },
    ]
  }
  return source.map((doc, index) => ({
    id: `Chunk #${index + 1}`,
    text: `${doc.filename} 的切片预览，包含解析后的局部文本与检索上下文。`,
    vectorId: `vec_${doc.id || index}`,
  }))
})

watch(() => props.knowledgeBases, (value) => {
  kbs.value = [...value]
  if (!selectedKB.value && kbs.value.length) selectKB(kbs.value[0].id)
}, { immediate: true })

onMounted(loadKBs)

async function loadKBs() {
  try {
    kbs.value = await listKnowledgeBases()
    if (!selectedKB.value && kbs.value.length) await selectKB(kbs.value[0].id)
  } catch {
    kbs.value = props.knowledgeBases.length ? [...props.knowledgeBases] : []
  }
}

async function selectKB(id) {
  selectedKB.value = kbs.value.find(kb => kb.id === id) || null
  activeTab.value = 'docs'
  if (!selectedKB.value) {
    docs.value = []
    return
  }
  try {
    docs.value = await listDocuments(id)
  } catch {
    docs.value = []
  }
}

async function createKB() {
  await createKnowledgeBase(newKB.value.name, newKB.value.description)
  newKB.value = { name: '', description: '' }
  showCreateForm.value = false
  await loadKBs()
  emit('knowledge-bases-updated')
}

async function removeKB(id) {
  await deleteKnowledgeBase(id)
  selectedKB.value = null
  docs.value = []
  await loadKBs()
  emit('knowledge-bases-updated')
}

async function removeDoc(docId) {
  if (!selectedKB.value) return
  await deleteDocument(selectedKB.value.id, docId)
  docs.value = docs.value.filter(doc => doc.id !== docId)
}

function triggerUpload() {
  fileInput.value?.click()
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file || !selectedKB.value) return
  await uploadDocument(selectedKB.value.id, file)
  await selectKB(selectedKB.value.id)
  emit('knowledge-bases-updated')
  event.target.value = ''
}

async function rebuildIndex(kbId) {
  await createIndex(kbId)
}

function formatSize(bytes) {
  if (bytes > 1024 * 1024) return `${(bytes / 1048576).toFixed(1)} MB`
  if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

function fileType(filename = '') {
  return filename.split('.').pop()?.toUpperCase() || 'FILE'
}
</script>

<style scoped>
.kb-workbench {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-secondary);
}

.kb-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  padding: 18px 22px 12px;
}

.kb-metrics article {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--surface);
  padding: 14px;
}

.kb-metrics span {
  color: var(--text-muted);
  font-size: 12px;
}

.kb-metrics strong {
  display: block;
  margin-top: 4px;
  font-size: 24px;
}

.kb-metrics .health {
  color: var(--accent);
  font-size: 20px;
}

.kb-layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
  min-height: 0;
  padding: 0 22px 22px;
}

.kb-rail,
.kb-detail {
  min-height: 0;
  overflow: auto;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--surface);
}

.kb-rail {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
}

.create-button {
  width: 100%;
}

.create-form {
  display: grid;
  gap: 8px;
  border: 1px dashed var(--border-color);
  border-radius: 8px;
  padding: 10px;
}

.create-form div {
  display: flex;
  gap: 8px;
}

.create-form input,
.kb-search,
.index-config input {
  width: 100%;
  min-height: 36px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 0 10px;
}

.kb-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.kb-card {
  display: grid;
  gap: 5px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--surface-strong);
  padding: 12px;
  text-align: left;
}

.kb-card.active {
  border-color: rgba(22, 101, 52, 0.35);
  background: var(--accent-soft);
}

.kb-card-title {
  font-weight: 750;
}

.kb-card small,
.kb-card span:last-child,
.detail-head p {
  color: var(--text-muted);
}

.kb-detail {
  padding: 18px;
}

.kb-detail-inner {
  display: flex;
  min-height: 100%;
  flex-direction: column;
  gap: 16px;
}

.detail-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}

.detail-head h1 {
  font-size: 24px;
}

.detail-actions,
.row-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.tabs {
  display: inline-flex;
  align-self: flex-start;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-secondary);
  padding: 3px;
}

.tabs button {
  min-height: 34px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  padding: 0 12px;
  color: var(--text-secondary);
}

.tabs button.active {
  background: var(--surface);
  color: var(--text-primary);
  font-weight: 700;
  box-shadow: var(--shadow-sm);
}

.table-shell {
  overflow: auto;
}

.doc-table {
  width: 100%;
  min-width: 720px;
  border-collapse: collapse;
}

.doc-table th,
.doc-table td {
  border-bottom: 1px solid var(--border-color);
  padding: 12px;
  text-align: left;
}

.doc-table th {
  color: var(--text-muted);
  font-size: 12px;
}

.file-type {
  color: var(--info);
  font-size: 12px;
  font-weight: 800;
}

.chunk-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}

.chunk-grid article {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--surface-strong);
  padding: 12px;
}

.chunk-grid span,
.chunk-grid small {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 700;
}

.chunk-grid p {
  margin: 8px 0;
  color: var(--text-secondary);
}

.index-config {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.index-config label {
  display: grid;
  gap: 6px;
  color: var(--text-muted);
  font-size: 12px;
}

.upload-zone {
  display: grid;
  gap: 4px;
  margin-top: auto;
  border: 1px dashed #9aa4b2;
  border-radius: 8px;
  background: var(--surface-strong);
  padding: 22px;
  color: var(--text-primary);
  text-align: center;
}

.upload-zone span {
  color: var(--text-muted);
  font-size: 13px;
}

@media (max-width: 980px) {
  .kb-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .kb-layout {
    grid-template-columns: 1fr;
  }

  .kb-rail {
    max-height: 280px;
  }
}
</style>
