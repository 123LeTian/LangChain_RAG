<template>
  <section class="kb-workbench">
    <header class="metric-bar">
      <article class="metric-card">
        <span>总知识库</span>
        <strong>{{ kbs.length }}</strong>
      </article>
      <article class="metric-card">
        <span>总文档</span>
        <strong>{{ totalDocs }}</strong>
      </article>
      <article class="metric-card">
        <span>总 Chunk</span>
        <strong>{{ totalChunks }}</strong>
      </article>
      <article class="metric-card status-card">
        <span>向量库状态</span>
        <strong><i></i>正常</strong>
      </article>
    </header>

    <div class="workspace-grid">
      <aside class="kb-panel">
        <button class="primary-button create-button" type="button" @click="openCreateModal">
          <span>+</span>
          新建知识库
        </button>

        <label class="search-box">
          <span>⌕</span>
          <input v-model="searchText" placeholder="搜索知识库" />
        </label>

        <div class="kb-list">
          <button
            v-for="kb in filteredKBs"
            :key="kb.id"
            :class="['kb-list-card', { active: selectedKB?.id === kb.id }]"
            type="button"
            @click="selectKB(kb.id)"
          >
            <i></i>
            <span class="card-title">{{ kb.name }}</span>
            <span class="card-desc">{{ kb.description || '暂无描述' }}</span>
            <span class="card-meta">
              <b>{{ kb.doc_count || 0 }} 文档 · {{ kb.chunk_count || 0 }} 切片</b>
              <em>正常</em>
            </span>
          </button>
        </div>
      </aside>

      <main class="main-workspace">
        <div v-if="selectedKB" class="workspace-inner">
          <header class="workspace-head">
            <div>
              <div class="title-row">
                <h1>{{ selectedKB.name }}</h1>
                <span class="soft-tag">BGE / OpenAI Adapter</span>
                <span class="health-tag">正常</span>
              </div>
              <p>{{ selectedKB.description || '文档与索引管理' }}</p>
            </div>
            <div class="head-actions">
              <button class="primary-button" type="button" :disabled="isUploading" @click="triggerUpload">
                {{ isUploading ? '上传中...' : '+ 上传文档' }}
              </button>
              <button class="secondary-button" type="button" @click="rebuildIndex(selectedKB.id)">重建索引</button>
              <button class="danger-outline-button" type="button" @click="openDeleteModal">删除知识库</button>
            </div>
          </header>

          <nav class="tabs">
            <button :class="{ active: activeTab === 'docs' }" type="button" @click="activeTab = 'docs'">文档列表</button>
            <button :class="{ active: activeTab === 'chunks' }" type="button" @click="activeTab = 'chunks'">切片预览</button>
            <button :class="{ active: activeTab === 'config' }" type="button" @click="activeTab = 'config'">索引配置</button>
          </nav>

          <section v-if="activeTab === 'docs'" class="docs-view">
            <button
              v-if="!docs.length"
              :class="['dropzone large', { dragging: isDragging, uploading: isUploading }]"
              type="button"
              :disabled="isUploading"
              @click="triggerUpload"
              @dragenter.prevent="handleDragEnter"
              @dragover.prevent="handleDragOver"
              @dragleave.prevent="handleDragLeave"
              @drop.prevent="handleDrop"
            >
              <span class="drop-icons">PDF / Word / TXT</span>
              <strong>{{ uploadZoneTitle }}</strong>
              <small>{{ uploadZoneSubtitle }}</small>
            </button>

            <button
              v-else
              :class="['dropzone compact', { dragging: isDragging, uploading: isUploading }]"
              type="button"
              :disabled="isUploading"
              @click="triggerUpload"
              @dragenter.prevent="handleDragEnter"
              @dragover.prevent="handleDragOver"
              @dragleave.prevent="handleDragLeave"
              @drop.prevent="handleDrop"
            >
              <strong>{{ uploadZoneTitle }}</strong>
              <small>{{ uploadZoneSubtitle }}</small>
            </button>

            <p v-if="uploadError" class="feedback error">{{ uploadError }}</p>
            <p v-else-if="uploadSuccess" class="feedback success">{{ uploadSuccess }}</p>

            <div v-if="docs.length" class="table-shell">
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
                    <td>
                      <div class="filename-cell">
                        <span>{{ fileIcon(doc.filename) }}</span>
                        <b>{{ doc.filename }}</b>
                      </div>
                    </td>
                    <td><span class="file-type">{{ doc.type || fileType(doc.filename) }}</span></td>
                    <td>{{ formatSize(doc.size_bytes || 0) }}</td>
                    <td>{{ doc.chunk_count || 0 }}</td>
                    <td><span :class="['doc-status', statusClass(doc.status)]">{{ statusText(doc.status) }}</span></td>
                    <td>
                      <div class="hover-actions">
                        <button type="button" title="预览" @click="openPreview(doc)">
                          {{ isPreviewing && previewingDocId === doc.id ? '读取中' : '预览' }}
                        </button>
                        <button type="button" title="下载" @click="downloadPreview(doc)">下载</button>
                        <button type="button" title="删除" class="danger" @click="removeDoc(doc.id)">删除</button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section v-else-if="activeTab === 'chunks'" class="chunk-split">
            <aside class="chunk-outline">
              <span>文档目录</span>
              <button
                v-for="doc in docs"
                :key="doc.id"
                :class="{ active: selectedChunkDocId === doc.id }"
                type="button"
                @click="selectedChunkDocId = doc.id"
              >
                <b>{{ doc.filename }}</b>
                <small>{{ doc.chunk_count || 0 }} chunks</small>
              </button>
              <p v-if="!docs.length">上传文档后会在这里显示目录。</p>
            </aside>
            <div class="chunk-cards">
              <article v-for="chunk in previewChunks" :key="chunk.id">
                <header>
                  <span>{{ chunk.id }}</span>
                  <em>{{ chunk.tokens }} tokens</em>
                </header>
                <p>{{ chunk.text }}</p>
                <small>vector hash: {{ chunk.vectorId }}</small>
              </article>
            </div>
          </section>

          <section v-else class="config-view">
            <div class="config-grid">
              <label class="range-field">
                <span>Chunk Size</span>
                <div>
                  <input v-model.number="indexConfig.chunkSize" type="range" min="200" max="1600" step="50" @input="markConfigDirty" />
                  <input v-model.number="indexConfig.chunkSize" type="number" min="200" max="1600" step="50" @input="markConfigDirty" />
                </div>
                <small>* 推荐切片大小：500 - 1000 tokens</small>
              </label>

              <label class="range-field">
                <span>Overlap Window</span>
                <div>
                  <input v-model.number="indexConfig.overlap" type="range" min="0" max="400" step="20" @input="markConfigDirty" />
                  <input v-model.number="indexConfig.overlap" type="number" min="0" max="400" step="20" @input="markConfigDirty" />
                </div>
                <small>* 推荐重叠窗口：80 - 160 tokens</small>
              </label>

              <label class="readonly-field">
                <span>Embedding</span>
                <input value="BGE / OpenAI Adapter" readonly />
              </label>
            </div>
          </section>
        </div>

        <div v-else class="empty-state">创建或选择一个知识库。</div>
      </main>
    </div>

    <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md,.markdown" hidden multiple @change="handleUpload" />

    <div v-if="isCreateModalOpen" class="modal-backdrop" @click.self="closeCreateModal">
      <form class="modal-card" @submit.prevent="createKB">
        <header>
          <span>新建知识库</span>
          <button type="button" @click="closeCreateModal">×</button>
        </header>
        <label>
          <span>知识库名称</span>
          <input v-model="newKB.name" placeholder="例如：企业制度库" required />
        </label>
        <label>
          <span>描述</span>
          <textarea v-model="newKB.description" placeholder="用于说明该知识库的用途" rows="3"></textarea>
        </label>
        <footer>
          <button class="secondary-button" type="button" @click="closeCreateModal">取消</button>
          <button class="primary-button" type="submit">确定创建</button>
        </footer>
      </form>
    </div>

    <div v-if="isDeleteModalOpen" class="modal-backdrop" @click.self="closeDeleteModal">
      <section class="modal-card danger-modal">
        <header>
          <span>删除知识库</span>
          <button type="button" @click="closeDeleteModal">×</button>
        </header>
        <p>
          删除后，该知识库下的文档记录会被移除。请输入
          <b>{{ selectedKB?.name }}</b>
          以确认操作。
        </p>
        <label>
          <span>确认名称</span>
          <input v-model="deleteConfirmName" :placeholder="selectedKB?.name || ''" />
        </label>
        <footer>
          <button class="secondary-button" type="button" @click="closeDeleteModal">取消</button>
          <button class="danger-button" type="button" :disabled="deleteConfirmName !== selectedKB?.name" @click="confirmDeleteKB">
            确认删除
          </button>
        </footer>
      </section>
    </div>

    <div v-if="previewOpen" class="preview-backdrop" @click.self="closePreview">
      <section class="preview-dialog" role="dialog" aria-modal="true" aria-label="文档预览">
        <header class="preview-head">
          <div>
            <span>文档预览</span>
            <h2>{{ previewData?.filename || '正在读取文档' }}</h2>
            <p v-if="previewData">
              {{ previewData.type?.toUpperCase() || 'FILE' }} · {{ formatSize(previewData.size_bytes || 0) }}
              <template v-if="previewData.truncated"> · 已截取前 20000 字</template>
            </p>
          </div>
          <button class="preview-close" type="button" aria-label="关闭预览" @click="closePreview">×</button>
        </header>

        <div v-if="isPreviewing" class="preview-state">正在提取文档文本...</div>
        <div v-else-if="previewError" class="preview-state error">{{ previewError }}</div>
        <pre v-else class="preview-content">{{ previewData?.text || '这个文档没有可预览的文本内容。' }}</pre>
      </section>
    </div>

    <div v-if="isConfigDirty" class="floating-actions">
      <span>索引配置有未保存修改</span>
      <button class="secondary-button" type="button" @click="resetConfig">重置</button>
      <button class="primary-button" type="button" @click="saveConfig">保存修改</button>
    </div>
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
  previewDocument,
  uploadDocument,
} from '../api/client'

const props = defineProps({
  knowledgeBases: {
    type: Array,
    default: () => [],
  },
})

const emit = defineEmits(['knowledge-bases-updated'])

const defaultIndexConfig = { chunkSize: 800, overlap: 120 }

const kbs = ref([])
const docs = ref([])
const selectedKB = ref(null)
const searchText = ref('')
const activeTab = ref('docs')
const fileInput = ref(null)
const newKB = ref({ name: '', description: '' })
const isCreateModalOpen = ref(false)
const isDeleteModalOpen = ref(false)
const deleteConfirmName = ref('')
const isConfigDirty = ref(false)
const indexConfig = ref({ ...defaultIndexConfig })
const isDragging = ref(false)
const isUploading = ref(false)
const uploadError = ref('')
const uploadSuccess = ref('')
const uploadingName = ref('')
const dragDepth = ref(0)
const selectedChunkDocId = ref('')
const previewOpen = ref(false)
const previewData = ref(null)
const previewError = ref('')
const isPreviewing = ref(false)
const previewingDocId = ref('')

const filteredKBs = computed(() => {
  const keyword = searchText.value.trim().toLowerCase()
  if (!keyword) return kbs.value
  return kbs.value.filter(kb => {
    const name = String(kb.name || '').toLowerCase()
    const desc = String(kb.description || '').toLowerCase()
    return name.includes(keyword) || desc.includes(keyword)
  })
})

const totalDocs = computed(() => kbs.value.reduce((sum, kb) => sum + Number(kb.doc_count || 0), 0))
const totalChunks = computed(() => kbs.value.reduce((sum, kb) => sum + Number(kb.chunk_count || 0), 0))
const uploadZoneTitle = computed(() => {
  if (isUploading.value) return `正在上传 ${uploadingName.value || '文档'}...`
  if (isDragging.value) return '松开鼠标开始上传'
  return docs.value.length ? '拖拽文件到这里，或点击继续上传' : '拖拽或点击上传 PDF / Word / TXT'
})
const uploadZoneSubtitle = computed(() => {
  if (isUploading.value) return '请稍候，上传完成后会自动刷新文档列表。'
  if (uploadSuccess.value) return uploadSuccess.value
  return '支持 PDF、DOCX、TXT、Markdown；可一次选择多个文件。'
})
const previewChunks = computed(() => {
  const sourceDocs = selectedChunkDocId.value
    ? docs.value.filter(doc => doc.id === selectedChunkDocId.value)
    : docs.value.slice(0, 4)
  if (!sourceDocs.length) {
    return [
      { id: 'Chunk #1', text: '上传文档后，这里会展示切片文本、Token 数与向量摘要。', vectorId: 'vec_empty_preview', tokens: 0 },
    ]
  }
  return sourceDocs.flatMap((doc, docIndex) => {
    const count = Math.max(1, Math.min(Number(doc.chunk_count || 1), 3))
    return Array.from({ length: count }, (_, index) => ({
      id: `Chunk #${docIndex + 1}.${index + 1}`,
      text: `${doc.filename} 的切片预览，包含解析后的局部文本与检索上下文。`,
      vectorId: `vec_${String(doc.id || index).slice(0, 10)}_${index + 1}`,
      tokens: Math.max(120, Math.min(980, Math.round((doc.size_bytes || 800) / 6))),
    }))
  })
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
    if (selectedKB.value) {
      selectedKB.value = kbs.value.find(kb => kb.id === selectedKB.value.id) || selectedKB.value
    }
  } catch {
    kbs.value = props.knowledgeBases.length ? [...props.knowledgeBases] : []
  }
}

async function selectKB(id) {
  selectedKB.value = kbs.value.find(kb => kb.id === id) || null
  activeTab.value = 'docs'
  selectedChunkDocId.value = ''
  uploadError.value = ''
  uploadSuccess.value = ''
  if (!selectedKB.value) {
    docs.value = []
    return
  }
  try {
    docs.value = await listDocuments(id)
    selectedChunkDocId.value = docs.value[0]?.id || ''
  } catch {
    docs.value = []
  }
}

function openCreateModal() {
  newKB.value = { name: '', description: '' }
  isCreateModalOpen.value = true
}

function closeCreateModal() {
  isCreateModalOpen.value = false
}

async function createKB() {
  const created = await createKnowledgeBase(newKB.value.name, newKB.value.description)
  closeCreateModal()
  await loadKBs()
  if (created?.id) await selectKB(created.id)
  emit('knowledge-bases-updated')
}

function openDeleteModal() {
  deleteConfirmName.value = ''
  isDeleteModalOpen.value = true
}

function closeDeleteModal() {
  isDeleteModalOpen.value = false
  deleteConfirmName.value = ''
}

async function confirmDeleteKB() {
  if (!selectedKB.value || deleteConfirmName.value !== selectedKB.value.name) return
  await deleteKnowledgeBase(selectedKB.value.id)
  closeDeleteModal()
  selectedKB.value = null
  docs.value = []
  await loadKBs()
  emit('knowledge-bases-updated')
}

async function removeDoc(docId) {
  if (!selectedKB.value) return
  await deleteDocument(selectedKB.value.id, docId)
  docs.value = docs.value.filter(doc => doc.id !== docId)
  if (selectedChunkDocId.value === docId) selectedChunkDocId.value = docs.value[0]?.id || ''
  if (previewData.value?.id === docId) closePreview()
  await loadKBs()
  emit('knowledge-bases-updated')
}

function triggerUpload() {
  if (isUploading.value || !selectedKB.value) return
  fileInput.value?.click()
}

async function handleUpload(event) {
  await uploadFiles(event.target.files)
  event.target.value = ''
}

function handleDragEnter() {
  if (isUploading.value) return
  dragDepth.value += 1
  isDragging.value = true
}

function handleDragOver() {
  if (isUploading.value) return
  isDragging.value = true
}

function handleDragLeave() {
  dragDepth.value = Math.max(0, dragDepth.value - 1)
  if (dragDepth.value === 0) isDragging.value = false
}

async function handleDrop(event) {
  dragDepth.value = 0
  isDragging.value = false
  await uploadFiles(event.dataTransfer?.files)
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || [])
  if (!files.length || !selectedKB.value || isUploading.value) return

  uploadError.value = ''
  uploadSuccess.value = ''

  const allowed = new Set(['pdf', 'docx', 'txt', 'md', 'markdown'])
  const invalid = files.find(file => !allowed.has(fileType(file.name).toLowerCase()))
  if (invalid) {
    uploadError.value = `暂不支持 ${invalid.name}，请上传 PDF、DOCX、TXT 或 Markdown。`
    return
  }

  isUploading.value = true
  let uploaded = 0
  const kbId = selectedKB.value.id

  try {
    for (const file of files) {
      uploadingName.value = file.name
      await uploadDocument(kbId, file)
      uploaded += 1
    }
    await selectKB(kbId)
    await loadKBs()
    emit('knowledge-bases-updated')
    uploadSuccess.value = `已上传 ${uploaded} 个文档。`
    activeTab.value = 'docs'
  } catch (error) {
    uploadError.value = error?.message || '上传失败，请稍后重试。'
  } finally {
    isUploading.value = false
    uploadingName.value = ''
  }
}

async function rebuildIndex(kbId) {
  await createIndex(kbId)
}

async function openPreview(doc) {
  if (!selectedKB.value || isPreviewing.value) return
  previewOpen.value = true
  previewData.value = null
  previewError.value = ''
  isPreviewing.value = true
  previewingDocId.value = doc.id

  try {
    previewData.value = await previewDocument(selectedKB.value.id, doc.id)
  } catch (error) {
    previewError.value = error?.message || '预览失败，请稍后重试。'
  } finally {
    isPreviewing.value = false
    previewingDocId.value = ''
  }
}

async function downloadPreview(doc) {
  if (!selectedKB.value) return
  try {
    const data = await previewDocument(selectedKB.value.id, doc.id)
    const blob = new Blob([data.text || ''], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${doc.filename || 'document'}.preview.txt`
    link.click()
    URL.revokeObjectURL(url)
  } catch (error) {
    uploadError.value = error?.message || '下载失败，请稍后重试。'
  }
}

function closePreview() {
  previewOpen.value = false
  previewData.value = null
  previewError.value = ''
}

function markConfigDirty() {
  isConfigDirty.value = true
}

function resetConfig() {
  indexConfig.value = { ...defaultIndexConfig }
  isConfigDirty.value = false
}

function saveConfig() {
  isConfigDirty.value = false
}

function formatSize(bytes) {
  if (bytes > 1024 * 1024) return `${(bytes / 1048576).toFixed(1)} MB`
  if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

function fileType(filename = '') {
  return filename.split('.').pop()?.toUpperCase() || 'FILE'
}

function fileIcon(filename = '') {
  const type = fileType(filename).toLowerCase()
  if (type === 'pdf') return 'PDF'
  if (type === 'docx') return 'DOC'
  if (type === 'md' || type === 'markdown') return 'MD'
  return 'TXT'
}

function statusClass(status = 'indexed') {
  if (['parsing', 'chunking', 'indexing', 'uploaded'].includes(status)) return 'processing'
  if (['error', 'failed'].includes(status)) return 'failed'
  return 'indexed'
}

function statusText(status = 'indexed') {
  if (['parsing', 'chunking', 'indexing', 'uploaded'].includes(status)) return '解析中'
  if (['error', 'failed'].includes(status)) return '失败'
  return '已索引'
}
</script>

<style scoped>
.kb-workbench {
  display: grid;
  grid-template-rows: auto 1fr;
  height: 100vh;
  overflow: hidden;
  background: #f8fafc;
  color: #0f172a;
}

.metric-bar {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  padding: 16px 22px 12px;
}

.metric-card {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  padding: 10px 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.metric-card span {
  color: #64748b;
  font-size: 12px;
  font-weight: 650;
}

.metric-card strong {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  font-size: 22px;
}

.status-card strong {
  color: #047857;
  font-size: 18px;
}

.status-card i {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #10b981;
  animation: pulse 1.6s ease-in-out infinite;
}

.workspace-grid {
  display: grid;
  grid-template-columns: minmax(260px, 300px) 1fr;
  gap: 16px;
  min-height: 0;
  padding: 0 22px 22px;
}

.kb-panel,
.main-workspace {
  min-height: 0;
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.kb-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
}

.primary-button,
.secondary-button,
.danger-button,
.danger-outline-button {
  min-height: 38px;
  border: 0;
  border-radius: 8px;
  padding: 0 14px;
  font-size: 13px;
  font-weight: 800;
  cursor: pointer;
  transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
}

.primary-button {
  background: #2563eb;
  color: #fff;
}

.primary-button:hover {
  background: #1d4ed8;
}

.secondary-button {
  border: 1px solid #e2e8f0;
  background: #fff;
  color: #334155;
}

.secondary-button:hover {
  border-color: #cbd5e1;
  background: #f8fafc;
}

.danger-button {
  background: #ef4444;
  color: #fff;
}

.danger-outline-button {
  border: 1px solid #fecdd3;
  background: #fff7ed;
  color: #c2410c;
}

.danger-outline-button:hover {
  border-color: #fdba74;
  background: #ffedd5;
}

.danger-button:disabled,
.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.create-button {
  width: 100%;
}

.create-button span {
  margin-right: 6px;
}

.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
  padding: 0 10px;
}

.search-box span {
  color: #94a3b8;
}

.search-box input {
  width: 100%;
  min-height: 38px;
  border: 0;
  outline: 0;
  background: transparent;
  color: #0f172a;
}

.kb-list {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
}

.kb-list-card {
  position: relative;
  display: grid;
  gap: 6px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  padding: 12px 12px 12px 14px;
  text-align: left;
  cursor: pointer;
}

.kb-list-card i {
  position: absolute;
  left: 0;
  top: 12px;
  bottom: 12px;
  width: 3px;
  border-radius: 999px;
  background: transparent;
}

.kb-list-card:hover {
  border-color: #cbd5e1;
  background: #f8fafc;
}

.kb-list-card.active {
  border-color: #bfdbfe;
  background: rgba(239, 246, 255, 0.72);
}

.kb-list-card.active i {
  background: #2563eb;
}

.card-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 15px;
  font-weight: 850;
}

.card-desc {
  overflow: hidden;
  color: #64748b;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}

.card-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.card-meta b {
  color: #64748b;
  font-size: 12px;
}

.card-meta em,
.health-tag {
  border-radius: 999px;
  background: #d1fae5;
  color: #047857;
  font-size: 11px;
  font-style: normal;
  font-weight: 800;
  padding: 3px 8px;
}

.main-workspace {
  padding: 18px 22px;
}

.workspace-inner {
  display: flex;
  min-height: 100%;
  flex-direction: column;
  gap: 16px;
}

.workspace-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
}

.title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.title-row h1 {
  font-size: 25px;
  line-height: 1.2;
}

.workspace-head p {
  margin-top: 6px;
  color: #64748b;
  font-size: 13px;
}

.soft-tag {
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  background: #f8fafc;
  color: #64748b;
  font-size: 11px;
  font-weight: 800;
  padding: 3px 8px;
}

.head-actions {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.tabs {
  display: inline-flex;
  align-self: flex-start;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #f8fafc;
  padding: 4px;
}

.tabs button {
  min-height: 34px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #64748b;
  padding: 0 14px;
  cursor: pointer;
  font-weight: 800;
}

.tabs button.active {
  background: #fff;
  color: #0f172a;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.docs-view {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 12px;
}

.dropzone {
  display: grid;
  place-items: center;
  gap: 7px;
  border: 1px dashed #94a3b8;
  border-radius: 12px;
  background: #f8fafc;
  color: #0f172a;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.dropzone.large {
  min-height: 260px;
  padding: 30px;
}

.dropzone.compact {
  grid-auto-flow: column;
  justify-content: center;
  min-height: 56px;
  padding: 12px 18px;
}

.dropzone:hover,
.dropzone.dragging {
  border-color: #2563eb;
  background: #eff6ff;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.08);
}

.drop-icons {
  border-radius: 999px;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 900;
  padding: 7px 12px;
}

.dropzone strong {
  font-size: 15px;
}

.dropzone small {
  color: #64748b;
}

.feedback {
  border-radius: 10px;
  padding: 9px 12px;
  font-size: 12px;
  font-weight: 750;
}

.feedback.success {
  border: 1px solid rgba(16, 185, 129, 0.24);
  background: #ecfdf5;
  color: #047857;
}

.feedback.error {
  border: 1px solid rgba(239, 68, 68, 0.22);
  background: #fff1f2;
  color: #be123c;
}

.table-shell {
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}

.doc-table {
  width: 100%;
  min-width: 780px;
  border-collapse: collapse;
}

.doc-table th,
.doc-table td {
  border-bottom: 1px solid #e2e8f0;
  padding: 12px;
  text-align: left;
}

.doc-table th {
  background: #f8fafc;
  color: #64748b;
  font-size: 12px;
}

.doc-table tr:last-child td {
  border-bottom: 0;
}

.filename-cell {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
}

.filename-cell span,
.file-type {
  border-radius: 7px;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 11px;
  font-weight: 900;
  padding: 4px 6px;
}

.filename-cell b {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-status {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 850;
  padding: 4px 8px;
}

.doc-status.indexed {
  background: #d1fae5;
  color: #047857;
}

.doc-status.processing {
  background: #dbeafe;
  color: #1d4ed8;
}

.doc-status.failed {
  background: #ffe4e6;
  color: #be123c;
}

.hover-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.hover-actions button {
  min-height: 28px;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  background: #fff;
  color: #475569;
  padding: 0 8px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 750;
}

.hover-actions button:hover {
  border-color: #cbd5e1;
  background: #f8fafc;
}

.hover-actions .danger {
  border-color: #fecdd3;
  color: #e11d48;
}

.hover-actions .danger:hover {
  background: #ffe4e6;
  color: #be123c;
}

.chunk-split {
  display: grid;
  grid-template-columns: minmax(220px, 30%) 1fr;
  gap: 14px;
  min-height: 0;
  flex: 1;
}

.chunk-outline,
.chunk-cards {
  min-height: 0;
  overflow: auto;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
}

.chunk-outline {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
}

.chunk-outline > span {
  color: #64748b;
  font-size: 12px;
  font-weight: 850;
}

.chunk-outline button {
  display: grid;
  gap: 4px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #fff;
  padding: 10px;
  text-align: left;
  cursor: pointer;
}

.chunk-outline button.active {
  border-color: #bfdbfe;
  background: #eff6ff;
}

.chunk-outline small,
.chunk-outline p {
  color: #64748b;
}

.chunk-cards {
  display: grid;
  align-content: start;
  gap: 12px;
  padding: 14px;
}

.chunk-cards article {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #f8fafc;
  padding: 14px;
}

.chunk-cards header {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: #64748b;
  font-size: 12px;
  font-weight: 850;
}

.chunk-cards p {
  margin: 10px 0;
  border-left: 3px solid #2563eb;
  padding-left: 10px;
  color: #334155;
  line-height: 1.7;
}

.chunk-cards small {
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.config-view {
  flex: 1;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.range-field,
.readonly-field {
  display: grid;
  gap: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #fff;
  padding: 14px;
}

.range-field span,
.readonly-field span {
  color: #64748b;
  font-size: 12px;
  font-weight: 850;
}

.range-field div {
  display: grid;
  grid-template-columns: 1fr 92px;
  gap: 10px;
  align-items: center;
}

.range-field input[type='number'],
.readonly-field input {
  width: 100%;
  min-height: 38px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0 10px;
}

.range-field small {
  color: #94a3b8;
}

.empty-state {
  display: grid;
  min-height: 320px;
  place-items: center;
  color: #64748b;
}

.modal-backdrop,
.preview-backdrop {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.42);
  padding: 24px;
  backdrop-filter: blur(6px);
}

.modal-card {
  display: grid;
  width: min(440px, 100%);
  gap: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  background: #fff;
  padding: 20px;
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.22);
}

.modal-card header,
.modal-card footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.modal-card header span {
  font-size: 18px;
  font-weight: 900;
}

.modal-card header button {
  width: 30px;
  height: 30px;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  background: #fff;
  cursor: pointer;
}

.modal-card label {
  display: grid;
  gap: 7px;
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

.modal-card input,
.modal-card textarea {
  width: 100%;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 10px 12px;
  color: #0f172a;
  font: inherit;
}

.danger-modal p {
  color: #475569;
  line-height: 1.7;
}

.danger-modal b {
  color: #be123c;
}

.preview-dialog {
  display: flex;
  width: min(920px, 100%);
  max-height: min(760px, calc(100vh - 48px));
  flex-direction: column;
  overflow: hidden;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.22);
}

.preview-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid #e2e8f0;
  padding: 18px 20px;
}

.preview-head span {
  color: #64748b;
  font-size: 12px;
  font-weight: 850;
}

.preview-head h2 {
  margin-top: 4px;
  font-size: 18px;
  word-break: break-word;
}

.preview-head p {
  margin-top: 6px;
  color: #64748b;
  font-size: 12px;
}

.preview-close {
  display: grid;
  width: 32px;
  height: 32px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  background: #f8fafc;
  color: #475569;
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
}

.preview-content {
  flex: 1;
  min-height: 260px;
  overflow: auto;
  margin: 0;
  padding: 20px;
  background: #f8fafc;
  color: #0f172a;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
}

.preview-state {
  display: grid;
  min-height: 260px;
  place-items: center;
  padding: 24px;
  color: #64748b;
  font-size: 14px;
}

.preview-state.error {
  color: #be123c;
}

.floating-actions {
  position: fixed;
  right: 26px;
  bottom: 24px;
  z-index: 50;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  background: #fff;
  padding: 10px;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
}

.floating-actions span {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.42;
    transform: scale(1.4);
  }
}

@media (max-width: 1100px) {
  .metric-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .workspace-grid,
  .chunk-split,
  .config-grid {
    grid-template-columns: 1fr;
  }

  .kb-panel {
    max-height: 330px;
  }

  .workspace-head {
    flex-direction: column;
  }
}
</style>
