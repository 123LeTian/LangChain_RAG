<template>
  <!-- 知识库管理页面 — 创建知识库、上传文档、查看索引进度 -->
  <div class="kb-page">
    <h2>📚 知识库管理</h2>

    <!-- 操作栏 -->
    <div class="toolbar">
      <button @click="showCreateForm = true" class="btn-primary">+ 新建知识库</button>
      <input v-model="searchText" placeholder="搜索知识库..." class="search-input" />
    </div>

    <!-- 知识库列表 -->
    <div class="kb-grid">
      <div v-for="kb in filteredKBs" :key="kb.id" class="kb-card" @click="selectKB(kb.id)">
        <div class="kb-card-header">
          <h3>{{ kb.name }}</h3>
          <span :class="['status-badge', kb.status]">{{ kb.status }}</span>
        </div>
        <p class="kb-desc">{{ kb.description }}</p>
        <div class="kb-stats">
          <span>📄 {{ kb.doc_count }} 文档</span>
          <span>🧩 {{ kb.chunk_count }} Chunks</span>
        </div>
        <div class="kb-actions">
          <button class="btn-sm" @click.stop="uploadDoc(kb.id)">上传文档</button>
          <button class="btn-sm" @click.stop="rebuildIndex(kb.id)">重建索引</button>
          <button class="btn-sm btn-danger" @click.stop="removeKB(kb.id)">删除</button>
        </div>
      </div>

      <!-- 新建卡片 -->
      <div v-if="showCreateForm" class="kb-card create-card">
        <h3>新建知识库</h3>
        <input v-model="newKB.name" placeholder="名称" />
        <input v-model="newKB.description" placeholder="描述（可选）" />
        <div class="form-actions">
          <button @click="createKB" class="btn-primary">确认</button>
          <button @click="showCreateForm = false" class="btn-secondary">取消</button>
        </div>
      </div>
    </div>

    <!-- 文档列表（选中知识库后显示） -->
    <div v-if="selectedKB" class="doc-section">
      <h3>{{ selectedKB.name }} — 文档列表</h3>
      <table class="doc-table">
        <thead>
          <tr><th>文件名</th><th>类型</th><th>状态</th><th>Chunks</th><th>大小</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="doc in docs" :key="doc.id">
            <td>{{ doc.filename }}</td>
            <td><span class="doc-type">{{ doc.type }}</span></td>
            <td><span :class="['status-badge', doc.status]">{{ doc.status }}</span></td>
            <td>{{ doc.chunk_count }}</td>
            <td>{{ formatSize(doc.size_bytes) }}</td>
            <td>
              <button class="btn-sm btn-danger" @click="removeDoc(doc.id)">删除</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 上传文件输入（隐藏） -->
    <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md" @change="handleUpload" style="display:none" />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  listKnowledgeBases,
  createKnowledgeBase,
  deleteKnowledgeBase,
  listDocuments,
  deleteDocument,
  uploadDocument,
  createIndex,
} from '../api/client'

// ===== 状态 =====
const kbs = ref([])
const docs = ref([])
const selectedKB = ref(null)
const showCreateForm = ref(false)
const searchText = ref('')
const fileInput = ref(null)
const newKB = ref({ name: '', description: '' })

// ===== 计算属性 =====
const filteredKBs = computed(() =>
  searchText.value
    ? kbs.value.filter(kb => kb.name.includes(searchText.value))
    : kbs.value
)

// ===== 方法 =====
async function loadKBs() {
  kbs.value = await listKnowledgeBases()
}
async function selectKB(id) {
  const kb = kbs.value.find(k => k.id === id)
  selectedKB.value = kb || null
  docs.value = kb ? await listDocuments(id) : []
}
async function createKB() {
  await createKnowledgeBase(newKB.value.name, newKB.value.description)
  showCreateForm.value = false
  newKB.value = { name: '', description: '' }
  await loadKBs()
}
async function removeKB(id) {
  await deleteKnowledgeBase(id)
  if (selectedKB.value?.id === id) { selectedKB.value = null; docs.value = [] }
  await loadKBs()
}
async function removeDoc(docId) {
  if (!selectedKB.value) return
  await deleteDocument(selectedKB.value.id, docId)
  docs.value = docs.value.filter(d => d.id !== docId)
}
function uploadDoc(kbId) {
  selectKB(kbId)
  fileInput.value?.click()
}
async function handleUpload(e) {
  const file = e.target.files?.[0]
  if (!file || !selectedKB.value) return
  await uploadDocument(selectedKB.value.id, file)
  await selectKB(selectedKB.value.id) // 刷新
  e.target.value = ''
}
async function rebuildIndex(kbId) {
  const { job_id } = await createIndex(kbId)
  alert(`索引构建已启动：${job_id}`)
}
function formatSize(bytes) {
  return bytes > 1024 * 1024 ? `${(bytes / 1048576).toFixed(1)} MB`
    : bytes > 1024 ? `${(bytes / 1024).toFixed(1)} KB`
    : `${bytes} B`
}

// ===== 初始化 =====
loadKBs()
</script>

<style scoped>
.kb-page { padding: 24px; }
.toolbar { display: flex; gap: 12px; margin: 16px 0; }
.search-input { padding: 8px 12px; border: 1px solid var(--border-color); border-radius: 6px; width: 240px; }
.kb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
.kb-card {
  padding: 16px; border: 1px solid var(--border-color); border-radius: 10px;
  cursor: pointer; transition: box-shadow .2s;
}
.kb-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.kb-card-header { display: flex; justify-content: space-between; align-items: center; }
.kb-desc { color: var(--text-muted); font-size: 13px; margin: 8px 0; }
.kb-stats { display: flex; gap: 16px; font-size: 13px; color: var(--text-muted); margin-bottom: 12px; }
.kb-actions { display: flex; gap: 8px; }
.create-card { border-style: dashed; }
.create-card input { width: 100%; padding: 8px; margin-top: 8px; border: 1px solid var(--border-color); border-radius: 6px; }
.form-actions { display: flex; gap: 8px; margin-top: 12px; }

.status-badge {
  padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600;
}
.status-badge.ready { background: #d4edda; color: #155724; }
.status-badge.indexing { background: #fff3cd; color: #856404; }
.status-badge.uploaded { background: #cce5ff; color: #004085; }
.status-badge.error { background: #f8d7da; color: #721c24; }

.doc-section { margin-top: 32px; }
.doc-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.doc-table th, .doc-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
.doc-type { text-transform: uppercase; font-size: 12px; font-weight: 600; color: var(--accent); }

.btn-primary { padding: 8px 16px; background: var(--accent); color: white; border: none; border-radius: 6px; cursor: pointer; }
.btn-secondary { padding: 8px 16px; background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; }
.btn-sm { padding: 4px 12px; font-size: 12px; border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer; background: var(--bg-secondary); }
.btn-danger { color: var(--danger); border-color: var(--danger); }
</style>
