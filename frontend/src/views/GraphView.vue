<template>
  <section class="graph-workspace">
    <div class="graph-controls">
      <select v-model="selectedKbId">
        <option value="">默认知识库</option>
        <option v-for="kb in knowledgeBases" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
      </select>
      <div class="entity-filter">
        <label><input v-model="filters.project" type="checkbox" /> 项目</label>
        <label><input v-model="filters.person" type="checkbox" /> 人物</label>
        <label><input v-model="filters.metric" type="checkbox" /> 数值</label>
      </div>
      <input v-model="searchText" placeholder="节点搜索..." />
      <button class="btn-primary" type="button" @click="loadGraph">刷新图谱</button>
    </div>

    <div class="graph-canvas">
      <svg viewBox="0 0 960 620" role="img" aria-label="知识图谱">
        <defs>
          <filter id="nodeShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.12" />
          </filter>
        </defs>

        <g class="links">
          <g v-for="link in visibleLinks" :key="`${link.source}-${link.target}`">
            <line
              :x1="positionOf(link.source).x"
              :y1="positionOf(link.source).y"
              :x2="positionOf(link.target).x"
              :y2="positionOf(link.target).y"
              @click="selectLink(link)"
            />
            <text
              :x="(positionOf(link.source).x + positionOf(link.target).x) / 2"
              :y="(positionOf(link.source).y + positionOf(link.target).y) / 2 - 8"
              text-anchor="middle"
            >
              {{ link.label || link.relation || '关联' }}
            </text>
          </g>
        </g>

        <g class="nodes">
          <g
            v-for="node in visibleNodes"
            :key="node.id"
            :class="['node', nodeType(node), { active: selectedItem?.id === node.id }]"
            :transform="`translate(${positionOf(node.id).x}, ${positionOf(node.id).y})`"
            @click="selectNode(node)"
          >
            <circle :r="node.type === 'community' ? 34 : 28" />
            <text y="5" text-anchor="middle">{{ shortName(node.name) }}</text>
          </g>
        </g>
      </svg>
    </div>

    <aside class="inspector" :class="{ open: inspectorOpen }">
      <div class="inspector-head">
        <div>
          <strong>{{ selectedItem?.name || selectedItem?.label || '实体详情' }}</strong>
          <span>{{ selectedItem?.type || selectedItem?.relation || '关系' }}</span>
        </div>
        <button class="icon-button" type="button" title="关闭" @click="inspectorOpen = false">×</button>
      </div>

      <div class="inspector-body">
        <dl>
          <div>
            <dt>类别</dt>
            <dd>{{ selectedItem?.type || selectedItem?.relation || '未知' }}</dd>
          </div>
          <div>
            <dt>关联边</dt>
            <dd>{{ relatedLinks.length }}</dd>
          </div>
          <div>
            <dt>追溯切片</dt>
            <dd>{{ selectedItem?.chunk_id || 'Chunk #2' }}</dd>
          </div>
        </dl>

        <h3>关系链</h3>
        <ul>
          <li v-for="link in relatedLinks" :key="`${link.source}-${link.target}`">
            {{ nodeName(link.source) }} → {{ link.label || link.relation || '关联' }} → {{ nodeName(link.target) }}
          </li>
        </ul>

        <h3>原始文本出处</h3>
        <p>{{ selectedItem?.source_text || '星河计划的项目负责人是李明，项目预算为 128 万元。' }}</p>
      </div>
    </aside>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { getGraphData } from '../api/client'

const props = defineProps({
  knowledgeBases: {
    type: Array,
    default: () => [],
  },
})

const selectedKbId = ref('')
const searchText = ref('')
const graphData = ref(null)
const selectedItem = ref(null)
const inspectorOpen = ref(false)
const filters = reactive({ project: true, person: true, metric: true })

const sampleGraph = {
  nodes: [
    { id: 'project', name: '星河计划', type: 'project', chunk_id: 'Chunk #1' },
    { id: 'owner', name: '李明', type: 'person', chunk_id: 'Chunk #1' },
    { id: 'budget', name: '128万元', type: 'metric', chunk_id: 'Chunk #2' },
    { id: 'milestone', name: '上线验收', type: 'project', chunk_id: 'Chunk #3' },
    { id: 'tech', name: 'GraphRAG', type: 'community', chunk_id: 'Chunk #4' },
  ],
  links: [
    { source: 'project', target: 'owner', label: '负责人' },
    { source: 'project', target: 'budget', label: '预算' },
    { source: 'project', target: 'milestone', label: '里程碑' },
    { source: 'tech', target: 'project', label: '应用于' },
  ],
}

const nodePositions = {
  project: { x: 430, y: 210 },
  owner: { x: 600, y: 110 },
  budget: { x: 620, y: 340 },
  milestone: { x: 300, y: 390 },
  tech: { x: 250, y: 150 },
}

const currentGraph = computed(() => graphData.value || sampleGraph)
const visibleNodes = computed(() => currentGraph.value.nodes.filter((node) => {
  const type = nodeType(node)
  const matchesType = filters[type] !== false
  const matchesSearch = !searchText.value || node.name?.includes(searchText.value)
  return matchesType && matchesSearch
}))
const visibleNodeIds = computed(() => new Set(visibleNodes.value.map(node => node.id)))
const visibleLinks = computed(() => currentGraph.value.links.filter(link =>
  visibleNodeIds.value.has(link.source) && visibleNodeIds.value.has(link.target)
))
const relatedLinks = computed(() => {
  if (!selectedItem.value) return []
  if (selectedItem.value.source && selectedItem.value.target) return [selectedItem.value]
  return currentGraph.value.links.filter(link =>
    link.source === selectedItem.value.id || link.target === selectedItem.value.id
  )
})

onMounted(() => {
  if (props.knowledgeBases.length) selectedKbId.value = props.knowledgeBases[0].id
})

async function loadGraph() {
  if (!selectedKbId.value) {
    graphData.value = sampleGraph
    return
  }
  try {
    graphData.value = await getGraphData(selectedKbId.value)
  } catch {
    graphData.value = sampleGraph
  }
}

function selectNode(node) {
  selectedItem.value = node
  inspectorOpen.value = true
}

function selectLink(link) {
  selectedItem.value = { ...link, name: link.label || link.relation || '关系' }
  inspectorOpen.value = true
}

function positionOf(id) {
  if (nodePositions[id]) return nodePositions[id]
  const index = currentGraph.value.nodes.findIndex(node => node.id === id)
  const angle = (index / Math.max(currentGraph.value.nodes.length, 1)) * Math.PI * 2
  return { x: 480 + Math.cos(angle) * 240, y: 300 + Math.sin(angle) * 180 }
}

function nodeType(node) {
  if (['person', 'project', 'metric'].includes(node.type)) return node.type
  return node.type === 'community' ? 'project' : 'metric'
}

function shortName(name = '') {
  return name.length > 5 ? `${name.slice(0, 5)}` : name
}

function nodeName(id) {
  return currentGraph.value.nodes.find(node => node.id === id)?.name || id
}
</script>

<style scoped>
.graph-workspace {
  position: relative;
  height: 100vh;
  overflow: hidden;
  background:
    linear-gradient(rgba(20, 27, 33, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20, 27, 33, 0.04) 1px, transparent 1px),
    #fbfcfd;
  background-size: 36px 36px;
}

.graph-controls {
  position: absolute;
  top: 18px;
  left: 50%;
  z-index: 2;
  display: flex;
  max-width: calc(100% - 40px);
  transform: translateX(-50%);
  align-items: center;
  gap: 8px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.9);
  padding: 8px;
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(12px);
}

.graph-controls select,
.graph-controls input {
  min-height: 34px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: var(--surface);
  padding: 0 12px;
}

.entity-filter {
  display: flex;
  gap: 8px;
  padding: 0 6px;
  color: var(--text-secondary);
  font-size: 13px;
}

.entity-filter label {
  display: flex;
  align-items: center;
  gap: 4px;
}

.graph-canvas {
  width: 100%;
  height: 100%;
}

.graph-canvas svg {
  width: 100%;
  height: 100%;
}

.links line {
  cursor: pointer;
  stroke: #9aa4b2;
  stroke-width: 2;
}

.links text {
  fill: var(--text-secondary);
  font-size: 13px;
  paint-order: stroke;
  stroke: white;
  stroke-width: 5px;
}

.node {
  cursor: pointer;
  filter: url(#nodeShadow);
}

.node circle {
  stroke: white;
  stroke-width: 3;
}

.node.project circle {
  fill: #2563eb;
}

.node.person circle {
  fill: #9333ea;
}

.node.metric circle {
  fill: #16a34a;
}

.node.active circle {
  stroke: #111827;
  stroke-width: 4;
}

.node text {
  pointer-events: none;
  fill: white;
  font-size: 13px;
  font-weight: 800;
}

.inspector {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: min(360px, 100%);
  transform: translateX(100%);
  border-left: 1px solid var(--border-color);
  background: var(--surface);
  box-shadow: var(--shadow-md);
  transition: transform 0.22s ease;
}

.inspector.open {
  transform: translateX(0);
}

.inspector-head {
  display: flex;
  min-height: 68px;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-color);
  padding: 0 16px;
}

.inspector-head div {
  display: flex;
  flex-direction: column;
}

.inspector-head span {
  color: var(--text-muted);
  font-size: 12px;
}

.inspector-body {
  display: grid;
  gap: 18px;
  padding: 16px;
}

.inspector-body dl {
  display: grid;
  gap: 10px;
}

.inspector-body dl div {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 8px;
}

.inspector-body dt,
.inspector-body h3 {
  color: var(--text-muted);
  font-size: 12px;
}

.inspector-body dd {
  font-weight: 700;
}

.inspector-body ul {
  display: grid;
  gap: 8px;
  padding-left: 18px;
  color: var(--text-secondary);
}

.inspector-body p {
  border-radius: 8px;
  background: var(--surface-strong);
  padding: 12px;
  color: var(--text-secondary);
}

@media (max-width: 920px) {
  .graph-controls {
    left: 16px;
    right: 16px;
    transform: none;
    flex-wrap: wrap;
    border-radius: 14px;
  }
}
</style>
