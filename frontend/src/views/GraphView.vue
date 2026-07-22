<template>
  <section class="graph-workspace">
    <div class="graph-controls">
      <div class="graph-picker">
        <button type="button" @click="toggleMenu('kb')">
          <span>知识库</span>
          <strong>{{ selectedKbLabel }}</strong>
        </button>
        <div v-if="openMenu === 'kb'" class="graph-popover">
          <button :class="{ selected: selectedKbId === '' }" type="button" @click="chooseKb('')">
            请选择知识库
          </button>
          <button
            v-for="kb in availableKnowledgeBases"
            :key="kb.id"
            :class="{ selected: selectedKbId === kb.id }"
            type="button"
            @click="chooseKb(kb.id)"
          >
            {{ kb.name }}
          </button>
        </div>
      </div>

      <div class="graph-picker model-picker">
        <button type="button" @click="toggleMenu('model')">
          <span>图谱模型</span>
          <strong>{{ selectedModelLabel }}</strong>
        </button>
        <div v-if="openMenu === 'model'" class="graph-popover">
          <button
            v-for="model in modelOptions"
            :key="model.id"
            :class="{ selected: selectedModelId === model.id }"
            type="button"
            @click="chooseModel(model.id)"
          >
            <span>{{ model.display_name }}</span>
            <small>{{ model.provider }} / {{ model.model_name }}</small>
          </button>
        </div>
      </div>

      <div class="toolbar-divider" aria-hidden="true"></div>

      <div class="entity-filter">
        <button
          v-for="item in filterOptions"
          :key="item.key"
          :class="['filter-pill', item.key, { active: filters[item.key] }]"
          :style="{ '--dot-color': item.color }"
          type="button"
          @click="filters[item.key] = !filters[item.key]"
        >
          <span></span>
          {{ item.label }}
        </button>
      </div>

      <div class="toolbar-divider" aria-hidden="true"></div>

      <input v-model="searchText" placeholder="节点搜索..." />
      <button
        class="btn-primary"
        :disabled="isGenerating || !selectedKbId"
        type="button"
        @click="generateGraph"
      >
        <span>{{ isGenerating ? '' : '⚡' }}</span>
        {{ isGenerating ? '生成中...' : '生成图谱' }}
      </button>
    </div>

    <div class="graph-body">
      <div class="graph-canvas">
      <div class="model-hint">
        <span>当前图谱模型</span>
        <strong>{{ selectedModelLabel }}</strong>
        <em>{{ graphBuildHint }}</em>
      </div>

      <svg
        ref="svgRef"
        :class="{ dragging: isDraggingCanvas }"
        viewBox="0 0 960 620"
        role="img"
        aria-label="知识图谱"
        @mousedown="startCanvasDrag"
        @mousemove="dragCanvas"
        @mouseup="endCanvasDrag"
        @mouseleave="endCanvasDrag"
        @wheel.prevent="handleCanvasWheel"
      >
        <defs>
          <filter id="nodeShadow" x="-24%" y="-24%" width="148%" height="148%">
            <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.12" />
          </filter>
          <marker
            id="edgeArrow"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
          </marker>
        </defs>

        <g :transform="canvasTransform">
          <g class="links">
            <g
              v-for="link in visibleLinks"
              :key="linkKey(link)"
              :class="['edge', highlightClassForLink(link)]"
            >
              <line
                :x1="edgePoints(link).x1"
                :y1="edgePoints(link).y1"
                :x2="edgePoints(link).x2"
                :y2="edgePoints(link).y2"
                marker-end="url(#edgeArrow)"
                @click="selectLink(link)"
              />
              <text
                v-if="showLinkLabel"
                :x="(positionOf(link.source).x + positionOf(link.target).x) / 2"
                :y="(positionOf(link.source).y + positionOf(link.target).y) / 2 - 8"
                text-anchor="middle"
              >
                {{ truncateLabel(link.label || link.relation || '关联', 12) }}
              </text>
            </g>
          </g>

          <g class="nodes">
            <g
              v-for="node in visibleNodes"
              :key="node.id"
              :class="['node', nodeType(node), highlightClassForNode(node)]"
              :transform="`translate(${positionOf(node.id).x}, ${positionOf(node.id).y})`"
              @click="selectNode(node)"
            >
              <title>{{ node.name }}</title>
              <circle :r="nodeRadius(node)" />
              <text :y="nodeRadius(node) + 17" text-anchor="middle">
                <tspan
                  v-for="(line, index) in nodeLabelLines(node.name)"
                  :key="`${node.id}-label-${index}`"
                  x="0"
                  :dy="index === 0 ? 0 : 13"
                >
                  {{ line }}
                </tspan>
              </text>
            </g>
          </g>
        </g>
      </svg>

      <div class="floating-toolbar" aria-label="图谱操作">
        <button type="button" title="放大" @click="zoomIn">+</button>
        <button type="button" title="缩小" @click="zoomOut">-</button>
        <div class="toolbar-menu">
          <button type="button" title="切换布局" @click="toggleMenu('layout')">{{ layoutLabel }}</button>
          <div v-if="openMenu === 'layout'" class="toolbar-popover">
            <button
              v-for="layout in layoutOptions"
              :key="layout.value"
              :class="{ selected: layoutMode === layout.value }"
              type="button"
              @click="setLayout(layout.value)"
            >
              {{ layout.label }}
            </button>
          </div>
        </div>
        <button type="button" title="导出 SVG" @click="exportSvg">SVG</button>
        <button type="button" title="导出 PNG" @click="exportPng">PNG</button>
        <button type="button" title="导出 JSON" @click="exportJson">JSON</button>
      </div>

      <div v-if="isGenerating" class="graph-state generating">
        <span class="graph-spinner"></span>
        <strong>生成图谱中</strong>
        <p>{{ generationStep }}</p>
      </div>

      <div v-else-if="!hasGenerated" class="graph-state">
        <strong>点击“生成图谱”开始构建</strong>
        <p>切换知识库或模型后不会自动生成，确认配置后再手动生成。</p>
      </div>

      <div v-else-if="!visibleNodes.length" class="graph-state">
        <strong>暂未抽取到可展示节点</strong>
        <p>请确认知识库里已有文档，或换一个更适合结构化抽取的模型后重新生成。</p>
      </div>
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
        <div class="meta-grid">
          <article>
            <span>类别</span>
            <strong>{{ selectedItem?.type || selectedItem?.relation || '未知' }}</strong>
          </article>
          <article>
            <span>关联边</span>
            <strong>{{ relatedLinks.length }}</strong>
          </article>
          <article>
            <span>追溯切片</span>
            <strong>{{ selectedItem?.chunk_id || '-' }}</strong>
          </article>
        </div>

        <details open>
          <summary>关系链</summary>
          <ul>
            <li v-for="link in relatedLinks" :key="linkKey(link)">
              {{ nodeName(link.source) }} -> {{ link.label || link.relation || '关联' }} -> {{ nodeName(link.target) }}
            </li>
          </ul>
        </details>

        <details open>
          <summary>原始文本出处</summary>
          <p class="source-text" v-html="highlightedSourceHtml"></p>
        </details>
      </div>
    </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { getGraphData, listChatModels, listKnowledgeBases, updateGraphModel } from '../api/client'

const props = defineProps({
  knowledgeBases: {
    type: Array,
    default: () => [],
  },
})

const selectedKbId = ref('')
const selectedModelId = ref(localStorage.getItem('graph_model_id') || '')
const searchText = ref('')
const graphData = ref(null)
const localKnowledgeBases = ref([])
const selectedItem = ref(null)
const inspectorOpen = ref(false)
const openMenu = ref('')
const modelOptions = ref([])
const graphError = ref('')
const isGenerating = ref(false)
const hasGenerated = ref(false)
const zoom = ref(1)
const pan = reactive({ x: 0, y: 0 })
const isDraggingCanvas = ref(false)
const dragStart = reactive({ x: 0, y: 0, panX: 0, panY: 0 })
const layoutMode = ref('force')
const generationStepIndex = ref(0)
const svgRef = ref(null)
let generationTimer = null
const SVG_VIEWBOX_WIDTH = 960
const SVG_VIEWBOX_HEIGHT = 620
const MIN_ZOOM = 0.55
const MAX_ZOOM = 2.2

const filters = reactive({ project: true, person: true, metric: true })
const emptyGraph = { nodes: [], links: [], communities: [], hit_path: [] }
const generationSteps = [
  '正在读取知识库文档...',
  '正在抽取核心实体...',
  '正在构建关系网络...',
  '正在生成社区结构...',
]
const filterOptions = [
  { key: 'project', label: '项目/机构', color: '#1890ff' },
  { key: 'person', label: '人物', color: '#52c41a' },
  { key: 'metric', label: '数值/概念', color: '#fa8c16' },
]
const layoutOptions = [
  { value: 'force', label: '力导向' },
  { value: 'ring', label: '环形' },
  { value: 'tree', label: '树状' },
]

const availableKnowledgeBases = computed(() =>
  props.knowledgeBases.length ? props.knowledgeBases : localKnowledgeBases.value
)
const currentGraph = computed(() => graphData.value || emptyGraph)
const selectedKbLabel = computed(() =>
  availableKnowledgeBases.value.find(kb => kb.id === selectedKbId.value)?.name || '请选择知识库'
)
const selectedModel = computed(() =>
  modelOptions.value.find(model => model.id === selectedModelId.value) || modelOptions.value[0] || null
)
const selectedModelLabel = computed(() => selectedModel.value?.display_name || 'Mock Chat')
const layoutLabel = computed(() => layoutOptions.find(item => item.value === layoutMode.value)?.label || '布局')
const generationStep = computed(() => generationSteps[generationStepIndex.value % generationSteps.length])
const canvasTransform = computed(() => `translate(${pan.x} ${pan.y}) scale(${zoom.value})`)
const graphBuildHint = computed(() => {
  if (isGenerating.value) return generationStep.value
  if (graphError.value) return graphError.value
  const build = graphData.value?.build
  if (!build) return '选择知识库和模型后，点击生成图谱'
  const hiddenCount = Math.max((build.entity_count || 0) - visibleNodes.value.length, 0)
  const suffix = hiddenCount ? ` · 已优先展示 ${visibleNodes.value.length} 个关键节点` : ''
  return `${build.mode === 'rule_based' ? '规则抽取' : '模型抽取'} · ${build.entity_count} 实体 · ${build.relationship_count} 关系 · ${build.community_count} 社区${suffix}`
})

const graphDegree = computed(() => {
  const degrees = new Map()
  currentGraph.value.nodes.forEach(node => degrees.set(node.id, 0))
  currentGraph.value.links.forEach((link) => {
    degrees.set(link.source, (degrees.get(link.source) || 0) + 1)
    degrees.set(link.target, (degrees.get(link.target) || 0) + 1)
  })
  return degrees
})

const filteredNodes = computed(() => currentGraph.value.nodes.filter((node) => {
  const type = nodeType(node)
  const matchesType = filters[type] !== false
  const query = searchText.value.trim().toLowerCase()
  const matchesSearch = !query || String(node.name || '').toLowerCase().includes(query)
  return matchesType && matchesSearch
}))

const visibleNodes = computed(() => {
  const nodes = filteredNodes.value
  const limit = searchText.value.trim() ? 96 : 56
  if (nodes.length <= limit) return nodes

  const selectedId = selectedItem.value?.id
  const communities = nodes.filter(node => node.type === 'community').slice(0, 8)
  const ranked = nodes
    .filter(node => node.type !== 'community')
    .sort((a, b) => {
      if (a.id === selectedId) return -1
      if (b.id === selectedId) return 1
      const degreeDiff = (graphDegree.value.get(b.id) || 0) - (graphDegree.value.get(a.id) || 0)
      if (degreeDiff) return degreeDiff
      return String(a.name || '').length - String(b.name || '').length
    })

  const seen = new Set()
  return [...communities, ...ranked]
    .filter((node) => {
      if (seen.has(node.id)) return false
      seen.add(node.id)
      return true
    })
    .slice(0, limit)
})

const visibleNodeIds = computed(() => new Set(visibleNodes.value.map(node => node.id)))
const visibleLinks = computed(() => currentGraph.value.links
  .filter(link => visibleNodeIds.value.has(link.source) && visibleNodeIds.value.has(link.target))
  .sort((a, b) => {
    const degreeA = (graphDegree.value.get(a.source) || 0) + (graphDegree.value.get(a.target) || 0)
    const degreeB = (graphDegree.value.get(b.source) || 0) + (graphDegree.value.get(b.target) || 0)
    return degreeB - degreeA
  })
  .slice(0, 110)
)
const showLinkLabel = computed(() => visibleLinks.value.length <= 36)
const visibleDegree = computed(() => {
  const degrees = new Map()
  visibleNodes.value.forEach(node => degrees.set(node.id, 0))
  visibleLinks.value.forEach((link) => {
    degrees.set(link.source, (degrees.get(link.source) || 0) + 1)
    degrees.set(link.target, (degrees.get(link.target) || 0) + 1)
  })
  return degrees
})
const graphLayout = computed(() => {
  if (layoutMode.value === 'ring') return ringLayout(visibleNodes.value)
  if (layoutMode.value === 'tree') return treeLayout(visibleNodes.value)
  return forceLikeLayout(visibleNodes.value)
})
const relatedLinks = computed(() => {
  if (!selectedItem.value) return []
  if (selectedItem.value.source && selectedItem.value.target) return [selectedItem.value]
  return currentGraph.value.links.filter(link =>
    link.source === selectedItem.value.id || link.target === selectedItem.value.id
  )
})
const focusNodeIds = computed(() => {
  if (!selectedItem.value || selectedItem.value.source) return new Set()
  const ids = new Set([selectedItem.value.id])
  const firstHop = new Set()
  currentGraph.value.links.forEach((link) => {
    if (link.source === selectedItem.value.id) firstHop.add(link.target)
    if (link.target === selectedItem.value.id) firstHop.add(link.source)
  })
  firstHop.forEach(id => ids.add(id))
  currentGraph.value.links.forEach((link) => {
    if (firstHop.has(link.source)) ids.add(link.target)
    if (firstHop.has(link.target)) ids.add(link.source)
  })
  return ids
})
const focusLinkKeys = computed(() => {
  if (!selectedItem.value) return new Set()
  if (selectedItem.value.source && selectedItem.value.target) return new Set([linkKey(selectedItem.value)])
  return new Set(currentGraph.value.links
    .filter(link => focusNodeIds.value.has(link.source) && focusNodeIds.value.has(link.target))
    .map(linkKey)
  )
})
const highlightedSourceHtml = computed(() => {
  const source = selectedItem.value?.source_text || '暂无可追溯文本。'
  const names = new Set()
  if (selectedItem.value?.name) names.add(selectedItem.value.name)
  relatedLinks.value.forEach((link) => {
    names.add(nodeName(link.source))
    names.add(nodeName(link.target))
  })
  return highlightText(source, [...names].filter(Boolean))
})

onMounted(async () => {
  await Promise.all([loadModels(), loadKnowledgeBaseOptions()])
  ensureSelectedKnowledgeBase()
})

watch(() => props.knowledgeBases, () => {
  ensureSelectedKnowledgeBase()
}, { deep: true })

watch(inspectorOpen, () => {
  window.setTimeout(() => {
    fitView()
  }, 320)
})

onBeforeUnmount(() => {
  stopGenerationTimer()
})

async function loadModels() {
  try {
    const data = await listChatModels()
    modelOptions.value = data.models || []
    const preferred = selectedModelId.value || data.default_model_id || modelOptions.value[0]?.id || ''
    selectedModelId.value = modelOptions.value.some(model => model.id === preferred)
      ? preferred
      : modelOptions.value[0]?.id || ''
    if (selectedModelId.value) localStorage.setItem('graph_model_id', selectedModelId.value)
  } catch {
    modelOptions.value = [{
      id: 'mock-chat',
      provider: 'mock',
      display_name: 'Mock Chat',
      model_name: 'mock-chat',
    }]
    selectedModelId.value = 'mock-chat'
  }
}

async function loadKnowledgeBaseOptions() {
  try {
    localKnowledgeBases.value = await listKnowledgeBases()
  } catch (error) {
    graphError.value = error?.message || '知识库列表加载失败'
    localKnowledgeBases.value = []
  }
}

function ensureSelectedKnowledgeBase() {
  const bases = availableKnowledgeBases.value
  if (!bases.length) {
    selectedKbId.value = ''
    return
  }
  if (!selectedKbId.value || !bases.some(kb => kb.id === selectedKbId.value)) {
    selectedKbId.value = bases[0].id
    resetGraphView()
  }
}

function toggleMenu(menu) {
  openMenu.value = openMenu.value === menu ? '' : menu
}

async function chooseKb(kbId) {
  selectedKbId.value = kbId
  openMenu.value = ''
  resetGraphView()
  if (selectedKbId.value && selectedModelId.value) {
    await saveGraphModel().catch(() => {})
  }
}

async function chooseModel(modelId) {
  selectedModelId.value = modelId
  localStorage.setItem('graph_model_id', modelId)
  openMenu.value = ''
  resetGraphView()
  await saveGraphModel().catch((error) => {
    graphError.value = error?.message || '图谱模型保存失败'
  })
}

async function generateGraph() {
  if (!selectedKbId.value) {
    graphError.value = '请先选择知识库'
    return
  }
  isGenerating.value = true
  startGenerationTimer()
  graphData.value = null
  selectedItem.value = null
  inspectorOpen.value = false
  try {
    graphError.value = ''
    graphData.value = await getGraphData(selectedKbId.value, selectedModelId.value)
    hasGenerated.value = true
    fitView()
    if (graphData.value?.model?.id) {
      selectedModelId.value = graphData.value.model.id
      localStorage.setItem('graph_model_id', selectedModelId.value)
    }
  } catch (error) {
    hasGenerated.value = true
    graphError.value = error?.message || '图谱生成失败，请检查模型 Key、知识库文档和后端服务'
    graphData.value = emptyGraph
  } finally {
    isGenerating.value = false
    stopGenerationTimer()
  }
}

async function saveGraphModel() {
  if (!selectedKbId.value || !selectedModelId.value) return
  const result = await updateGraphModel(selectedKbId.value, selectedModelId.value)
  if (result?.model?.id) {
    selectedModelId.value = result.model.id
    localStorage.setItem('graph_model_id', selectedModelId.value)
  }
}

function resetGraphView() {
  graphData.value = null
  graphError.value = ''
  hasGenerated.value = false
  selectedItem.value = null
  inspectorOpen.value = false
  fitView()
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
  return graphLayout.value.get(id) || { x: 480, y: 310 }
}

function edgePoints(link) {
  const source = positionOf(link.source)
  const target = positionOf(link.target)
  const dx = target.x - source.x
  const dy = target.y - source.y
  const length = Math.sqrt(dx * dx + dy * dy) || 1
  const targetNode = currentGraph.value.nodes.find(node => node.id === link.target)
  const offset = nodeRadius(targetNode || {}) + 9
  return {
    x1: source.x,
    y1: source.y,
    x2: target.x - (dx / length) * offset,
    y2: target.y - (dy / length) * offset,
  }
}

function nodeRadius(node) {
  if (node.type === 'community') return visibleNodes.value.length > 40 ? 17 : 22
  const base = visibleNodes.value.length > 60 ? 7 : visibleNodes.value.length > 36 ? 10 : 16
  const degreeBoost = Math.min(visibleDegree.value.get(node.id) || 0, 5)
  return base + degreeBoost
}

function nodeType(node) {
  const type = String(node.type || '').toLowerCase()
  if (['person', 'people', 'human', '人物'].includes(type)) return 'person'
  if (['metric', 'number', 'date', 'amount', 'concept', '数值', '概念'].includes(type)) return 'metric'
  if (type === 'community') return 'community'
  return 'project'
}

function truncateLabel(name = '', max = 6) {
  const text = String(name)
  return text.length > max ? `${text.slice(0, max)}...` : text
}

function nodeLabelLines(name = '') {
  const text = String(name).trim()
  if (text.length <= 10) return [text]
  if (text.length <= 18) return [text.slice(0, 9), text.slice(9)]
  return [text.slice(0, 9), `${text.slice(9, 17)}...`]
}

function nodeName(id) {
  return currentGraph.value.nodes.find(node => node.id === id)?.name || id
}

function linkKey(link) {
  return `${link.source}-${link.target}-${link.relation || link.label || ''}`
}

function highlightClassForNode(node) {
  if (!selectedItem.value) return ''
  if (selectedItem.value.source) return ''
  if (selectedItem.value.id === node.id) return 'active'
  return focusNodeIds.value.has(node.id) ? 'highlighted' : 'dimmed'
}

function highlightClassForLink(link) {
  if (!selectedItem.value) return ''
  return focusLinkKeys.value.has(linkKey(link)) ? 'highlighted' : 'dimmed'
}

function forceLikeLayout(nodes) {
  const ordered = orderedNodes(nodes)
  const positions = new Map()
  if (!ordered.length) return positions
  if (ordered.length === 1) {
    positions.set(ordered[0].id, { x: 480, y: 310 })
    return positions
  }
  const center = { x: 480, y: 310 }
  positions.set(ordered[0].id, center)
  const goldenAngle = Math.PI * (3 - Math.sqrt(5))
  ordered.slice(1).forEach((node, index) => {
    const rank = index + 1
    const radius = Math.sqrt(rank / ordered.length)
    const angle = rank * goldenAngle
    positions.set(node.id, {
      x: center.x + Math.cos(angle) * 405 * radius,
      y: center.y + Math.sin(angle) * 245 * radius,
    })
  })
  return positions
}

function ringLayout(nodes) {
  const ordered = orderedNodes(nodes)
  const positions = new Map()
  if (!ordered.length) return positions
  const center = { x: 480, y: 310 }
  const radiusX = ordered.length > 24 ? 370 : 300
  const radiusY = ordered.length > 24 ? 230 : 180
  ordered.forEach((node, index) => {
    const angle = (index / ordered.length) * Math.PI * 2 - Math.PI / 2
    positions.set(node.id, {
      x: center.x + Math.cos(angle) * radiusX,
      y: center.y + Math.sin(angle) * radiusY,
    })
  })
  return positions
}

function treeLayout(nodes) {
  const ordered = orderedNodes(nodes)
  const positions = new Map()
  if (!ordered.length) return positions
  const levels = [1, 4, 8, 14, 28]
  let index = 0
  levels.forEach((count, level) => {
    const row = ordered.slice(index, index + count)
    const y = 90 + level * 108
    row.forEach((node, itemIndex) => {
      const step = 780 / Math.max(row.length, 1)
      positions.set(node.id, { x: 90 + step * (itemIndex + 0.5), y })
    })
    index += count
  })
  ordered.slice(index).forEach((node, itemIndex) => {
    positions.set(node.id, {
      x: 90 + (itemIndex % 14) * 58,
      y: 545,
    })
  })
  return positions
}

function orderedNodes(nodes) {
  return [...nodes].sort((a, b) => {
    const degreeDiff = (visibleDegree.value.get(b.id) || 0) - (visibleDegree.value.get(a.id) || 0)
    if (degreeDiff) return degreeDiff
    return String(a.name || '').localeCompare(String(b.name || ''))
  })
}

function setLayout(value) {
  layoutMode.value = value
  openMenu.value = ''
  fitView()
}

function zoomIn() {
  zoom.value = clampZoom(zoom.value + 0.12)
}

function zoomOut() {
  zoom.value = clampZoom(zoom.value - 0.12)
}

function fitView() {
  zoom.value = 1
  pan.x = 0
  pan.y = 0
}

function clampZoom(value) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number(value.toFixed(2))))
}

function handleCanvasWheel(event) {
  if (!svgRef.value) return
  const rect = svgRef.value.getBoundingClientRect()
  if (!rect.width || !rect.height) return

  const pointer = {
    x: ((event.clientX - rect.left) / rect.width) * SVG_VIEWBOX_WIDTH,
    y: ((event.clientY - rect.top) / rect.height) * SVG_VIEWBOX_HEIGHT,
  }
  const oldZoom = zoom.value
  const zoomDelta = event.deltaY < 0 ? 0.12 : -0.12
  const nextZoom = clampZoom(oldZoom + zoomDelta)
  if (nextZoom === oldZoom) return

  const ratio = nextZoom / oldZoom
  pan.x = pointer.x - (pointer.x - pan.x) * ratio
  pan.y = pointer.y - (pointer.y - pan.y) * ratio
  zoom.value = nextZoom
}

function startCanvasDrag(event) {
  if (event.button !== 0 || event.target.closest?.('.node, .edge')) return
  isDraggingCanvas.value = true
  dragStart.x = event.clientX
  dragStart.y = event.clientY
  dragStart.panX = pan.x
  dragStart.panY = pan.y
}

function dragCanvas(event) {
  if (!isDraggingCanvas.value) return
  pan.x = dragStart.panX + event.clientX - dragStart.x
  pan.y = dragStart.panY + event.clientY - dragStart.y
}

function endCanvasDrag() {
  isDraggingCanvas.value = false
}

function exportJson() {
  downloadFile('knowledge-graph.json', JSON.stringify(currentGraph.value, null, 2), 'application/json')
}

function exportSvg() {
  if (!svgRef.value) return
  const svgText = new XMLSerializer().serializeToString(svgRef.value)
  downloadFile('knowledge-graph.svg', svgText, 'image/svg+xml')
}

function exportPng() {
  if (!svgRef.value) return
  const svgText = new XMLSerializer().serializeToString(svgRef.value)
  const image = new Image()
  const blob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  image.onload = () => {
    const canvas = document.createElement('canvas')
    canvas.width = 1920
    canvas.height = 1240
    const context = canvas.getContext('2d')
    context.fillStyle = '#fbfcfd'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
    URL.revokeObjectURL(url)
    canvas.toBlob((pngBlob) => {
      if (!pngBlob) return
      downloadBlob('knowledge-graph.png', pngBlob)
    })
  }
  image.src = url
}

function downloadFile(filename, content, type) {
  downloadBlob(filename, new Blob([content], { type }))
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function startGenerationTimer() {
  stopGenerationTimer()
  generationStepIndex.value = 0
  generationTimer = window.setInterval(() => {
    generationStepIndex.value += 1
  }, 1300)
}

function stopGenerationTimer() {
  if (generationTimer) {
    window.clearInterval(generationTimer)
    generationTimer = null
  }
}

function highlightText(text, terms) {
  let html = escapeHtml(text)
  const cleanTerms = [...new Set(terms.map(term => String(term).trim()).filter(term => term.length >= 2))]
    .sort((a, b) => b.length - a.length)
  cleanTerms.forEach((term) => {
    const pattern = new RegExp(escapeRegExp(escapeHtml(term)), 'gi')
    html = html.replace(pattern, match => `<mark>${match}</mark>`)
  })
  return html
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
</script>

<style scoped>
.graph-workspace {
  display: flex;
  min-width: 0;
  height: 100vh;
  flex-direction: column;
  overflow: hidden;
  background:
    linear-gradient(rgba(20, 27, 33, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20, 27, 33, 0.04) 1px, transparent 1px),
    #f8fafc;
  background-size: 36px 36px;
}

.graph-controls {
  position: relative;
  z-index: 5;
  display: flex;
  flex: none;
  flex-shrink: 0;
  width: auto;
  max-width: none;
  min-height: 54px;
  flex-wrap: nowrap;
  align-items: center;
  gap: 10px;
  align-self: stretch;
  margin: 12px 24px 8px;
  border: 1px solid rgba(226, 232, 240, 0.92);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.9);
  padding: 7px 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 10px 24px rgba(15, 23, 42, 0.05);
  box-sizing: border-box;
  overflow: visible;
  backdrop-filter: blur(14px);
}

.graph-controls input {
  flex: 0 0 170px;
  width: 170px;
  min-width: 170px;
  height: 36px;
  min-height: 36px;
  border: 1px solid #e2e8f0;
  border-radius: 9px;
  background: #f8fafc;
  padding: 0 11px;
  color: #0f172a;
  font-size: 12px;
  outline: 0;
  transition: width 0.16s ease, border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
}

.graph-controls input:focus {
  width: 210px;
  flex-basis: 210px;
  border-color: #93c5fd;
  background: #ffffff;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
}

.graph-picker {
  position: relative;
  flex: 0 0 auto;
  min-width: 0;
}

.graph-picker > button {
  display: flex;
  width: 178px;
  height: 36px;
  min-height: 36px;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 9px;
  background: #f8fafc;
  padding: 0 10px;
  color: #0f172a;
  cursor: pointer;
  transition: border-color 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
}

.graph-picker > button:hover {
  border-color: #cbd5e1;
  background: #ffffff;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}

.graph-picker > button span {
  flex: 0 0 auto;
  color: #64748b;
  font-size: 11px;
  font-weight: 750;
}

.graph-picker > button strong {
  min-width: 0;
  max-width: 104px;
  overflow: hidden;
  font-size: 13px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-picker > button {
  width: 210px;
}

.model-picker > button strong {
  max-width: 130px;
}

.graph-popover,
.toolbar-popover {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  z-index: 40;
  display: grid;
  min-width: 220px;
  max-height: 280px;
  gap: 4px;
  overflow: auto;
  border: 1px solid var(--border-color);
  border-radius: 14px;
  background: var(--surface);
  padding: 6px;
  box-shadow: var(--shadow-md);
}

.graph-popover button,
.toolbar-popover button {
  display: grid;
  gap: 2px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  padding: 9px 10px;
  color: var(--text-primary);
  text-align: left;
  cursor: pointer;
  font-weight: 750;
}

.graph-popover button:hover,
.graph-popover button.selected,
.toolbar-popover button:hover,
.toolbar-popover button.selected {
  background: rgba(37, 99, 235, 0.09);
  color: #1d4ed8;
}

.graph-popover small {
  color: var(--text-muted);
  font-size: 11px;
}

.toolbar-divider {
  flex: 0 0 auto;
  width: 1px;
  height: 20px;
  background: #e2e8f0;
}

.entity-filter {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 7px;
  color: #475569;
  font-size: 12px;
}

.filter-pill {
  display: flex;
  flex: 0 0 auto;
  height: 32px;
  align-items: center;
  gap: 6px;
  border: 1px solid #e2e8f0;
  border-radius: 9px;
  background: #f8fafc;
  color: #94a3b8;
  padding: 0 11px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 650;
  transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease, transform 0.16s ease;
  white-space: nowrap;
}

.filter-pill:hover {
  background: #f1f5f9;
  color: #475569;
}

.filter-pill.active {
  font-weight: 800;
}

.filter-pill.project.active {
  border-color: #bfdbfe;
  background: #eff6ff;
  color: #1d4ed8;
}

.filter-pill.person.active {
  border-color: #bbf7d0;
  background: #f0fdf4;
  color: #15803d;
}

.filter-pill.metric.active {
  border-color: #fde68a;
  background: #fffbeb;
  color: #b45309;
}

.filter-pill span {
  width: 8px;
  height: 8px;
  flex: 0 0 auto;
  border-radius: 999px;
  background: var(--dot-color);
}

.btn-primary:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.graph-controls .btn-primary {
  flex: 0 0 auto;
  height: 36px;
  min-height: 36px;
  border-radius: 9px;
  background: #0f172a;
  padding: 0 16px;
  font-size: 12px;
  font-weight: 800;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.18);
}

.graph-controls .btn-primary:hover:not(:disabled) {
  background: #1e293b;
}

.graph-controls .btn-primary span {
  color: #fbbf24;
}

.graph-body {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 1;
  overflow: hidden;
}

.model-hint {
  position: absolute;
  left: 22px;
  bottom: 22px;
  z-index: 2;
  display: flex;
  max-width: min(640px, calc(100% - 44px));
  align-items: center;
  gap: 8px;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  padding: 8px 12px;
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(12px);
}

.model-hint span {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 750;
}

.model-hint strong {
  color: #1d4ed8;
  font-size: 13px;
}

.model-hint em {
  overflow: hidden;
  color: var(--text-secondary);
  font-size: 12px;
  font-style: normal;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.graph-canvas {
  position: relative;
  min-width: 0;
  min-height: 0;
  flex: 1;
  overflow: hidden;
  transition: flex-basis 0.3s ease, width 0.3s ease;
}

.graph-canvas svg {
  width: 100%;
  height: 100%;
  cursor: grab;
  user-select: none;
}

.graph-canvas svg.dragging {
  cursor: grabbing;
}

.floating-toolbar {
  position: absolute;
  right: 22px;
  bottom: 22px;
  z-index: 4;
  display: flex;
  gap: 6px;
  align-items: center;
  border: 1px solid rgba(226, 232, 240, 0.88);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.84);
  padding: 7px;
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(14px);
}

.floating-toolbar button {
  min-width: 32px;
  height: 32px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: #475569;
  cursor: pointer;
  font-size: 12px;
  font-weight: 800;
}

.floating-toolbar button:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.toolbar-menu {
  position: relative;
}

.toolbar-popover {
  top: auto;
  right: 0;
  bottom: calc(100% + 8px);
  left: auto;
  min-width: 130px;
}

.graph-state {
  position: absolute;
  top: 50%;
  left: 50%;
  z-index: 2;
  display: grid;
  width: min(420px, calc(100% - 40px));
  transform: translate(-50%, -50%);
  place-items: center;
  gap: 10px;
  border: 1px dashed rgba(148, 163, 184, 0.55);
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.72);
  padding: 26px;
  color: var(--text-secondary);
  text-align: center;
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(10px);
}

.graph-state strong {
  color: var(--text-primary);
  font-size: 18px;
}

.graph-state p {
  max-width: 320px;
  font-size: 13px;
  line-height: 1.7;
}

.graph-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(37, 99, 235, 0.16);
  border-top-color: #2563eb;
  border-radius: 999px;
  animation: graph-spin 0.8s linear infinite;
}

@keyframes graph-spin {
  to {
    transform: rotate(360deg);
  }
}

.edge line {
  cursor: pointer;
  stroke: #94a3b8;
  stroke-opacity: 0.76;
  stroke-width: 1.6;
}

.edge text {
  fill: var(--text-secondary);
  font-size: 11px;
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
  fill: #1890ff;
}

.node.person circle {
  fill: #52c41a;
}

.node.metric circle {
  fill: #fa8c16;
}

.node.community circle {
  fill: #0f766e;
}

.node.active circle {
  stroke: #111827;
  stroke-width: 4;
}

.node text {
  pointer-events: none;
  fill: #334155;
  font-size: 11px;
  font-weight: 800;
  paint-order: stroke;
  stroke: rgba(255, 255, 255, 0.92);
  stroke-width: 4px;
}

.node.dimmed,
.edge.dimmed {
  opacity: 0.15;
}

.node.highlighted,
.edge.highlighted,
.node.active {
  opacity: 1;
}

.edge.highlighted line {
  stroke: #2563eb;
  stroke-width: 2.4;
}

.inspector {
  position: relative;
  z-index: 6;
  width: 0;
  min-width: 0;
  flex: 0 0 0;
  flex-shrink: 0;
  overflow: hidden;
  border-left: 1px solid var(--border-color);
  background: rgba(255, 255, 255, 0.98);
  box-shadow: var(--shadow-md);
  transition: flex-basis 0.3s ease, width 0.3s ease;
}

.inspector.open {
  width: 380px;
  flex-basis: 380px;
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
  min-width: 0;
  flex-direction: column;
}

.inspector-head strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.inspector-head span {
  color: var(--text-muted);
  font-size: 12px;
}

.inspector-body {
  display: grid;
  gap: 14px;
  padding: 16px;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.meta-grid article {
  display: grid;
  gap: 4px;
  border: 1px solid #eef2f7;
  border-radius: 12px;
  background: #f8fafc;
  padding: 10px;
}

.meta-grid span,
.inspector-body summary {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 750;
}

.meta-grid strong {
  overflow: hidden;
  color: var(--text-primary);
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.inspector-body details {
  border: 1px solid #eef2f7;
  border-radius: 14px;
  background: #ffffff;
  padding: 12px;
}

.inspector-body ul {
  display: grid;
  gap: 8px;
  margin-top: 10px;
  padding-left: 18px;
  color: var(--text-secondary);
}

.source-text {
  margin-top: 10px;
  border-radius: 10px;
  background: #f8fafc;
  padding: 12px;
  color: var(--text-secondary);
  line-height: 1.7;
}

.source-text :deep(mark) {
  border-radius: 4px;
  background: #fef3c7;
  color: #92400e;
  padding: 0 2px;
}

@media (max-width: 920px) {
  .graph-controls {
    margin: 10px 12px 8px;
    padding: 7px 10px;
    flex-wrap: nowrap;
    border-radius: 12px;
  }

  .graph-controls input {
    flex-basis: 150px;
    width: 150px;
    min-width: 150px;
  }

  .inspector.open {
    width: min(340px, 48vw);
    flex-basis: min(340px, 48vw);
  }

  .model-hint {
    border-radius: 14px;
    align-items: flex-start;
    flex-direction: column;
  }

  .floating-toolbar {
    right: 12px;
    bottom: 12px;
    flex-wrap: wrap;
    max-width: calc(100% - 24px);
  }
}

@media (max-width: 1280px) {
  .graph-picker > button {
    width: 150px;
  }

  .graph-picker > button strong {
    max-width: 78px;
  }

  .model-picker > button {
    width: 180px;
  }

  .model-picker > button strong {
    max-width: 108px;
  }

  .graph-controls input {
    flex-basis: 145px;
    width: 145px;
    min-width: 145px;
  }

  .filter-pill {
    padding: 0 9px;
  }
}
</style>
