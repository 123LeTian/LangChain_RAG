<template>
  <!-- 图谱可视化页面 — 节点、边、社区、命中路径 -->
  <div class="graph-page">
    <h2>🕸️ 知识图谱</h2>

    <div class="graph-toolbar">
      <select v-model="selectedKbId">
        <option value="kb_001">产品技术手册</option>
        <option value="kb_002">公司内部制度</option>
        <option value="kb_003">行业白皮书合集</option>
      </select>
      <button @click="loadGraph" class="btn-primary">加载图谱</button>
    </div>

    <!-- 图谱画布（使用 CSS 变通实现力导向布局的静态可视化） -->
    <div v-if="graphData" class="graph-container">
      <div class="graph-canvas">
        <svg viewBox="0 0 800 500" class="graph-svg">
          <!-- 边（连线） -->
          <line
            v-for="(link, i) in graphData.links" :key="'l'+i"
            :x1="nodePos[link.source]?.x"
            :y1="nodePos[link.source]?.y"
            :x2="nodePos[link.target]?.x"
            :y2="nodePos[link.target]?.y"
            :class="['graph-link', { hit: isHitEdge(link) }]"
          />
          <!-- 节点 -->
          <g v-for="node in graphData.nodes" :key="node.id">
            <circle
              :cx="nodePos[node.id]?.x"
              :cy="nodePos[node.id]?.y"
              :r="node.type === 'community' ? 18 : 12"
              :class="['graph-node', node.type, { hit: graphData.hit_path?.includes(node.id) }]"
            />
            <text
              :x="nodePos[node.id]?.x"
              :y="(nodePos[node.id]?.y || 0) + (node.type === 'community' ? 30 : 24)"
              text-anchor="middle"
              class="node-label"
            >{{ node.name }}</text>
          </g>
        </svg>
      </div>

      <!-- 侧边：社区摘要 -->
      <div class="community-panel">
        <h3>社区摘要</h3>
        <div v-for="c in graphData.communities" :key="c.id" class="community-card">
          <h4>{{ c.name }}</h4>
          <p>{{ c.summary }}</p>
          <div class="community-stats">
            <span>实体: {{ c.entity_count }}</span>
            <span>关系: {{ c.relation_count }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-if="!graphData" class="empty-state">
      <p>选择知识库并点击"加载图谱"</p>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { getGraphData } from '../api/client'

const selectedKbId = ref('kb_001')
const graphData = ref(null)

// 预定义节点坐标（模拟力导向布局，未来可接入 ECharts）
const nodePos = reactive({
  e1: { x: 300, y: 150 }, e2: { x: 150, y: 80 },
  e3: { x: 500, y: 300 }, e4: { x: 650, y: 400 },
  e5: { x: 480, y: 120 }, c1: { x: 380, y: 280 },
})

function isHitEdge(link) {
  return graphData.value?.hit_path?.includes(link.source) &&
         graphData.value?.hit_path?.includes(link.target)
}

async function loadGraph() {
  graphData.value = await getGraphData(selectedKbId.value)
}
</script>

<style scoped>
.graph-page { padding: 24px; }
.graph-toolbar { display: flex; gap: 12px; margin: 16px 0; }
.graph-toolbar select { padding: 8px; border-radius: 6px; border: 1px solid var(--border-color); }
.graph-container { display: flex; gap: 24px; }
.graph-canvas { flex: 1; border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden; }
.graph-svg { width: 100%; height: auto; }

.graph-link { stroke: #ccc; stroke-width: 2; }
.graph-link.hit { stroke: var(--accent); stroke-width: 3; }

.graph-node { fill: #6baed6; stroke: white; stroke-width: 2; }
.graph-node.community { fill: #fdae6b; }
.graph-node.hit { fill: var(--accent); stroke: var(--accent); stroke-width: 3; }

.node-label { font-size: 11px; fill: var(--text-primary); }

.community-panel { width: 320px; }
.community-card { padding: 12px; margin-bottom: 12px; background: var(--bg-secondary); border-radius: 8px; }
.community-card h4 { margin: 0 0 4px; }
.community-card p { font-size: 13px; color: var(--text-secondary); line-height: 1.5; }
.community-stats { display: flex; gap: 16px; font-size: 12px; color: var(--text-muted); margin-top: 8px; }
.empty-state { text-align: center; margin-top: 80px; color: var(--text-muted); }
</style>
