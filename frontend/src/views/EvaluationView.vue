<template>
  <!-- 评测页面 — 五模式对比表/图 -->
  <div class="eval-page">
    <h2>📊 评测对比</h2>

    <div class="eval-toolbar">
      <select v-model="selectedKbId">
        <option value="kb_001">产品技术手册</option>
        <option value="kb_002">公司内部制度</option>
      </select>
      <button @click="runEval" class="btn-primary" :disabled="running">
        {{ running ? '评测中...' : '运行评测' }}
      </button>
      <span class="eval-note">选择知识库，点击按钮进行五模式对比</span>
    </div>

    <!-- 结果对比表 -->
    <div v-if="results.length > 0" class="eval-results">
      <table class="eval-table">
        <thead>
          <tr>
            <th>模式</th>
            <th>Hit@3</th>
            <th>MRR</th>
            <th>首 Token 延迟</th>
            <th>总耗时</th>
            <th>Token 总数</th>
            <th>估算费用</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in results" :key="r.run_id">
            <td><strong>{{ r.mode.toUpperCase() }}</strong></td>
            <td>{{ (r.metrics.retrieval.hit_at_3 * 100).toFixed(1) }}%</td>
            <td>{{ r.metrics.retrieval.mrr.toFixed(3) }}</td>
            <td>{{ r.metrics.system.first_token_latency_ms.toFixed(0) }}ms</td>
            <td>{{ r.latency_ms.toFixed(0) }}ms</td>
            <td>{{ r.metrics.system.total_tokens }}</td>
            <td>${{ r.metrics.system.estimated_cost_usd.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>

      <!-- 简易柱状图（CSS Bar Chart） -->
      <div class="bar-chart">
        <h3>Hit@3 对比</h3>
        <div class="bar-row" v-for="r in results" :key="r.run_id">
          <span class="bar-label">{{ r.mode }}</span>
          <div class="bar-bg">
            <div class="bar-fill" :style="{ width: (r.metrics.retrieval.hit_at_3 * 100) + '%' }" />
          </div>
          <span class="bar-value">{{ (r.metrics.retrieval.hit_at_3 * 100).toFixed(1) }}%</span>
        </div>
      </div>

      <div class="bar-chart">
        <h3>总耗时对比</h3>
        <div class="bar-row" v-for="r in results" :key="'t'+r.run_id">
          <span class="bar-label">{{ r.mode }}</span>
          <div class="bar-bg">
            <div class="bar-fill latency" :style="{ width: (r.latency_ms / maxLatency * 100) + '%' }" />
          </div>
          <span class="bar-value">{{ r.latency_ms.toFixed(0) }}ms</span>
        </div>
      </div>
    </div>

    <div v-if="!results.length" class="empty-state">
      <p>点击"运行评测"查看五模式对比结果</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { runEvaluation, getEvaluationResult } from '../api/client'

const selectedKbId = ref('kb_001')
const results = ref([])
const running = ref(false)

const maxLatency = computed(() =>
  Math.max(...results.value.map(r => r.latency_ms), 1)
)

async function runEval() {
  running.value = true
  try {
    const { run_id } = await runEvaluation(selectedKbId.value, [
      'naive', 'advanced', 'modular', 'graph', 'agentic'
    ])
    results.value = await getEvaluationResult(run_id)
  } catch (e) {
    alert('评测失败：' + e.message)
  } finally {
    running.value = false
  }
}
</script>

<style scoped>
.eval-page { padding: 24px; }
.eval-toolbar { display: flex; gap: 12px; align-items: center; margin: 16px 0; }
.eval-toolbar select { padding: 8px; border-radius: 6px; }
.eval-note { color: var(--text-muted); font-size: 13px; }
.eval-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
.eval-table th, .eval-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
.eval-table th { font-weight: 600; font-size: 13px; color: var(--text-muted); }

.bar-chart { margin: 24px 0; }
.bar-chart h3 { margin-bottom: 12px; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.bar-label { width: 80px; font-size: 13px; font-weight: 600; }
.bar-bg { flex: 1; height: 20px; background: var(--bg-secondary); border-radius: 10px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 10px; transition: width .5s; }
.bar-fill.latency { background: #fdae6b; }
.bar-value { width: 80px; font-size: 13px; text-align: right; }
.empty-state { text-align: center; margin-top: 80px; color: var(--text-muted); }
</style>
