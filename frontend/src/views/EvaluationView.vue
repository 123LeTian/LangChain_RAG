<template>
  <div class="eval-page">
    <section class="filter-card">
      <div class="filter-heading">
        <div>
          <p class="eyebrow">Evaluation Comparison</p>
          <h2>RAG 模式评测</h2>
        </div>
        <button
          class="run-button"
          :disabled="running || loadingKbs || loadingModels || !selectedKbId || !selectedModelId"
          @click="runEval"
        >
          <span v-if="running" class="spinner" aria-hidden="true" />
          <svg v-else class="button-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 3l14 9-14 9V3z" />
          </svg>
          {{ running ? '评测中' : '运行评测' }}
        </button>
      </div>

      <div class="filter-controls">
        <div class="control-group">
          <span class="control-label">知识库</span>
          <CustomSelect
            v-model="selectedKbId"
            :options="kbSelectOptions"
            :disabled="loadingKbs || !kbSelectOptions.length"
            placeholder="选择知识库"
          />
        </div>
        <div class="control-group">
          <span class="control-label">公共模型</span>
          <CustomSelect
            v-model="selectedModelId"
            :options="modelSelectOptions"
            :disabled="loadingModels || !modelSelectOptions.length"
            placeholder="选择模型"
          />
        </div>
        <div class="control-group compact">
          <span class="control-label">样本数量</span>
          <CustomSelect
            v-model="selectedSampleLimit"
            :options="sampleLimitOptions"
            placeholder="选择样本数"
          />
        </div>
      </div>
    </section>

    <section v-if="running" class="metrics-card">
      <div class="card-title-row">
        <div>
          <p class="eyebrow">Running</p>
          <h3>正在生成评测结果</h3>
        </div>
      </div>
      <div class="skeleton-table">
        <div v-for="row in 5" :key="row" class="skeleton-row">
          <span class="skeleton-cell short" />
          <span class="skeleton-cell" />
          <span class="skeleton-cell" />
          <span class="skeleton-cell" />
          <span class="skeleton-cell wide" />
        </div>
      </div>
    </section>

    <template v-else-if="results.length > 0">
      <section class="metrics-card">
        <div class="card-title-row">
          <div>
            <p class="eyebrow">Metrics</p>
            <h3>综合指标</h3>
          </div>
          <span class="summary-pill">{{ results.length }} 种模式</span>
        </div>

        <div class="table-wrap">
          <table class="metrics-table">
            <thead>
              <tr>
                <th>模式</th>
                <th class="numeric">Hit@3</th>
                <th class="numeric">MRR</th>
                <th class="numeric">答案覆盖率</th>
                <th class="numeric">P50 延迟</th>
                <th class="numeric">总耗时</th>
                <th class="numeric">Token 总数</th>
                <th class="numeric">估算费用</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in results" :key="r.mode">
                <td>
                  <span class="mode-name">{{ formatMode(r.mode) }}</span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricHit(r), bestHit) }">
                    {{ formatPercent(metricHit(r)) }}
                  </span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricMrr(r), bestMrr) }">
                    {{ metricMrr(r).toFixed(3) }}
                  </span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricCoverage(r), bestCoverage) }">
                    {{ formatPercent(metricCoverage(r)) }}
                  </span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricP50(r), bestP50) }">
                    {{ formatLatency(metricP50(r)) }}
                  </span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricLatency(r), bestLatency) }">
                    {{ formatLatency(metricLatency(r)) }}
                  </span>
                </td>
                <td class="numeric">
                  <span class="metric-value" :class="{ best: isBest(metricTokens(r), bestTokens) }">
                    {{ formatNumber(metricTokens(r)) }}
                  </span>
                </td>
                <td class="numeric">{{ formatCost(r.metrics.system.estimated_cost_usd) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="visual-grid">
        <article class="visual-card tone-indigo">
          <div class="visual-heading">
            <span class="visual-icon" aria-hidden="true">#</span>
            <h3>Hit@3 命中率</h3>
          </div>
          <div class="chart-rows">
            <div v-for="row in chartRows('hit')" :key="'hit-' + row.label" class="chart-row" :title="`${row.label}: ${row.value}`">
              <span class="chart-label">{{ row.label }}</span>
              <div class="chart-track">
                <div class="chart-fill" :style="{ width: `${clampPercent(row.percent)}%` }" />
              </div>
              <span class="chart-value">{{ row.value }}</span>
            </div>
          </div>
        </article>

        <article class="visual-card tone-emerald">
          <div class="visual-heading">
            <span class="visual-icon" aria-hidden="true">%</span>
            <h3>答案覆盖率</h3>
          </div>
          <div class="chart-rows">
            <div v-for="row in chartRows('coverage')" :key="'coverage-' + row.label" class="chart-row" :title="`${row.label}: ${row.value}`">
              <span class="chart-label">{{ row.label }}</span>
              <div class="chart-track">
                <div class="chart-fill" :style="{ width: `${clampPercent(row.percent)}%` }" />
              </div>
              <span class="chart-value">{{ row.value }}</span>
            </div>
          </div>
        </article>

        <article class="visual-card tone-sky">
          <div class="visual-heading">
            <span class="visual-icon" aria-hidden="true">ms</span>
            <h3>总耗时对比</h3>
          </div>
          <div class="chart-rows">
            <div v-for="row in chartRows('latency')" :key="'latency-' + row.label" class="chart-row" :title="`${row.label}: ${row.value}`">
              <span class="chart-label">{{ row.label }}</span>
              <div class="chart-track">
                <div class="chart-fill" :style="{ width: `${clampPercent(row.percent)}%` }" />
              </div>
              <span class="chart-value">{{ row.value }}</span>
            </div>
          </div>
        </article>
      </section>
    </template>

    <section v-else class="empty-card">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 3v18h18" />
          <path d="M7 15v2" />
          <path d="M12 9v8" />
          <path d="M17 5v12" />
        </svg>
      </div>
      <h3>暂无评测数据</h3>
      <p>选择知识库与模型后点击“运行评测”开启对比。</p>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import CustomSelect from '../components/CustomSelect.vue'
import { listKnowledgeBases, listChatModels, runEvaluation, getEvaluationResult } from '../api/client'

const selectedKbId = ref('')
const selectedModelId = ref('')
const selectedSampleLimit = ref(1)
const kbOptions = ref([])
const modelOptions = ref([])
const loadingKbs = ref(false)
const loadingModels = ref(false)
const results = ref([])
const running = ref(false)

const sampleLimitOptions = [
  { value: 1, label: '1 条样本', detail: '快速验收' },
  { value: 3, label: '3 条样本', detail: '轻量对比' },
  { value: 5, label: '5 条样本', detail: '默认评估' },
  { value: 10, label: '10 条样本', detail: '更稳定' },
  { value: 0, label: '全部样本', detail: '完整运行' },
]

const kbSelectOptions = computed(() =>
  kbOptions.value.map(kb => ({
    value: kb.id,
    label: kb.name,
    detail: `${kb.chunk_count || kb.doc_count || 0} chunks`,
  }))
)

const modelSelectOptions = computed(() =>
  modelOptions.value.map(model => ({
    value: model.id,
    label: model.display_name,
    detail: model.model_name || model.provider,
  }))
)

const bestHit = computed(() => maxMetric(metricHit))
const bestMrr = computed(() => maxMetric(metricMrr))
const bestCoverage = computed(() => maxMetric(metricCoverage))
const bestP50 = computed(() => minMetric(metricP50))
const bestLatency = computed(() => minMetric(metricLatency))
const bestTokens = computed(() => minMetric(metricTokens))

function metricHit(row) {
  return Number(row.metrics?.retrieval?.hit_at_3 || 0)
}

function metricMrr(row) {
  return Number(row.metrics?.retrieval?.mrr || 0)
}

function metricCoverage(row) {
  return Number(row.metrics?.answer?.coverage || 0)
}

function metricP50(row) {
  return Number(row.metrics?.system?.first_token_latency_ms || 0)
}

function metricLatency(row) {
  return Number(row.latency_ms || row.metrics?.system?.total_latency_ms || 0)
}

function metricTokens(row) {
  return Number(row.metrics?.system?.total_tokens || 0)
}

function maxMetric(getter) {
  if (!results.value.length) return null
  return Math.max(...results.value.map(getter))
}

function minMetric(getter) {
  const values = results.value.map(getter).filter(value => value > 0)
  return values.length ? Math.min(...values) : null
}

function isBest(value, best) {
  return best !== null && Math.abs(Number(value) - Number(best)) < 0.000001
}

function formatMode(mode) {
  const value = String(mode || '')
  return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase()
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function formatLatency(value) {
  const ms = Number(value || 0)
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  return `${Math.round(ms)} ms`
}

function formatNumber(value) {
  return Math.round(Number(value || 0)).toLocaleString()
}

function formatCost(value) {
  return `$${Number(value || 0).toFixed(4)}`
}

function clampPercent(value) {
  return Math.max(0, Math.min(Number(value || 0), 100))
}

function chartRows(type) {
  const maxLatency = Math.max(...results.value.map(metricLatency), 1)
  return results.value.map(row => {
    if (type === 'latency') {
      const latency = metricLatency(row)
      return {
        label: formatMode(row.mode),
        percent: latency / maxLatency * 100,
        value: formatLatency(latency),
      }
    }
    const getter = type === 'coverage' ? metricCoverage : metricHit
    const value = getter(row)
    return {
      label: formatMode(row.mode),
      percent: value * 100,
      value: formatPercent(value),
    }
  })
}

async function loadKnowledgeBases() {
  loadingKbs.value = true
  try {
    const data = await listKnowledgeBases()
    kbOptions.value = data || []
    selectedKbId.value =
      kbOptions.value.find(kb => Number(kb.chunk_count || 0) > 0)?.id ||
      kbOptions.value[0]?.id ||
      ''
  } catch (e) {
    console.warn('failed to load knowledge bases for evaluation', e)
    kbOptions.value = []
    selectedKbId.value = ''
  } finally {
    loadingKbs.value = false
  }
}

async function loadModels() {
  loadingModels.value = true
  try {
    const data = await listChatModels()
    modelOptions.value = (data.models || []).filter(model => model.enabled)
    if (!modelOptions.value.length) {
      modelOptions.value = [{ id: 'mock-chat', display_name: 'Mock Chat', model_name: 'mock-chat', enabled: true }]
    }
    selectedModelId.value =
      modelOptions.value.find(model => model.id === data.default_model_id)?.id ||
      modelOptions.value[0]?.id ||
      data.default_model_id ||
      'mock-chat'
  } catch (e) {
    console.warn('failed to load models for evaluation', e)
    modelOptions.value = [{ id: 'mock-chat', display_name: 'Mock Chat', model_name: 'mock-chat', enabled: true }]
    selectedModelId.value = 'mock-chat'
  } finally {
    loadingModels.value = false
  }
}

async function runEval() {
  running.value = true
  try {
    const { run_id } = await runEvaluation(
      selectedKbId.value,
      ['naive', 'advanced', 'modular', 'graph', 'agentic'],
      selectedModelId.value || null,
      selectedSampleLimit.value > 0 ? selectedSampleLimit.value : null,
    )
    results.value = await getEvaluationResult(run_id)
  } catch (e) {
    alert('评测失败: ' + (e?.message || e))
  } finally {
    running.value = false
  }
}

onMounted(() => {
  loadKnowledgeBases()
  loadModels()
})
</script>

<style scoped>
.eval-page {
  height: 100vh;
  overflow: auto;
  padding: 28px;
  background: #f8fafc;
  color: #0f172a;
}

.filter-card,
.metrics-card,
.visual-card,
.empty-card {
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  background: #fff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
}

.filter-card {
  width: min(1280px, 100%);
  margin: 0 auto;
  padding: 20px;
}

.filter-heading,
.card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.eyebrow {
  margin: 0 0 4px;
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

h2,
h3 {
  margin: 0;
  color: #0f172a;
}

h2 {
  font-size: 24px;
  letter-spacing: 0;
}

h3 {
  font-size: 16px;
  letter-spacing: 0;
}

.filter-controls {
  display: flex;
  align-items: end;
  gap: 14px;
  margin-top: 18px;
  flex-wrap: wrap;
}

.control-group {
  display: grid;
  flex: 1 1 240px;
  min-width: 220px;
  gap: 7px;
}

.control-group:first-child {
  min-width: 260px;
}

.control-group.compact {
  flex: 0 0 170px;
  min-width: 150px;
  width: 170px;
}

.control-label {
  color: #475569;
  font-size: 12px;
  font-weight: 700;
}

.run-button {
  display: inline-flex;
  min-width: 116px;
  height: 42px;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 0;
  border-radius: 11px;
  background: #0f172a;
  color: #fff;
  padding: 0 16px;
  font-weight: 700;
  cursor: pointer;
  transition: background .18s ease, transform .18s ease, box-shadow .18s ease;
}

.run-button:hover:not(:disabled) {
  background: #1e293b;
  box-shadow: 0 8px 18px rgba(15, 23, 42, .16);
  transform: translateY(-1px);
}

.run-button:disabled {
  cursor: not-allowed;
  opacity: .65;
}

.button-icon {
  display: block;
  width: 15px !important;
  height: 15px !important;
  flex: 0 0 15px;
  fill: currentColor;
}

.spinner {
  width: 15px;
  height: 15px;
  border: 2px solid rgba(255, 255, 255, .35);
  border-top-color: #fff;
  border-radius: 999px;
  animation: spin .8s linear infinite;
}

.metrics-card {
  width: min(1280px, 100%);
  margin-top: 18px;
  margin-right: auto;
  margin-left: auto;
  padding: 18px;
}

.summary-pill,
.metric-value.best {
  border-radius: 999px;
  background: #dcfce7;
  color: #166534;
  font-size: 12px;
  font-weight: 800;
}

.summary-pill {
  padding: 6px 10px;
}

.table-wrap {
  margin-top: 16px;
  overflow-x: auto;
  border: 1px solid #e2e8f0;
  border-radius: 13px;
}

.metrics-table {
  width: 100%;
  min-width: 900px;
  border-collapse: collapse;
}

.metrics-table th {
  background: #f8fafc;
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.metrics-table th,
.metrics-table td {
  border-bottom: 1px solid #e2e8f0;
  padding: 13px 16px;
  text-align: left;
  white-space: nowrap;
}

.metrics-table tr:last-child td {
  border-bottom: 0;
}

.metrics-table .numeric {
  text-align: right;
}

.mode-name {
  font-weight: 800;
}

.metric-value {
  display: inline-flex;
  justify-content: flex-end;
  min-width: 58px;
  padding: 4px 8px;
  border-radius: 999px;
}

.visual-grid {
  display: grid;
  width: min(1280px, 100%);
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
  margin-right: auto;
  margin-left: auto;
}

.visual-card {
  min-width: 0;
  padding: 18px;
}

.visual-heading {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 10px;
  margin-bottom: 16px;
}

.visual-icon {
  display: inline-grid;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  place-items: center;
  border-radius: 8px;
  background: #f8fafc;
  color: #475569;
  font-size: 11px;
  font-weight: 850;
}

.chart-rows {
  display: grid;
  gap: 13px;
}

.chart-row {
  display: grid;
  grid-template-columns: 78px minmax(96px, 1fr) 78px;
  align-items: center;
  gap: 10px;
}

.chart-label {
  overflow: hidden;
  color: #334155;
  font-size: 13px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chart-track {
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: #f1f5f9;
}

.chart-fill {
  height: 100%;
  border-radius: 999px;
  transition: width .45s ease, filter .18s ease;
}

.chart-row:hover .chart-fill {
  filter: brightness(.96);
}

.chart-value {
  color: #334155;
  font-size: 13px;
  font-weight: 800;
  text-align: right;
}

.tone-indigo .chart-fill {
  background: linear-gradient(90deg, #818cf8, #4f46e5);
}

.tone-emerald .chart-fill {
  background: linear-gradient(90deg, #6ee7b7, #10b981);
}

.tone-sky .chart-fill {
  background: linear-gradient(90deg, #7dd3fc, #0ea5e9);
}

.empty-card {
  display: grid;
  width: min(1280px, 100%);
  min-height: 340px;
  margin-top: 18px;
  margin-right: auto;
  margin-left: auto;
  place-items: center;
  padding: 40px 20px;
  text-align: center;
}

.empty-icon {
  display: grid;
  width: 54px;
  height: 54px;
  place-items: center;
  border-radius: 16px;
  background: #eef2ff;
  color: #4f46e5;
}

.empty-icon svg {
  display: block;
  width: 26px !important;
  height: 26px !important;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 2;
}

.empty-card h3 {
  margin-top: 16px;
  font-size: 18px;
}

.empty-card p {
  margin: 8px 0 0;
  color: #64748b;
  font-size: 14px;
}

.skeleton-table {
  display: grid;
  gap: 10px;
  margin-top: 18px;
}

.skeleton-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1.4fr;
  gap: 14px;
  border-bottom: 1px solid #f1f5f9;
  padding: 12px 0;
}

.skeleton-cell {
  height: 18px;
  border-radius: 999px;
  background: linear-gradient(90deg, #f1f5f9, #e2e8f0, #f1f5f9);
  background-size: 200% 100%;
  animation: shimmer 1.1s ease-in-out infinite;
}

.skeleton-cell.short {
  width: 72px;
}

.skeleton-cell.wide {
  width: 100%;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes shimmer {
  to { background-position-x: -200%; }
}

@media (max-width: 1180px) {
  .visual-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .eval-page {
    padding: 18px;
  }

  .filter-heading,
  .card-title-row {
    align-items: stretch;
    flex-direction: column;
  }

  .run-button {
    width: 100%;
  }

  .control-group,
  .control-group.compact {
    min-width: 0;
    width: 100%;
  }

  .visual-grid {
    grid-template-columns: 1fr;
  }
}
</style>
