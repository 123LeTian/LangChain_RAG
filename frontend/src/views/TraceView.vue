<template>
  <!-- Trace 追踪页面 — 输入 trace_id 查看完整执行轨迹 -->
  <div class="trace-page">
    <h2>🔍 执行追踪</h2>

    <div class="trace-search">
      <input v-model="traceId" placeholder="输入 Trace ID（如 mock_trace_1）" @keydown.enter="loadTrace" />
      <button @click="loadTrace" class="btn-primary" :disabled="!traceId.trim()">查询</button>
    </div>

    <div v-if="error" class="error-msg">{{ error }}</div>

    <div v-if="events.length > 0" class="trace-timeline">
      <div v-for="(e, i) in events" :key="i" :class="['timeline-item', e.stage]">
        <div class="timeline-marker" />
        <div class="timeline-content">
          <div class="stage-header">
            <span class="stage-name">{{ e.stage }}</span>
            <span class="stage-time">{{ e.duration_ms }}ms</span>
          </div>
          <div class="stage-detail">
            <div v-if="e.input_summary"><strong>输入：</strong>{{ e.input_summary }}</div>
            <div v-if="e.output_summary"><strong>输出：</strong>{{ e.output_summary }}</div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="!events.length && !error" class="empty-state">
      <p>输入 Trace ID 查看完整执行轨迹</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { getTrace } from '../api/client'

const traceId = ref('')
const events = ref([])
const error = ref('')

async function loadTrace() {
  error.value = ''
  events.value = []
  try {
    events.value = await getTrace(traceId.value.trim())
  } catch (e) {
    error.value = e.message
  }
}
</script>

<style scoped>
.trace-page { padding: 24px; max-width: 800px; }
.trace-search { display: flex; gap: 12px; margin: 16px 0; }
.trace-search input { flex: 1; padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; }
.error-msg { color: var(--danger); padding: 12px; background: #f8d7da; border-radius: 6px; margin-bottom: 12px; }

.trace-timeline { position: relative; padding-left: 32px; }
.trace-timeline::before {
  content: ''; position: absolute; left: 8px; top: 0; bottom: 0;
  width: 2px; background: var(--border-color);
}
.timeline-item { position: relative; margin-bottom: 20px; }
.timeline-marker {
  position: absolute; left: -24px; top: 6px; width: 14px; height: 14px;
  border-radius: 50%; background: var(--accent); border: 2px solid white;
}
.timeline-item.error .timeline-marker { background: var(--danger); }
.timeline-item.complete .timeline-marker { background: var(--success); }

.stage-header { display: flex; justify-content: space-between; align-items: center; }
.stage-name { font-weight: 600; text-transform: uppercase; font-size: 13px; color: var(--accent); }
.stage-time { font-size: 12px; color: var(--text-muted); }
.stage-detail { margin-top: 4px; font-size: 13px; color: var(--text-secondary); }

.empty-state { text-align: center; margin-top: 80px; color: var(--text-muted); }
</style>
