<template>
  <!--
    根组件 — 顶部导航 + 路由视图
    A 同学负责的前端入口
  -->
  <div id="app-root">
    <!-- 顶部导航栏 -->
    <nav class="app-nav">
      <div class="nav-brand">🧪 LangChain RAG</div>
      <div class="nav-links">
        <router-link to="/chat" active-class="active">💬 对话</router-link>
        <router-link to="/knowledge" active-class="active">📚 知识库</router-link>
        <router-link to="/graph" active-class="active">🕸️ 图谱</router-link>
        <router-link to="/trace" active-class="active">🔍 追踪</router-link>
        <router-link to="/evaluation" active-class="active">📊 评测</router-link>
      </div>
      <div class="nav-status">
        <span class="status-dot" :class="healthStatus" />
        {{ healthText }}
      </div>
    </nav>

    <!-- 路由视图 -->
    <router-view />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { healthCheck } from './api/client'

// ===== 健康检查 — 验证前后端连通性 =====
const healthStatus = ref('checking')
const healthText = ref('连接中...')

onMounted(async () => {
  try {
    const res = await healthCheck()
    healthStatus.value = 'ok'
    healthText.value = `API v${res.version}`
  } catch {
    healthStatus.value = 'error'
    healthText.value = 'API 离线'
  }
})
</script>

<style scoped>
#app-root { height: 100vh; display: flex; flex-direction: column; }

/* ===== 导航栏 ===== */
.app-nav {
  display: flex; align-items: center; padding: 0 24px;
  height: 56px; background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  gap: 32px; flex-shrink: 0;
}
.nav-brand { font-weight: 700; font-size: 18px; color: var(--accent); }
.nav-links { display: flex; gap: 8px; }
.nav-links a {
  text-decoration: none; color: var(--text-secondary); padding: 6px 14px;
  border-radius: 6px; font-size: 14px; transition: background .2s;
}
.nav-links a:hover { background: var(--bg-secondary); }
.nav-links a.active { background: var(--accent); color: white; }

.nav-status { margin-left: auto; font-size: 13px; display: flex; align-items: center; gap: 6px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.checking { background: #ffc107; }
.status-dot.ok { background: #28a745; }
.status-dot.error { background: #dc3545; }
</style>
