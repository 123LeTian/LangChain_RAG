/**
 * Vue Router 配置 — 五个页面的路由映射
 */
import { createRouter, createWebHashHistory } from 'vue-router'

// 视图懒加载（按需分割打包）
const ChatView = () => import('../views/ChatView.vue')
const KnowledgeBaseView = () => import('../views/KnowledgeBaseView.vue')
const TraceView = () => import('../views/TraceView.vue')
const GraphView = () => import('../views/GraphView.vue')
const EvaluationView = () => import('../views/EvaluationView.vue')

const router = createRouter({
  // 使用 Hash 模式避免部署时后端需配置 fallback
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'Chat', component: ChatView, meta: { title: '对话' } },
    { path: '/knowledge', name: 'Knowledge', component: KnowledgeBaseView, meta: { title: '知识库' } },
    { path: '/trace/:traceId?', name: 'Trace', component: TraceView, meta: { title: '追踪' } },
    { path: '/graph', name: 'Graph', component: GraphView, meta: { title: '图谱' } },
    { path: '/evaluation', name: 'Evaluation', component: EvaluationView, meta: { title: '评测' } },
  ],
})

export default router
