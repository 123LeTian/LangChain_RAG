import { createRouter, createWebHashHistory } from 'vue-router'

const ChatView = () => import('../views/ChatView.vue')
const KnowledgeBaseView = () => import('../views/KnowledgeBaseView.vue')
const TraceView = () => import('../views/TraceView.vue')
const GraphView = () => import('../views/GraphView.vue')
const EvaluationView = () => import('../views/EvaluationView.vue')

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'Chat', component: ChatView, meta: { title: '问答实验台' } },
    { path: '/knowledge', name: 'Knowledge', component: KnowledgeBaseView, meta: { title: '知识库管理' } },
    { path: '/graph', name: 'Graph', component: GraphView, meta: { title: '知识图谱' } },
    { path: '/evaluation', name: 'Evaluation', component: EvaluationView, meta: { title: '模式评测' } },
    { path: '/trace/:traceId?', name: 'Trace', component: TraceView, meta: { title: '执行追踪' } },
  ],
})

export default router
