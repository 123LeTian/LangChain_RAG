/**
 * Vue 应用入口 — 挂载路由与全局样式
 */
import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'

createApp(App).use(router).mount('#app')
