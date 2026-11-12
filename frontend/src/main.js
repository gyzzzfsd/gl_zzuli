import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import IngestView from './views/IngestView.vue'
import ChatView from './views/ChatView.vue'
import './styles.css'

// 两个"端"用路由区分：入库端 / 问答端
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/ingest' },
    { path: '/ingest', name: 'ingest', component: IngestView },
    { path: '/chat', name: 'chat', component: ChatView }
  ]
})

createApp(App).use(router).mount('#app')