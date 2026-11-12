<script setup>
import { onMounted, ref } from 'vue'
import { getApiKey, setApiKey } from './api'

// API Key 只在后端启用了鉴权时才需要，存 localStorage 方便刷新后保留
const key = ref('')

onMounted(() => {
  key.value = getApiKey()
})

function saveKey() {
  setApiKey(key.value.trim())
}
</script>

<template>
  <header class="app-header">
    <div class="brand">
      <span class="brand-mark">🧬</span>
      <div>
        Bio Knowledge Retrieval
        <small>生物知识检索系统</small>
      </div>
    </div>

    <nav class="nav">
      <RouterLink to="/ingest">文档入库</RouterLink>
      <RouterLink to="/chat">知识问答</RouterLink>
    </nav>

    <div class="header-right">
      <span>API Key</span>
      <input
        v-model="key"
        class="key-input"
        type="password"
        placeholder="后端未启用鉴权可留空"
        @change="saveKey"
      />
    </div>
  </header>

  <RouterView />
</template>