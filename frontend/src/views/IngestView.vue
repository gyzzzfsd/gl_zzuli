<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getDocuments, getStatus, uploadFile } from '../api'

// 导入流水线的节点顺序（与后端 task_util 的中文名保持一致），用于渲染进度条
const PIPELINE = [
  '上传文件',
  '检查文件',
  'Markdown图片处理',
  '文档切分',
  '主体名称识别',
  '向量生成',
  '导入向量数据库',
  '导入知识图谱'
]

const file = ref(null)
const dragging = ref(false)
const inputEl = ref(null)
const uploadPercent = ref(0)
const uploading = ref(false)
const task = ref(null)          // { status, done_list, running_list, error }
const taskId = ref('')
const docs = ref([])
const loadingDocs = ref(false)
const message = ref('')
const error = ref('')

let timer = null

const doneList = computed(() => task.value?.done_list || [])
const status = computed(() => task.value?.status || '')
const finished = computed(() => status.value === 'completed' || status.value === 'failed')

// 上传阶段算 10%，流水线节点平均分剩下的 90%
const percent = computed(() => {
  if (!taskId.value) return uploadPercent.value * 0.1
  if (status.value === 'completed') return 100
  const done = doneList.value.length
  return Math.min(99, 10 + Math.round((done / PIPELINE.length) * 90))
})

function isDone(name) {
  return doneList.value.includes(name)
}

function isActive(name) {
  if (finished.value) return false
  if (isDone(name)) return false
  // 第一个未完成的节点视为"进行中"
  const idx = PIPELINE.indexOf(name)
  return PIPELINE.slice(0, idx).every(isDone)
}

function pickFile(f) {
  error.value = ''
  message.value = ''
  if (!f) return
  const ok = /\.(pdf|md)$/i.test(f.name)
  if (!ok) {
    error.value = '只支持 .pdf 或 .md 文件'
    return
  }
  file.value = f
}

function onFileChange(e) {
  pickFile(e.target.files?.[0])
  e.target.value = ''   // 允许重复选择同一个文件
}

function onDrop(e) {
  dragging.value = false
  pickFile(e.dataTransfer?.files?.[0])
}

async function startUpload() {
  if (!file.value || uploading.value) return
  uploading.value = true
  uploadPercent.value = 0
  error.value = ''
  message.value = ''
  task.value = null
  taskId.value = ''

  try {
    const res = await uploadFile(file.value, (p) => (uploadPercent.value = p))
    taskId.value = res.task_id
    await pollOnce()
    startPolling()
  } catch (e) {
    error.value = e.message || '上传失败'
  } finally {
    uploading.value = false
  }
}

async function pollOnce() {
  if (!taskId.value) return
  try {
    task.value = await getStatus(taskId.value)
    if (task.value.status === 'completed') {
      stopPolling()
      message.value = '入库完成'
      file.value = null
      await loadDocs()
    } else if (task.value.status === 'failed') {
      stopPolling()
      error.value = task.value.error || '导入失败，请查看后端日志'
    }
  } catch (e) {
    // 轮询失败不立刻中止，可能只是瞬时网络问题；连续失败由全局错误兜底
    error.value = e.message || '查询任务状态失败'
  }
}

function startPolling() {
  stopPolling()
  timer = setInterval(pollOnce, 2000)
}

function stopPolling() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

async function loadDocs() {
  loadingDocs.value = true
  try {
    const res = await getDocuments()
    docs.value = res.items || []
  } catch (e) {
    error.value = e.message || '读取知识库列表失败'
  } finally {
    loadingDocs.value = false
  }
}

onMounted(loadDocs)
onBeforeUnmount(stopPolling)
</script>

<template>
  <main class="page">
    <h1 class="page-title">文档入库</h1>
    <p class="page-sub">
      上传生物医学文献（PDF / Markdown），系统会自动解析、按标题切片、识别知识主体，
      并写入向量库与知识图谱。
    </p>

    <!-- 上传 -->
    <section class="card">
      <h2 class="card-title">上传文档</h2>
      <p class="card-desc">
        PDF 走 MinerU 解析（较慢，分钟级）；Markdown 直接进入切分流程（秒级）。
        单文件上限 50 MB。
      </p>

      <div
        class="dropzone"
        :class="{ dragging }"
        @click="inputEl?.click()"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
      >
        <div class="big">📄</div>
        <div>拖拽文件到这里，或点击选择</div>
        <div class="hint">支持 .pdf / .md</div>
        <div v-if="file" class="file-chip">
          <span>📎 {{ file.name }}</span>
          <span>{{ (file.size / 1024).toFixed(1) }} KB</span>
        </div>
      </div>
      <input ref="inputEl" type="file" accept=".pdf,.md" hidden @change="onFileChange" />

      <div class="row" style="margin-top: 16px">
        <button class="btn btn-primary" :disabled="!file || uploading" @click="startUpload">
          {{ uploading ? '上传中…' : '开始入库' }}
        </button>
        <button class="btn btn-ghost" :disabled="uploading" @click="file = null">清空</button>
        <div class="spacer" />
        <span v-if="taskId" class="tag">task {{ taskId.slice(0, 8) }}</span>
      </div>

      <!-- 进度 -->
      <div v-if="taskId">
        <div class="bar"><i :style="{ width: percent + '%' }" /></div>
        <div class="steps">
          <span
            v-for="name in PIPELINE"
            :key="name"
            class="step"
            :class="{ done: isDone(name), active: isActive(name) }"
          >{{ name }}</span>
        </div>
      </div>

      <div v-if="error" class="alert alert-error">{{ error }}</div>
      <div v-else-if="message" class="alert alert-ok">{{ message }}</div>
    </section>

    <!-- 知识库现状 -->
    <section class="card">
      <div class="row">
        <h2 class="card-title" style="margin: 0">已入库的知识主体</h2>
        <div class="spacer" />
        <button class="btn btn-ghost" :disabled="loadingDocs" @click="loadDocs">
          {{ loadingDocs ? '刷新中…' : '刷新' }}
        </button>
      </div>
      <p class="card-desc">按知识主体聚合，chunk 数表示该主体在向量库中的切片数量。</p>

      <div v-if="!docs.length" class="empty">知识库还是空的，先上传一份文档吧。</div>
      <table v-else>
        <thead>
          <tr>
            <th style="width: 45%">知识主体</th>
            <th>来源文档</th>
            <th style="width: 110px">切片数</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(d, i) in docs" :key="i">
            <td><strong>{{ d.item_name }}</strong></td>
            <td class="muted">{{ d.file_title || '—' }}</td>
            <td><span class="tag">{{ d.chunks }}</span></td>
          </tr>
        </tbody>
      </table>
    </section>
  </main>
</template>