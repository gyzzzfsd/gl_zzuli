<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import {
  askStream,
  clearHistory,
  getDocuments,
  getHistory,
  getSessionId,
  newSession,
  setSessionId
} from '../api'

const messages = ref([])      // [{ role: 'user'|'assistant', content, streaming }]
const input = ref('')
const busy = ref(false)
const progress = ref('')
const error = ref('')
const sessionId = ref('')
const scrollEl = ref(null)
const kbEmpty = ref(false)
let streamHandle = null

const canSend = computed(() => input.value.trim().length > 0 && !busy.value)

/**
 * 把答案拆成正文 + 图片 URL 列表。
 * 后端约定图片放在末尾的【图片】区块里，每行一个 <url>。
 */
function splitImages(text) {
  const raw = text || ''
  const idx = raw.indexOf('【图片】')
  if (idx < 0) return { body: raw, images: [] }

  const body = raw.slice(0, idx).trimEnd()
  const images = raw
    .slice(idx + '【图片】'.length)
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^<|>$/g, ''))
    .filter((u) => /^https?:\/\//i.test(u))

  return { body, images }
}

async function scrollToBottom() {
  await nextTick()
  if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight
}

async function ensureSession() {
  let sid = getSessionId()
  if (!sid) {
    const res = await newSession()
    sid = res.session_id
    setSessionId(sid)
  }
  sessionId.value = sid
}

async function loadHistory() {
  if (!sessionId.value) return
  try {
    const res = await getHistory(sessionId.value)
    messages.value = (res.items || []).map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.text || '',
      streaming: false
    }))
    await scrollToBottom()
  } catch {
    // 历史读不到不影响新提问，静默忽略
  }
}

async function checkKb() {
  try {
    const res = await getDocuments()
    kbEmpty.value = !(res.items || []).length
  } catch {
    kbEmpty.value = false
  }
}

async function send() {
  const question = input.value.trim()
  if (!question || busy.value) return

  error.value = ''
  busy.value = true
  progress.value = '正在检索…'
  input.value = ''

  messages.value.push({ role: 'user', content: question, streaming: false })
  // 用 reactive 包一层：流式增量是直接改这个对象的 content，
  // 若只 push 普通对象，改动不会触发视图更新（改的是原始对象，绕过了代理）。
  const assistant = reactive({ role: 'assistant', content: '', streaming: true })
  messages.value.push(assistant)
  await scrollToBottom()

  try {
    streamHandle = await askStream({
      question,
      sessionId: sessionId.value,
      onProgress: (p) => {
        const running = p.running_list || []
        const done = p.done_list || []
        if (running.length) progress.value = `正在执行：${running.join('、')}`
        else if (done.length) progress.value = `已完成 ${done.length} 个步骤`
        else progress.value = '正在多路检索（向量 / HyDE / 图谱 / 联网）…'
      },
      onDelta: (text) => {
        assistant.content += text
        scrollToBottom()
      },
      onFinal: (answer) => {
        // 以后端 FINAL 为准：它包含清洗掉伪造图片后的最终版本
        assistant.content = answer || assistant.content
        assistant.streaming = false
        busy.value = false
        progress.value = ''
        scrollToBottom()
      },
      onError: (e) => {
        error.value = e.message || '流式连接失败'
        assistant.streaming = false
        busy.value = false
        progress.value = ''
      }
    })
  } catch (e) {
    error.value = e.message || '提问失败'
    assistant.streaming = false
    busy.value = false
    progress.value = ''
  }
}

function onKeydown(e) {
  // Enter 发送，Shift+Enter 换行
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    send()
  }
}

async function resetSession() {
  if (busy.value) return
  try {
    if (sessionId.value) await clearHistory(sessionId.value)
  } catch {
    // 清空失败也继续换新会话
  }
  const res = await newSession()
  setSessionId(res.session_id)
  sessionId.value = res.session_id
  messages.value = []
  error.value = ''
}

onMounted(async () => {
  await ensureSession()
  await loadHistory()
  await checkKb()
})

onBeforeUnmount(() => {
  streamHandle?.close?.()
})
</script>

<template>
  <div class="chat-shell">
    <div ref="scrollEl" class="chat-scroll">
      <div class="chat-inner">
        <!-- 空态 -->
        <div v-if="!messages.length" class="card">
          <h2 class="card-title">知识问答</h2>
          <p class="card-desc">
            基于已入库的生物医学文献回答。系统会先识别问题里的知识主体（基因 / 物种 / 通路 / 方法等），
            再走向量检索、HyDE、知识图谱与联网检索多路召回，最后经重排后生成答案。
          </p>
          <div v-if="kbEmpty" class="alert alert-error">
            知识库还是空的，请先到「文档入库」上传一份文献。
          </div>
          <div v-else class="row">
            <button class="btn" @click="input = '便携式血糖仪 G1如何使用？'">
              试试：便携式血糖仪 G1如何使用？
            </button>
            <button class="btn" @click="input = '指夹式脉搏血氧仪 O1读数偏低怎么回事？'">
              试试：指夹式脉搏血氧仪 O1读数偏低怎么回事？
            </button>
          </div>
        </div>

        <!-- 消息 -->
        <div
          v-for="(msg, i) in messages"
          :key="i"
          class="msg"
          :class="msg.role"
        >
          <div class="avatar">{{ msg.role === 'user' ? '用户' : 'AI' }}</div>
          <div class="bubble">
            <div class="content" :class="{ typing: msg.streaming && !msg.content }">
              {{ splitImages(msg.content).body }}
            </div>

            <div v-if="splitImages(msg.content).images.length" class="images">
              <img
                v-for="(url, k) in splitImages(msg.content).images"
                :key="k"
                :src="url"
                alt="参考图"
                loading="lazy"
              />
            </div>

            <div v-if="msg.role === 'assistant' && !msg.streaming" class="meta">
              {{ msg.content ? '已完成' : '（无回答）' }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="composer">
      <div v-if="progress" class="progress-line">
        <span class="dot" />
        <span>{{ progress }}</span>
      </div>
      <div v-if="error" class="progress-line" style="color: var(--danger)">{{ error }}</div>

      <div class="composer-inner">
        <textarea
          v-model="input"
          rows="1"
          placeholder="问点什么…（Enter 发送，Shift+Enter 换行）"
          @keydown="onKeydown"
        />
        <button class="btn btn-primary" :disabled="!canSend" @click="send">
          {{ busy ? '检索中…' : '发送' }}
        </button>
        <button class="btn btn-ghost" :disabled="busy" @click="resetSession">新会话</button>
      </div>
    </div>
  </div>
</template>