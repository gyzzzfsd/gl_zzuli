/**
 * 后端接口封装。
 *
 * 约定：
 * - 请求统一走 /api 前缀，开发环境由 vite proxy 转发到后端，生产环境由 Nginx 转发。
 * - 如果后端配置了 API_KEY，普通请求用 X-API-Key 头；SSE 用 ?api_key= 查询参数
 *   （EventSource 不支持自定义请求头）。
 */

const KEY_STORAGE = 'bio_api_key'
const SESSION_STORAGE = 'bio_session_id'

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || ''
}

export function setApiKey(value) {
  if (value) localStorage.setItem(KEY_STORAGE, value)
  else localStorage.removeItem(KEY_STORAGE)
}

function authHeaders() {
  const key = getApiKey()
  return key ? { 'X-API-Key': key } : {}
}

async function parseError(res) {
  try {
    const data = await res.json()
    return data.detail || JSON.stringify(data)
  } catch {
    return `HTTP ${res.status}`
  }
}

async function request(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { ...(options.headers || {}), ...authHeaders() }
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

/** 上传文档，返回 { task_id }。onProgress 拿到 0-100 的百分比。 */
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/import/upload')
    const key = getApiKey()
    if (key) xhr.setRequestHeader('X-API-Key', key)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch (err) {
          reject(new Error('响应不是合法 JSON'))
        }
      } else {
        let msg = `HTTP ${xhr.status}`
        try {
          msg = JSON.parse(xhr.responseText).detail || msg
        } catch { /* 保留默认错误信息 */ }
        reject(new Error(msg))
      }
    }
    xhr.onerror = () => reject(new Error('上传失败：网络连接中断'))
    xhr.send(form)
  })
}

/** 查询导入任务状态：{ status, done_list, running_list } */
export function getStatus(taskId) {
  return request(`/api/import/status/${encodeURIComponent(taskId)}`)
}

/** 已入库的知识主体列表：{ items: [{item_name, file_title, chunks}] } */
export function getDocuments() {
  return request('/api/import/documents')
}

/** 新建会话：{ session_id } */
export function newSession() {
  return request('/api/session/new')
}

export function getHistory(sessionId) {
  return request(`/api/history/${encodeURIComponent(sessionId)}`)
}

export function clearHistory(sessionId) {
  return request(`/api/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

/**
 * 流式提问。
 *
 * 流程：POST /api/query(is_stream=true) 拿 task_id → 连 SSE 收增量。
 *
 * @returns {{taskId: string, sessionId: string, close: Function}}
 */
export async function askStream({ question, sessionId, onProgress, onDelta, onFinal, onError }) {
  const data = await request('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: question, session_id: sessionId, is_stream: true })
  })

  const key = getApiKey()
  const url =
    `/api/query/stream/${encodeURIComponent(data.task_id)}` +
    (key ? `?api_key=${encodeURIComponent(key)}` : '')

  const es = new EventSource(url)
  let finished = false

  const close = () => {
    finished = true
    es.close()
  }

  es.addEventListener('progress', (e) => {
    try {
      onProgress?.(JSON.parse(e.data))
    } catch { /* 进度事件解析失败不影响主流程 */ }
  })

  es.addEventListener('delta', (e) => {
    try {
      onDelta?.(JSON.parse(e.data).delta || '')
    } catch { /* 单条增量失败可忽略 */ }
  })

  es.addEventListener('final', (e) => {
    if (finished) return
    try {
      onFinal?.(JSON.parse(e.data).answer || '')
    } finally {
      close()
    }
  })

  es.onerror = () => {
    if (finished) return
    close()
    onError?.(new Error('流式连接中断，请重试'))
  }

  return { taskId: data.task_id, sessionId: data.session_id, close }
}

export function getSessionId() {
  return localStorage.getItem(SESSION_STORAGE) || ''
}

export function setSessionId(id) {
  if (id) localStorage.setItem(SESSION_STORAGE, id)
  else localStorage.removeItem(SESSION_STORAGE)
}