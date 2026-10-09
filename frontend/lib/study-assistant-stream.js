import { API_BASE_URL, createRequestLifecycle, waitForRequest } from './api/core'
import { assertAuthContext, captureAuthContext, clearAuthSession, getAccessToken, refreshAccessToken } from './auth-client'

const RUN_ID_HEADER = 'X-RAG-Run-ID'

/**
 * 解析响应文本，保留字符串错误以兼容后端非 JSON 响应。
 * @param {string} raw 原始响应文本
 * @returns {object|string|null} 解析后的响应体
 */
function parseJsonBody(raw) {
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

/**
 * 解析单个 SSE 帧中的 data 载荷。
 * @param {string} frame SSE 原始帧
 * @returns {object|null} 事件载荷
 */
function parseSseFrame(frame) {
  const payload = frame
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n')
    .trim()
  if (!payload) return null
  try {
    return JSON.parse(payload)
  } catch {
    const error = new Error('流式响应不是合法 JSON')
    error.terminal = true
    throw error
  }
}

/**
 * 把非成功 HTTP 响应转换为保留 run 信息的错误。
 * @param {Response} response Fetch 响应
 * @returns {Promise<Error>} 可供 workspace 恢复 run 的错误
 */
async function buildResponseError(response) {
  const body = parseJsonBody(await response.text())
  const detail = body?.detail
  const message = typeof detail === 'object'
    ? detail.detail || `请求失败：${response.status}`
    : detail || body || `请求失败：${response.status}`
  const error = new Error(message)
  error.status = response.status
  error.body = body
  error.run_id = body?.run_id || detail?.run_id || null
  error.run_status = body?.run_status || detail?.run_status || null
  return error
}

/**
 * 打开带认证的 SSE 请求，认证刷新时复用原幂等键和请求体。
 * @param {object} options 请求路径、方法、载荷、幂等键和取消信号
 * @returns {Promise<Response>} 已验证成功的流响应
 */
async function fetchEventStream({
  path,
  method = 'GET',
  body = null,
  idempotencyKey = '',
  signal,
  retryAuth = true,
  authContext,
}) {
  assertAuthContext(authContext)
  signal?.throwIfAborted()
  const headers = new Headers({ Accept: 'text/event-stream' })
  if (body !== null) headers.set('Content-Type', 'application/json')
  if (idempotencyKey) headers.set('Idempotency-Key', idempotencyKey)
  const accessToken = getAccessToken()
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    signal,
    ...(body !== null ? { body: JSON.stringify(body) } : {}),
  })
  assertAuthContext(authContext)
  signal?.throwIfAborted()
  if (response.status === 401 && retryAuth) {
    await waitForRequest(refreshAccessToken({ expectedContext: authContext }), signal)
    assertAuthContext(authContext)
    signal.throwIfAborted()
    return fetchEventStream({ path, method, body, idempotencyKey, signal, authContext, retryAuth: false })
  }
  if (!response.ok) {
    const error = await buildResponseError(response)
    assertAuthContext(authContext)
    if (response.status === 401) clearAuthSession({ expectedContext: authContext, reason: error })
    throw error
  }
  return response
}

/**
 * 将缓存拆成完整 SSE 帧和待续传尾部。
 * @param {string} buffer 当前缓存
 * @param {boolean} final 是否已经读到响应结尾
 * @returns {{frames: string[], remainder: string}} 拆分结果
 */
function splitSseFrames(buffer, final) {
  const frames = []
  let remainder = buffer
  while (true) {
    const boundary = remainder.indexOf('\n\n')
    if (boundary === -1) break
    frames.push(remainder.slice(0, boundary))
    remainder = remainder.slice(boundary + 2)
  }
  if (final && remainder.trim()) {
    frames.push(remainder)
    remainder = ''
  }
  return { frames, remainder }
}

/**
 * 构造终态事件错误，保留取消和 run 状态供上层区分。
 * @param {object} payload durable run 终态载荷
 * @returns {Error} 带终态标记的错误
 */
function buildTerminalError(payload) {
  const runStatus = payload?.run_status || ''
  const error = new Error(payload?.detail || (runStatus === 'cancelled' ? '研究运行已取消' : '研究失败'))
  error.terminal = true
  error.cancelled = runStatus === 'cancelled'
  error.run_id = payload?.run_id || null
  error.run_status = runStatus || null
  return error
}

/**
 * 分发单个持久化事件并返回是否已完成。
 * @param {object} payload SSE 载荷
 * @param {object} callbacks token、事件、完成和 sequence 回调
 * @returns {boolean} 是否收到成功 done 终态
 */
function dispatchPayload(payload, callbacks) {
  const sequence = Number(payload?.sequence)
  if (Number.isInteger(sequence) && sequence > 0) callbacks.onSequence(sequence)
  if (payload?.type === 'token') {
    callbacks.onToken(payload.content || '')
    return false
  }
  if (payload?.type === 'done') {
    callbacks.onDone(payload)
    return true
  }
  if (payload?.type === 'error' || payload?.type === 'terminal') {
    throw buildTerminalError(payload)
  }
  callbacks.onEvent(payload)
  return false
}

/**
 * 消费一次 SSE 订阅，断流时抛错交由 durable reconnect 续订。
 * @param {Response} response SSE 响应
 * @param {object} callbacks 事件回调
 * @returns {Promise<object>} 最终 done 载荷
 */
async function consumeResearchStream(response, callbacks) {
  const reader = response.body?.getReader()
  if (!reader) {
    const error = new Error('当前环境不支持流式读取')
    error.terminal = true
    throw error
  }
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await waitForRequest(reader.read(), callbacks.signal)
    assertAuthContext(callbacks.authContext)
    callbacks.signal.throwIfAborted()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n')
    const split = splitSseFrames(buffer, done)
    buffer = split.remainder
    for (const frame of split.frames) {
      const payload = parseSseFrame(frame)
      if (!payload) continue
      if (dispatchPayload(payload, callbacks)) {
        await reader.cancel()
        return payload
      }
    }
    if (done) break
  }
  throw new Error('流式响应提前结束')
}

/**
 * 等待下一次连接，并在用户停止时立即结束等待。
 * @param {number} delayMs 重连等待毫秒数
 * @param {AbortSignal|undefined} signal 取消信号
 * @returns {Promise<void>} 等待结果
 */
function waitBeforeRetry(delayMs, signal) {
  if (signal?.aborted) return Promise.reject(signal.reason || new Error('研究运行已取消'))
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, delayMs)
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal.reason || new Error('研究运行已取消'))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * 判断连接错误是否允许复用当前 durable run 继续订阅。
 * @param {Error} error 当前错误
 * @param {AbortSignal|undefined} signal 取消信号
 * @param {number} attempt 当前重试次数
 * @param {number} maxRetries 最大重试次数
 * @returns {boolean} 是否继续订阅
 */
function shouldRetry(error, signal, attempt, maxRetries) {
  if (signal?.aborted || error?.terminal || attempt >= maxRetries) return false
  return !error?.status || error.status >= 500
}

/**
 * 执行一次 durable stream：首次 POST，获得 run 后只按 sequence GET 重连。
 * @param {object} options 请求和事件回调
 * @returns {Promise<object>} 最终 done 载荷
 */
async function runDurableRequest(options) {
  let runId = ''
  let afterSequence = 0
  let lastError = null
  for (let attempt = 0; attempt <= options.maxRetries; attempt += 1) {
    try {
      const path = runId
        ? `/api/v1/rag-runs/${encodeURIComponent(runId)}/events/stream?after_sequence=${afterSequence}`
        : options.path
      const response = await fetchEventStream({
        path,
        method: runId ? 'GET' : 'POST',
        body: runId ? null : options.body,
        idempotencyKey: runId ? '' : options.idempotencyKey,
        signal: options.signal,
        authContext: options.authContext,
      })
      if (!runId) {
        runId = response.headers.get(RUN_ID_HEADER) || ''
        if (!runId) throw buildTerminalError({ detail: '流式响应缺少 durable run 标识' })
        options.onRun(runId)
      }
      return await consumeResearchStream(response, {
        ...options,
        onSequence: (sequence) => { afterSequence = Math.max(afterSequence, sequence) },
      })
    } catch (error) {
      lastError = error
      if (!shouldRetry(error, options.signal, attempt, options.maxRetries)) throw error
      await waitBeforeRetry(1000 * (attempt + 1), options.signal)
    }
  }
  throw lastError || new Error('流式连接失败')
}

/** 【研究】【会话隔离】@param {object} options 请求和回调 @returns {Promise<object>} 当前会话的研究结果 */
async function runDurableStream(options) {
  const authContext = captureAuthContext()
  const lifecycle = createRequestLifecycle({ externalSignal: options.signal, timeoutMs: null, context: authContext })
  try {
    return await waitForRequest(runDurableRequest({ ...options, authContext, signal: lifecycle.signal }), lifecycle.signal)
  } finally {
    lifecycle.dispose()
  }
}

/**
 * 启动深度研究 durable stream。
 * @param {object} options 深度研究请求与回调
 * @returns {Promise<object>} 最终研究载荷
 */
export function runResearchStream({
  query,
  sessionId,
  conversationId,
  projectId,
  paperIds = [],
  researchMode,
  intent = 'deep_research',
  idempotencyKey,
  onRun = () => {},
  onToken = () => {},
  onDone = () => {},
  onEvent = () => {},
  maxRetries = 3,
  signal,
}) {
  return runDurableStream({
    path: '/api/v1/research/stream',
    body: {
      query,
      session_id: sessionId || null,
      conversation_id: conversationId || null,
      project_id: projectId || null,
      paper_ids: Array.isArray(paperIds) ? paperIds : [],
      research_mode: researchMode || null,
      intent: intent || null,
    },
    idempotencyKey,
    onRun,
    onToken,
    onDone,
    onEvent,
    maxRetries,
    signal,
  })
}

/**
 * 启动跨论文问答 durable stream。
 * @param {object} options 跨论文问答请求与回调
 * @returns {Promise<object>} 最终回答载荷
 */
export function runCrossPaperStream({
  query,
  paperIds = [],
  sessionId,
  conversationId,
  projectId,
  chatHistory = [],
  dialogueMode,
  idempotencyKey,
  onRun = () => {},
  onToken = () => {},
  onDone = () => {},
  onEvent = () => {},
  maxRetries = 3,
  signal,
}) {
  return runDurableStream({
    path: '/api/v1/qa/cross-paper/stream',
    body: {
      query,
      paper_ids: Array.isArray(paperIds) ? paperIds : [],
      chat_history: Array.isArray(chatHistory) ? chatHistory : [],
      session_id: sessionId || null,
      conversation_id: conversationId || null,
      project_id: projectId || null,
      dialogue_mode: dialogueMode || null,
    },
    idempotencyKey,
    onRun,
    onToken,
    onDone,
    onEvent,
    maxRetries,
    signal,
  })
}
