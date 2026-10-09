import {
  AuthSessionChangedError,
  assertAuthContext,
  captureAuthContext,
  clearAuthSession,
  getAccessToken,
  isCurrentAuthContext,
  refreshAccessToken,
  subscribeAuthSession,
} from '../auth-client'

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000'
// 【请求】【等待时限】沿用三分钟总时限，覆盖认证续期和一次重放
const DEFAULT_TIMEOUT_MS = 180_000
const MILLISECONDS_PER_SECOND = 1000

/** @param {string} raw HTTP 正文 @returns {unknown} 解析后的正文或原始错误文本 */
function parseJsonBody(raw) {
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}

/** @param {Response} response HTTP 响应 @returns {Promise<unknown>} 已读取的正文 */
async function requestBody(response) {
  return parseJsonBody(await response.text())
}

/** @param {object} initHeaders 显式请求头 @param {boolean} skipAuth 是否跳过认证 @returns {Headers} 请求头 */
function buildHeaders(initHeaders = {}, skipAuth = false) {
  const headers = new Headers(initHeaders)
  if (!skipAuth) {
    const token = getAccessToken()
    if (token && !headers.has('Authorization')) headers.set('Authorization', 'Bearer ' + token)
  }
  return headers
}

/** @param {object} response 含 HTTP 状态的响应 @param {unknown} body 响应正文 @returns {Error} 保留状态的错误 */
function buildRequestError(response, body) {
  const detail = body?.detail
  const message = detail !== null && typeof detail === 'object'
    ? detail.detail || '请求失败：' + response.status
    : detail || body || '请求失败：' + response.status
  const error = new Error(message)
  error.status = response.status
  error.body = body
  error.run_id = body?.run_id || detail?.run_id || null
  error.run_status = body?.run_status || detail?.run_status || null
  return error
}

/**
 * 【请求】【会话隔离】把超时、外部取消和账号切换绑定到当前请求
 * @param {{externalSignal?: AbortSignal, timeoutMs: number, context: object|null}} options 生命周期参数
 * @returns {object} 请求信号、超时状态及清理入口
 */
export function createRequestLifecycle({ externalSignal, timeoutMs, context }) {
  const controller = new AbortController()
  let timedOut = false
  const forwardAbort = () => controller.abort(externalSignal.reason)
  if (externalSignal?.aborted) controller.abort(externalSignal.reason)
  else externalSignal?.addEventListener('abort', forwardAbort, { once: true })
  const unsubscribe = context ? subscribeAuthSession((_session, reason) => {
    if (!isCurrentAuthContext(context)) controller.abort(reason || new AuthSessionChangedError())
  }) : () => {}
  const timer = timeoutMs == null ? null : setTimeout(() => {
    timedOut = true
    controller.abort(new DOMException('Request timed out', 'AbortError'))
  }, timeoutMs)
  return {
    signal: controller.signal,
    get timedOut() { return timedOut },
    dispose() {
      clearTimeout(timer)
      externalSignal?.removeEventListener('abort', forwardAbort)
      unsubscribe()
    },
  }
}

/**
 * 【请求】【取消等待】在等共享刷新时也响应取消，后台完成后不能重放已取消请求
 * @param {Promise<unknown>} pending 当前请求执行
 * @param {AbortSignal} signal 本次请求的取消信号
 * @returns {Promise<unknown>} 完成结果或明确取消原因
 */
export function waitForRequest(pending, signal) {
  return new Promise((resolve, reject) => {
    const detach = () => signal.removeEventListener('abort', onAbort)
    const onAbort = () => {
      detach()
      reject(signal.reason)
    }
    signal.addEventListener('abort', onAbort, { once: true })
    pending.then(
      (body) => {
        detach()
        resolve(body)
      },
      (error) => {
        detach()
        reject(error)
      },
    )
    if (signal.aborted) onAbort()
  })
}

/**
 * 【请求】【身份绑定】响应和认证重试沿用发起请求的账号上下文
 * @param {object} options 请求选项及固定认证上下文
 * @param {AbortSignal} signal 本次请求信号
 * @returns {Promise<unknown>} 当前会话的响应
 */
async function performRequest(options, signal) {
  const { path, init, skipAuth, retryAuth, authContext, responseType } = options
  signal.throwIfAborted()
  if (!skipAuth) assertAuthContext(authContext)
  const response = await fetch(API_BASE_URL + path, {
    ...init,
    headers: buildHeaders(init.headers || {}, skipAuth),
    signal,
  })
  const body = response.ok && responseType === 'blob' ? await response.blob() : await requestBody(response)
  signal.throwIfAborted()
  if (!skipAuth) assertAuthContext(authContext)
  if (response.status === 401 && !skipAuth && retryAuth) {
    await refreshAccessToken({ expectedContext: authContext })
    signal.throwIfAborted()
    assertAuthContext(authContext)
    return performRequest({ ...options, retryAuth: false }, signal)
  }
  if (!response.ok) {
    const error = buildRequestError(response, body)
    if (response.status === 401 && !skipAuth) clearAuthSession({ expectedContext: authContext, reason: error })
    throw error
  }
  return body
}

/**
 * 【请求】【JSON 请求】执行有时限且绑定当前会话的 JSON 请求
 * @param {object} options 路径、请求体、超时和认证选项
 * @returns {Promise<unknown>} 响应；账号变化时明确拒绝旧请求结果
 */
export async function requestJson({
  path, init = {}, timeoutMs = DEFAULT_TIMEOUT_MS, skipAuth = false, retryAuth = true, authContext, responseType = 'json',
}) {
  const context = skipAuth ? null : authContext || captureAuthContext()
  if (context) assertAuthContext(context)
  const lifecycle = createRequestLifecycle({ externalSignal: init.signal, timeoutMs, context })
  try {
    const pending = performRequest({ path, init, skipAuth, retryAuth, authContext: context, responseType }, lifecycle.signal)
    const body = await waitForRequest(pending, lifecycle.signal)
    if (context) assertAuthContext(context)
    return body
  } catch (error) {
    if (context && !isCurrentAuthContext(context) && lifecycle.signal.reason !== error) {
      throw new AuthSessionChangedError()
    }
    if (lifecycle.timedOut) {
      throw new Error('请求超时（' + Math.round(timeoutMs / MILLISECONDS_PER_SECOND) + 's），请稍后重试')
    }
    throw error
  } finally {
    lifecycle.dispose()
  }
}

/** @param {object} options 认证文件请求 @returns {Promise<Blob>} 同一会话的已授权文件 */
export const requestBlob = (options) => requestJson({ ...options, responseType: 'blob' })

/** @param {File} file 上传原文件 @param {object} fields 额外表单字段 @returns {FormData} 上传表单 */
function createUploadForm(file, fields) {
  const form = new FormData()
  form.append('file', file)
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') form.append(key, value)
  })
  return form
}

/**
 * 【上传】【响应归属】解析 XHR 结果前核对原账号，401 不能清理另一个账号
 * @param {XMLHttpRequest} xhr 完成的请求
 * @param {{context: object|null, finish: Function}} options 原上下文和完成回调
 * @returns {void} 完成或拒绝本次请求
 */
function finishUploadResponse(xhr, { context, finish }) {
  try {
    if (context) assertAuthContext(context)
    const body = parseJsonBody(xhr.responseText)
    if (xhr.status >= 200 && xhr.status < 300) {
      finish(null, body)
      return
    }
    const error = buildRequestError(xhr, body)
    if (xhr.status === 401 && context) clearAuthSession({ expectedContext: context, reason: error })
    finish(error)
  } catch (error) {
    finish(error)
  }
}

/**
 * 【上传】【传输进度】保留原上传协议，账号切换取消本地监听
 * @param {object} options 接口、文件、附加字段及进度回调
 * @returns {Promise<unknown>} 同一会话收到的受理结果，不自动重发文件
 */
export function uploadWithProgress({ path, file, onProgress, fields = {}, skipAuth = false, authContext, signal }) {
  const context = skipAuth ? null : authContext || captureAuthContext()
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let unsubscribe = () => {}
    let settled = false
    const onAbort = () => {
      finish(signal.reason || new DOMException('上传已取消', 'AbortError'))
      xhr.abort()
    }
    const finish = (error, body) => {
      if (settled) return
      settled = true
      unsubscribe()
      signal?.removeEventListener('abort', onAbort)
      xhr.upload.onprogress = null
      xhr.onload = xhr.onerror = xhr.onabort = xhr.ontimeout = null
      if (error) reject(error)
      else resolve(body)
    }
    xhr.open('POST', API_BASE_URL + path, true)
    xhr.timeout = DEFAULT_TIMEOUT_MS
    xhr.ontimeout = () => finish(new Error('上传响应超时，请查回原上传状态'))
    if (!skipAuth && getAccessToken()) xhr.setRequestHeader('Authorization', 'Bearer ' + getAccessToken())
    xhr.upload.onprogress = (event) => {
      if ((context && !isCurrentAuthContext(context)) || !event.lengthComputable || !onProgress) return
      const percent = Math.min(100, Math.round(event.loaded / event.total * 100))
      onProgress({ percent, loaded: event.loaded, total: event.total })
    }
    xhr.onload = () => {
      unsubscribe()
      finishUploadResponse(xhr, { context, finish })
    }
    xhr.onerror = () => finish(context && !isCurrentAuthContext(context)
      ? new AuthSessionChangedError() : new Error('网络错误或上传失败'))
    xhr.onabort = () => finish(new DOMException('上传已取消', 'AbortError'))
    if (context) unsubscribe = subscribeAuthSession((_session, reason) => {
      if (isCurrentAuthContext(context)) return
      finish(reason || new AuthSessionChangedError())
      xhr.abort()
    })
    try {
      if (context) assertAuthContext(context)
      signal?.throwIfAborted()
      signal?.addEventListener('abort', onAbort, { once: true })
      xhr.send(createUploadForm(file, fields))
    } catch (error) {
      finish(error)
    }
  })
}
