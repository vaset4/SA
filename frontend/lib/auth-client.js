import { StorageService } from './storage-service'
import { buildWorkspaceRedirectTarget } from './workspace-routes'
import { normalizeAuthSession, normalizeRefreshSession } from './auth-session.mjs'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000'
const EMPTY_SESSION = { accessToken: '', refreshToken: '', tokenType: 'bearer', expiresIn: 0, user: null }
const listeners = new Set()
let sessionCache = { ...EMPTY_SESSION }
let hydrated = false
let sessionVersion = 0
let refreshRequest = null

/** 【认证】【会话隔离】标识因会话切换而取消的旧操作 */
export class AuthSessionChangedError extends Error {
  /** @returns {AuthSessionChangedError} 不携带凭证的取消原因 */
  constructor() {
    super('账号已切换，本次操作已取消')
    this.name = 'AuthSessionChangedError'
    this.code = 'AUTH_SESSION_CHANGED'
  }
}

/** 【认证】【会话读取】首次读取持久会话；无参数、无返回值 */
function ensureHydrated() {
  if (hydrated) return
  hydrated = true
  sessionCache = normalizeAuthSession(StorageService.getAuthSession())
}

/** 【认证】【会话读取】返回独立快照；无参数，返回当前会话 */
export function getAuthSession() {
  ensureHydrated()
  return { ...sessionCache, user: sessionCache.user ? { ...sessionCache.user } : null }
}

/** 【认证】【会话隔离】捕获当前身份边界；返回不包含 token 的上下文 */
export function captureAuthContext() {
  const session = getAuthSession()
  return Object.freeze({ version: sessionVersion, userId: session.user?.user_id || '' })
}

/** @param {object} context 请求开始时的会话上下文 @returns {boolean} 是否仍属于同一会话 */
export function isCurrentAuthContext(context) {
  const current = captureAuthContext()
  return Boolean(context && context.version === current.version && context.userId === current.userId)
}

/** @param {object} context 原会话上下文 @returns {void} 失效时明确抛出取消错误 */
export function assertAuthContext(context) {
  if (!isCurrentAuthContext(context)) throw new AuthSessionChangedError()
}

/** 【认证】【会话通知】@param {Error|undefined} reason 失效原因 @returns {void} 通知订阅者 */
function emit(reason) {
  const snapshot = getAuthSession()
  listeners.forEach((listener) => {
    try {
      listener(snapshot, reason)
    } catch (error) {
      console.error('【认证】【会话通知】监听器执行失败:', error)
    }
  })
}

/** 【认证】【会话保存】保存当前会话或移除凭证；无参数、无返回值 */
function persistSession() {
  if (sessionCache.accessToken || sessionCache.refreshToken) {
    StorageService.saveAuthSession(sessionCache)
    return
  }
  StorageService.clearAuthSession()
}

/** @returns {string} 当前 access token，仅用于请求头 */
export function getAccessToken() {
  return getAuthSession().accessToken
}

/** @returns {boolean} 当前浏览器是否持有 access token */
export function isAuthenticated() {
  return Boolean(getAccessToken())
}

/**
 * 【认证】【会话提交】登录产生新边界，刷新和资料更新保持同一边界
 * @param {object} nextSession 已取得的完整会话
 * @param {{expectedContext?: object, preserveContext?: boolean}} options 原上下文及是否保留边界
 * @returns {object} 已发布的当前会话
 */
export function setAuthSession(nextSession, { expectedContext, preserveContext = false } = {}) {
  ensureHydrated()
  if (expectedContext) assertAuthContext(expectedContext)
  const next = normalizeAuthSession(nextSession)
  const userId = String(next.user?.user_id || '').trim()
  if (!next.accessToken || !next.refreshToken || !userId) {
    const error = new Error('认证响应缺少完整会话或用户身份')
    error.code = 'AUTH_SESSION_INVALID'
    throw error
  }
  if (preserveContext && (!expectedContext || userId !== expectedContext.userId)) {
    const error = new Error('服务端身份与当前账号不一致，请重新登录')
    error.code = 'AUTH_SESSION_INVALID'
    throw error
  }
  // 1. 【认证】【会话切换】新登录清除上一会话的轻量选中项，令旧请求上下文失效
  if (!preserveContext) {
    sessionVersion += 1
    StorageService.saveActiveConversationId('')
  }
  sessionCache = { ...next, user: { ...next.user, user_id: userId } }
  persistSession()
  emit()
  return getAuthSession()
}

/**
 * 【认证】【会话退出】只允许当前上下文清理会话
 * @param {{redirect?: boolean, expectedContext?: object, reason?: Error}} options 跳转策略、原上下文及失效原因
 * @returns {void} 本地清理完成
 */
export function clearAuthSession({ redirect = true, expectedContext, reason } = {}) {
  ensureHydrated()
  if (expectedContext) assertAuthContext(expectedContext)
  sessionVersion += 1
  sessionCache = { ...EMPTY_SESSION }
  StorageService.saveActiveConversationId('')
  persistSession()
  emit(reason)
  if (redirect && typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
    const target = buildWorkspaceRedirectTarget(window.location.pathname, window.location.search)
    window.location.replace('/login?redirect=' + encodeURIComponent(target))
  }
}

/** @param {Function} listener 会话监听器 @returns {Function} 取消订阅函数 */
export function subscribeAuthSession(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** @param {string} path 接口路径 @param {object} init 请求参数 @returns {Promise<unknown>} 响应正文 */
async function fetchJson(path, init = {}) {
  const response = await fetch(API_BASE_URL + path, init)
  const raw = await response.text()
  let body = null
  try {
    body = raw ? JSON.parse(raw) : null
  } catch {
    body = raw
  }
  if (!response.ok) {
    const detail = body?.detail
    const message = typeof detail === 'string' ? detail : '请求失败：' + response.status
    const error = new Error(message)
    error.status = response.status
    throw error
  }
  return body
}

/**
 * 【认证】【令牌刷新】刷新结果只能提交回发起刷新的同一会话
 * @param {object} context 原会话上下文
 * @param {string} refreshToken 原会话 refresh token
 * @returns {Promise<string>} 同一会话的新 access token
 */
async function performRefresh(context, refreshToken) {
  let responseReceived = false
  try {
    const payload = await fetchJson('/api/v1/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
    responseReceived = true
    assertAuthContext(context)
    setAuthSession(normalizeRefreshSession(payload), { expectedContext: context, preserveContext: true })
    return getAccessToken()
  } catch (error) {
    if (!isCurrentAuthContext(context)) throw new AuthSessionChangedError()
    // 1. 【认证】【令牌刷新】无效凭证或不完整成功响应使当前会话失效；网络故障保留原错误
    if (error.status === 401 || responseReceived) clearAuthSession({ expectedContext: context, reason: error })
    throw error
  }
}

/**
 * 【认证】【令牌刷新】只合并同一身份边界内的并发刷新
 * @param {{expectedContext?: object}} options 请求原上下文
 * @returns {Promise<string>} 当前上下文的新 access token
 */
export async function refreshAccessToken({ expectedContext } = {}) {
  const context = expectedContext || captureAuthContext()
  assertAuthContext(context)
  if (refreshRequest?.context.version === context.version) return refreshRequest.promise
  const session = getAuthSession()
  if (!session.refreshToken) {
    const error = new Error('缺少 refresh token，请重新登录')
    error.status = 401
    clearAuthSession({ expectedContext: context, reason: error })
    throw error
  }
  const request = { context, promise: performRefresh(context, session.refreshToken) }
  refreshRequest = request
  try {
    return await request.promise
  } finally {
    // 1. 【认证】【令牌刷新】旧刷新结束不能清除新会话正在执行的刷新
    if (refreshRequest === request) refreshRequest = null
  }
}
