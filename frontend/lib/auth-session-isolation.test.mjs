/** 账号切换、请求取消及 store 晚到响应回归；作者：xxx */
import assert from 'node:assert/strict'
import test from 'node:test'
import { deferred, jsonResponse, loadAuthRequestModules, loadStore, sessionPayload } from './auth-test-support.mjs'

/** 无参数，返回当前事件轮次内已就绪 Promise 的排空信号 */
const flush = () => new Promise(setImmediate)

for (const failure of [false, true]) {
  test('旧刷新' + (failure ? '失败' : '成功') + '都不能覆盖新账号或清理其刷新任务', async () => {
    const first = deferred()
    const second = deferred()
    const calls = []
    const { authClient } = await loadAuthRequestModules((_url, init) => {
      calls.push(JSON.parse(init.body).refresh_token)
      return calls.length === 1 ? first.promise : second.promise
    })
    authClient.setAuthSession(sessionPayload('a'))
    const oldResult = authClient.refreshAccessToken().catch((error) => error)
    authClient.setAuthSession(sessionPayload('b'))
    const newResult = authClient.refreshAccessToken()
    if (failure) first.reject(new Error('old refresh failed'))
    else first.resolve(jsonResponse(sessionPayload('a', 'refreshed')))
    assert.equal((await oldResult).code, 'AUTH_SESSION_CHANGED')
    assert.equal(authClient.getAuthSession().user.user_id, 'b')
    const joined = authClient.refreshAccessToken()
    assert.equal(calls.length, 2)
    second.resolve(jsonResponse(sessionPayload('b', 'refreshed')))
    await Promise.all([newResult, joined])
    assert.equal(authClient.getAccessToken(), 'b-refreshed-access')
  })
}

test('临时刷新网络错误保留原凭证和真实错误', async () => {
  const failure = new TypeError('connection unavailable')
  const { authClient } = await loadAuthRequestModules(async () => { throw failure })
  authClient.setAuthSession(sessionPayload('a'))
  await assert.rejects(authClient.refreshAccessToken(), (error) => error === failure)
  assert.equal(authClient.getAccessToken(), 'a-initial-access')
})

test('刷新返回不完整会话时明确失败并清理原凭证', async () => {
  const { authClient } = await loadAuthRequestModules(async () => jsonResponse({ access_token: 'incomplete' }))
  authClient.setAuthSession(sessionPayload('a'))
  await assert.rejects(authClient.refreshAccessToken(), /缺少 token/)
  assert.equal(authClient.getAccessToken(), '')
})

for (const status of [200, 401]) {
  test('切换账号后拒绝旧 HTTP ' + status + '，不能用新 token 重放', async () => {
    const response = deferred()
    const calls = []
    const { authClient, apiCore } = await loadAuthRequestModules((url, init) => {
      calls.push({ url, token: new Headers(init.headers).get('Authorization') })
      return response.promise
    })
    authClient.setAuthSession(sessionPayload('a'))
    const result = apiCore.requestJson({ path: '/old-request' }).catch((error) => error)
    authClient.setAuthSession(sessionPayload('b'))
    response.resolve(jsonResponse({ detail: 'old response' }, status))
    assert.equal((await result).code, 'AUTH_SESSION_CHANGED')
    await flush()
    assert.equal(authClient.getAccessToken(), 'b-initial-access')
    assert.deepEqual(calls.map(({ token }) => token), ['Bearer a-initial-access'])
  })
}

test('读取响应正文期间切换账号也必须拒绝旧结果', async () => {
  const body = deferred()
  const reading = deferred()
  const { authClient, apiCore } = await loadAuthRequestModules(async () => ({
    ok: true, status: 200, text: () => { reading.resolve(); return body.promise },
  }))
  authClient.setAuthSession(sessionPayload('a'))
  const result = apiCore.requestJson({ path: '/body' }).catch((error) => error)
  await reading.promise
  authClient.setAuthSession(sessionPayload('b'))
  body.resolve('{"owner":"a"}')
  assert.equal((await result).code, 'AUTH_SESSION_CHANGED')
  assert.equal(authClient.getAuthSession().user.user_id, 'b')
})

for (const cancellation of ['timeout', 'abort']) {
  test('等待刷新时' + cancellation + '能结束等待，刷新晚到不再重发', async () => {
    const refresh = deferred()
    const started = deferred()
    const controller = new AbortController()
    let timeout
    let resourceCalls = 0
    const modules = await loadAuthRequestModules((url) => {
      if (url.endsWith('/auth/refresh')) { started.resolve(); return refresh.promise }
      resourceCalls += 1
      return Promise.resolve(jsonResponse({ detail: 'expired' }, 401))
    }, { globals: { setTimeout: (callback) => { timeout = callback; return 1 }, clearTimeout: () => undefined } })
    modules.authClient.setAuthSession(sessionPayload('a'))
    const result = modules.apiCore.requestJson({ path: '/protected', init: { signal: controller.signal } }).catch((error) => error)
    await started.promise
    const joined = modules.authClient.refreshAccessToken()
    if (cancellation === 'timeout') timeout()
    else controller.abort()
    const error = await result
    if (cancellation === 'timeout') assert.match(error.message, /请求超时/)
    else assert.equal(error.name, 'AbortError')
    refresh.resolve(jsonResponse(sessionPayload('a', 'refreshed')))
    await joined
    await flush()
    assert.equal(resourceCalls, 1)
  })
}

test('同账号重新登录也使原操作上下文失效', async () => {
  const { authClient, storageState } = await loadAuthRequestModules(async () => jsonResponse({}))
  authClient.setAuthSession(sessionPayload('a'))
  const context = authClient.captureAuthContext()
  storageState.activeId = 'a-conversation'
  authClient.setAuthSession(sessionPayload('a', 'new-login'))
  assert.equal(authClient.isCurrentAuthContext(context), false)
  assert.equal(storageState.activeId, '')
  assert.throws(() => authClient.clearAuthSession({ expectedContext: context }), { code: 'AUTH_SESSION_CHANGED' })
  assert.equal(authClient.getAccessToken(), 'a-new-login-access')
})

test('错误正文 detail=null 时仍保留真实 HTTP 错误', async () => {
  const { apiCore } = await loadAuthRequestModules(async () => jsonResponse({ detail: null }, 503))
  await assert.rejects(apiCore.requestJson({ path: '/error', skipAuth: true }), (error) => error.status === 503)
})

/** 模拟 XHR 网络事件端口，回调执行的仍是真实上传封装 */
class UploadTransport {
  upload = {}
  headers = {}
  /** 无参数，记录已创建的网络实例 */
  constructor() { UploadTransport.current = this }
  /** @param {string} method HTTP 方法 @param {string} url 路径 @returns {void} 记录目标 */
  open(method, url) { this.method = method; this.url = url }
  /** @param {string} name 请求头 @param {string} value 值 @returns {void} 记录请求头 */
  setRequestHeader(name, value) { this.headers[name] = value }
  /** @param {FormData} form 原文件表单 @returns {void} 记录传输开始 */
  send(form) { this.form = form }
  /** 无参数，标记传输已取消 */
  abort() { this.aborted = true; this.onabort?.() }
}

test('切换账号终止旧上传监听，晚到进度和 401 不影响新账号', async () => {
  const { authClient, apiCore } = await loadAuthRequestModules(async () => jsonResponse({}), {
    globals: { XMLHttpRequest: UploadTransport },
  })
  authClient.setAuthSession(sessionPayload('a'))
  const progress = []
  const result = apiCore.uploadWithProgress({ path: '/upload', file: new Blob(['pdf']), onProgress: (value) => progress.push(value) }).catch((error) => error)
  const xhr = UploadTransport.current
  const lateLoad = xhr.onload
  const lateProgress = xhr.upload.onprogress
  authClient.setAuthSession(sessionPayload('b'))
  xhr.status = 401
  xhr.responseText = '{"detail":"expired"}'
  lateLoad()
  lateProgress({ lengthComputable: true, loaded: 1, total: 1 })
  assert.equal((await result).code, 'AUTH_SESSION_CHANGED')
  assert.equal(xhr.aborted, true)
  assert.equal(xhr.headers.Authorization, 'Bearer a-initial-access')
  assert.equal(progress.length, 0)
  assert.equal(authClient.getAccessToken(), 'b-initial-access')
})

/** @param {object} modules 请求模块 @returns {Promise<object>} 真实认证 store */
const authStore = (modules) => loadStore(modules, { path: '../stores/auth-store.js', component: 'AuthStoreProvider' })

test('恢复 A 用户的响应晚到，不能覆盖 B 的资料与页面状态', async (t) => {
  const response = deferred()
  const modules = await loadAuthRequestModules(() => response.promise)
  modules.authClient.setAuthSession(sessionPayload('a'))
  const store = await authStore(modules)
  t.after(() => store.unmount())
  store.render()
  store.runEffects()
  modules.authClient.setAuthSession(sessionPayload('b'))
  response.resolve(jsonResponse({ user: { user_id: 'a' } }))
  await flush()
  assert.equal(modules.authClient.getAuthSession().user.user_id, 'b')
  assert.equal(store.state.session.user.user_id, 'b')
  assert.equal(store.state.error, '')
  assert.equal(store.state.ready, true)
})

test('旧登录响应不能替换期间已登录的账号', async (t) => {
  const response = deferred()
  const modules = await loadAuthRequestModules(() => response.promise)
  const store = await authStore(modules)
  t.after(() => store.unmount())
  const actions = store.render().props.value
  store.runEffects()
  const result = actions.login({ username: 'a', password: 'unit-password' }).catch((error) => error)
  modules.authClient.setAuthSession(sessionPayload('b'))
  response.resolve(jsonResponse(sessionPayload('a')))
  assert.equal((await result).code, 'AUTH_SESSION_CHANGED')
  assert.equal(store.state.session.user.user_id, 'b')
  assert.equal(store.state.busy, false)
  assert.equal(store.state.error, '')
})

for (const status of [200, 503]) {
  test('旧 logout 的 HTTP ' + status + '不能清理新账号', async (t) => {
    const response = deferred()
    const modules = await loadAuthRequestModules((url) => url.endsWith('/auth/logout')
      ? response.promise : Promise.resolve(jsonResponse({ user: { user_id: 'a' } })))
    modules.authClient.setAuthSession(sessionPayload('a'))
    const store = await authStore(modules)
    t.after(() => store.unmount())
    const actions = store.render().props.value
    store.runEffects()
    await flush()
    const result = actions.logout({ redirect: false })
    modules.authClient.setAuthSession(sessionPayload('b'))
    response.resolve(jsonResponse({ revoked: true }, status))
    await result
    await flush()
    assert.equal(modules.authClient.getAccessToken(), 'b-initial-access')
    assert.equal(store.state.session.user.user_id, 'b')
    assert.equal(store.state.error, '')
  })
}

test('远端 logout 失败清理本机会话并保留未确认吊销提示', async (t) => {
  const modules = await loadAuthRequestModules((url) => Promise.resolve(url.endsWith('/auth/logout')
    ? jsonResponse({ detail: 'server unavailable' }, 503) : jsonResponse({ user: { user_id: 'a' } })))
  modules.authClient.setAuthSession(sessionPayload('a'))
  const store = await authStore(modules)
  t.after(() => store.unmount())
  const actions = store.render().props.value
  store.runEffects()
  await flush()
  await actions.logout({ redirect: false })
  assert.equal(modules.authClient.getAccessToken(), '')
  assert.equal(store.state.busy, false)
  assert.match(store.state.error, /远端会话未确认吊销/)
})

test('正常刷新保持页面版本，账号切换才重建子树', async (t) => {
  const modules = await loadAuthRequestModules(async (url) => jsonResponse(url.endsWith('/auth/refresh')
    ? sessionPayload('a', 'refreshed') : { user: { user_id: 'a' } }))
  modules.authClient.setAuthSession(sessionPayload('a'))
  const store = await authStore(modules)
  t.after(() => store.unmount())
  const initialKey = store.render().props.children.key
  store.runEffects()
  await flush()
  modules.storageState.activeId = 'a-conversation'
  await modules.authClient.refreshAccessToken()
  assert.equal(store.render().props.children.key, initialKey)
  assert.equal(modules.storageState.activeId, 'a-conversation')
  modules.authClient.setAuthSession(sessionPayload('b'))
  assert.notEqual(store.render().props.children.key, initialKey)
  assert.equal(modules.storageState.activeId, '')
})

test('恢复用户遇到网络故障时不伪装成凭证失效', async (t) => {
  const modules = await loadAuthRequestModules(async () => { throw new TypeError('network unavailable') })
  modules.authClient.setAuthSession(sessionPayload('a'))
  const store = await authStore(modules)
  t.after(() => store.unmount())
  store.render()
  store.runEffects()
  await flush()
  assert.equal(modules.authClient.getAccessToken(), 'a-initial-access')
  assert.match(store.state.error, /network unavailable/)
})

/** @param {object} modules 请求模块 @param {object} endpoints 可控对话接口 @returns {Promise<object>} 真实对话 store */
function conversationStore(modules, endpoints) {
  return loadStore(modules, {
    path: '../stores/conversation-store.js', component: 'ConversationStoreProvider',
    dependencies: {
      './auth-store': { useAuthStore: () => ({ ready: true, isAuthenticated: true, sessionContext: modules.authClient.captureAuthContext() }) },
      '../lib/api/conversations': {
        listConversations: async () => ({ conversations: [] }), getConversation: async () => ({}),
        createConversationRecord: endpoints.createConversationRecord, deleteConversationRecord: async () => ({}), updateConversationRecord: async () => ({}),
      },
      '../lib/api/research': { createSession: endpoints.createSession },
      '../utils/message-utils': { sameMessages: () => false },
      '../lib/study-assistant-workspace.mjs': { pickMessageExtras: () => ({}) },
    },
  })
}

for (const hydration of [false, true]) {
  test((hydration ? '默认对话恢复' : '新建对话') + '中途切换账号，不把 A 研究会话关联给 B', async (t) => {
    const session = deferred()
    const started = deferred()
    let createCalls = 0
    const modules = await loadAuthRequestModules(async () => jsonResponse({}))
    modules.authClient.setAuthSession(sessionPayload('a'))
    const store = await conversationStore(modules, {
      createSession: () => { started.resolve(); return session.promise },
      createConversationRecord: async () => { createCalls += 1; return { id: 'new-conversation' } },
    })
    t.after(() => store.unmount())
    const actions = store.render().props.value
    const result = hydration ? null : actions.createNew().catch((error) => error)
    if (hydration) store.runEffects()
    await started.promise
    modules.authClient.setAuthSession(sessionPayload('b'))
    session.resolve({ session_id: 'a-research-session' })
    if (result) assert.equal((await result).code, 'AUTH_SESSION_CHANGED')
    await flush()
    assert.equal(createCalls, 0)
    assert.equal(store.state.conversations.length, 0)
    assert.equal(modules.storageState.activeId, '')
  })
}
