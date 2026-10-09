/** 认证请求和 store 的隔离测试工具，作者：xxx */
import fs from 'node:fs'
import { createRequire } from 'node:module'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'

const require = createRequire(import.meta.url)

/** 无参数，返回可由测试显式控制完成顺序的 Promise */
export function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

/** @param {object} payload JSON 正文 @param {number} status HTTP 状态 @returns {Response} 测试响应 */
export function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } })
}

/** @param {string} userId 用户 @param {string} revision 令牌版本标签 @returns {object} 完整会话 */
export function sessionPayload(userId, revision = 'initial') {
  return {
    access_token: userId + '-' + revision + '-access',
    refresh_token: userId + '-' + revision + '-refresh',
    user: { user_id: userId, username: userId, role: 'user' },
  }
}

/** @param {object} exports 依赖端口 @param {object} context VM 上下文 @returns {SyntheticModule} 可注入模块 */
function dependencyModule(exports, context) {
  return new SyntheticModule(Object.keys(exports), function initialize() {
    Object.entries(exports).forEach(([name, value]) => this.setExport(name, value))
  }, { context })
}

/** @param {string} path 相对测试目录的源码路径 @param {object} context VM 上下文 @returns {SourceTextModule} 真实源码模块 */
function sourceModule(path, context) {
  return new SourceTextModule(fs.readFileSync(new URL(path, import.meta.url), 'utf8'), { context, identifier: path })
}

/** @param {Function} fetchImplementation 网络端口 @param {object} options 浏览器或计时端口 @returns {Promise<object>} 真实认证与 API 模块 */
export async function loadAuthRequestModules(fetchImplementation, options = {}) {
  const storageState = { session: options.session || null, activeId: 'previous-conversation' }
  const context = createContext({
    AbortController, DOMException, Error, TypeError, Headers, FormData, clearTimeout, console,
    fetch: fetchImplementation, process: { env: {} }, setTimeout,
    ...options.globals,
  })
  const storage = dependencyModule({ StorageService: {
    getAuthSession: () => storageState.session,
    saveAuthSession: (session) => { storageState.session = session },
    clearAuthSession: () => { storageState.session = null },
    saveActiveConversationId: (value) => { storageState.activeId = value },
    getActiveConversationId: () => storageState.activeId,
    normalizeSessionId: (value) => value || '',
  } }, context)
  const authClient = sourceModule('./auth-client.js', context)
  const apiCore = sourceModule('./api/core.js', context)
  const dependencies = new Map([
    ['./api/core', apiCore], ['./auth-client', authClient],
    ['./storage-service', storage], ['../lib/storage-service', storage],
    ['./workspace-routes', dependencyModule({ buildWorkspaceRedirectTarget: () => '/projects' }, context)],
    ['./auth-session.mjs', sourceModule('./auth-session.mjs', context)],
    ['../auth-client', authClient], ['../lib/auth-client', authClient], ['./core', apiCore],
  ])
  await apiCore.link((specifier) => dependencies.get(specifier))
  await apiCore.evaluate()
  const authApi = sourceModule('./api/auth.js', context)
  await authApi.link((specifier) => dependencies.get(specifier))
  await authApi.evaluate()
  dependencies.set('../lib/api/auth', authApi)
  return { authClient: authClient.namespace, apiCore: apiCore.namespace, authApi: authApi.namespace, context, dependencies, storageState }
}

/** @param {object} modules 真实认证模块 @param {string} path 相对源码路径 @returns {Promise<object>} 已连接的模块导出 */
export async function loadClientModule(modules, path) {
  const client = sourceModule(path, modules.context)
  await client.link((name) => modules.dependencies.get(name))
  await client.evaluate()
  return client.namespace
}

/** 无参数，返回确定性 reducer/effect 驱动器；只模拟 React 调度，不复制任何业务逻辑 */
function createHookDriver() {
  let state
  let effectCursor = 0
  let pending = []
  const effects = []
  const react = {
    Fragment: Symbol('Fragment'),
    createContext: (value) => ({ value, Provider: Symbol('Provider') }),
    useContext: (context) => context.value,
    useMemo: (factory) => factory(),
    useReducer(reducer, initial, initialize) {
      if (state === undefined) state = initialize ? initialize(initial) : initial
      return [state, (action) => { state = reducer(state, action) }]
    },
    useEffect(effect, dependencies) {
      const index = effectCursor++
      const previous = effects[index]
      if (previous && dependencies.every((value, i) => Object.is(value, previous.dependencies[i]))) return
      pending.push(() => {
        previous?.cleanup?.()
        effects[index] = { dependencies, cleanup: effect() }
      })
    },
  }
  return {
    react,
    get state() { return state },
    render(component) { effectCursor = 0; return component({ children: null }) },
    runEffects() { const queue = pending; pending = []; queue.forEach((effect) => effect()) },
    unmount() { effects.forEach((effect) => effect?.cleanup?.()); pending = [] },
  }
}

/** @param {object} modules 请求模块 @param {object} options store 路径及依赖端口 @returns {Promise<object>} 执行真实 store 源码的测试入口 */
export async function loadStore(modules, { path, component, dependencies = {} }) {
  const hooks = createHookDriver()
  const exports = { jsx: (type, props, key) => ({ type, props, key }) }
  const mapping = new Map(modules.dependencies)
  mapping.set('react', dependencyModule(hooks.react, modules.context))
  mapping.set('react/jsx-runtime', dependencyModule(exports, modules.context))
  for (const [name, values] of Object.entries(dependencies)) mapping.set(name, dependencyModule(values, modules.context))
  const source = fs.readFileSync(new URL(path, import.meta.url), 'utf8')
  const { code } = require('next/dist/build/swc').transformSync(source, {
    filename: path, jsc: { parser: { syntax: 'ecmascript', jsx: true }, target: 'es2022', transform: { react: { runtime: 'automatic' } } },
    module: { type: 'es6' },
  })
  const store = new SourceTextModule(code, { context: modules.context, identifier: path })
  await store.link((name) => mapping.get(name))
  await store.evaluate()
  return { ...hooks, get state() { return hooks.state }, render: () => hooks.render(store.namespace[component]) }
}
