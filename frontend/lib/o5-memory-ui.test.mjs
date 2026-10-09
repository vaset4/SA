/** 【项目记忆】【交互回归】执行真实组件及API封装，所有网络端口隔离，作者：xxx */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'

import { hasPendingCandidates } from './result-delivery.mjs'
import { turnDepthView, summarizeTurnCapabilities, describeMemoryWriteback } from './research-turn.mjs'

const require = createRequire(import.meta.url)

/** @param {string} file 模块路径 @param {object} ports 依赖端口 @returns {Promise<object>} 真实模块与确定性React调度 */
async function loadModule(file, ports = {}) {
  const state = []
  let cursor = 0
  const react = {
    useState(initial) {
      const index = cursor++
      if (!(index in state)) state[index] = initial
      return [state[index], (value) => { state[index] = typeof value === 'function' ? value(state[index]) : value }]
    },
    useEffect() {},
  }
  const context = createContext({ URLSearchParams, console })
  const modules = { react, 'react/jsx-runtime': { jsx: (type, props) => ({ type, props }),
    jsxs: (type, props) => ({ type, props }), Fragment: 'fragment' },
    '../../lib/study-assistant-workspace.mjs': { getError: (error) => error.message },
    '../../lib/result-delivery.mjs': { hasPendingCandidates }, ...ports }
  const source = fs.readFileSync(new URL(file, import.meta.url), 'utf8')
  const { code } = require('next/dist/build/swc').transformSync(source, {
    filename: file, jsc: { parser: { syntax: 'ecmascript', jsx: true }, target: 'es2022',
      transform: { react: { runtime: 'automatic' } } }, module: { type: 'es6' },
  })
  const module = new SourceTextModule(code, { context })
  await module.link((name) => new SyntheticModule(Object.keys(modules[name]), function initialize() {
    for (const [key, value] of Object.entries(modules[name])) this.setExport(key, value)
  }, { context }))
  await module.evaluate()
  return { api: module.namespace, render(name, props) { cursor = 0; return module.namespace[name](props) } }
}

/** @param {object} tree JSX树 @param {Function} predicate 匹配规则 @returns {object[]} 匹配元素 */
function nodes(tree, predicate) {
  if (!tree || typeof tree !== 'object') return []
  if (Array.isArray(tree)) return tree.flatMap((item) => nodes(item, predicate))
  return [...(predicate(tree) ? [tree] : []), ...nodes(tree.props?.children, predicate)]
}

/** @param {object} value 跨VM数据 @returns {object} 可比较的普通JSON数据 */
function plain(value) { return JSON.parse(JSON.stringify(value)) }

/** @param {object} tree 实际组件JSX树 @returns {string} 用户可见文本，不把属性或源码当显示证据 */
function visibleText(tree) {
  if (tree == null || typeof tree === 'boolean') return ''
  if (typeof tree !== 'object') return String(tree)
  if (Array.isArray(tree)) return tree.map(visibleText).join(' ')
  return visibleText(tree.props?.children)
}

/** @param {object} workspace 活动运行与事件 @returns {Promise<object>} 真实聊天主卡，隔离无关子组件 */
async function loadAnswerShell(workspace) {
  const trace = await loadModule('./research-trace.js', {
    './external-research-mode': { externalResearchModeLabel: (mode) => mode, normalizeExternalResearchMode: (mode) => mode },
  })
  const children = ['MessageContent', 'MessageEvidencePanel', 'MessageMemoryCapture', 'MessageMemoryCandidates',
    'MessageResearchPanel', 'MessageResearchSteps', 'MessageTracePanel', 'MessageTurnSummary',
    'MessagePersistence', 'MessageMemoryWritebackRetry']
  const ports = Object.fromEntries(children.map((name) => [`./${name}`, { [name]: () => null }]))
  return loadModule('../components/chat/MessageAnswerShell.jsx', { ...ports,
    '../../stores/workspace-store': { useWorkspaceStore: () => workspace },
    '../../lib/research-turn.mjs': { turnDepthView }, '../../lib/research-trace': trace.api,
  })
}

for (const depth of ['minimal', 'lookup', 'research']) {
  test(`聊天${depth}运行卡直接展示真实额度等待与可取消提示`, async () => {
    const workspace = { activeRun: { run_id: 'current' }, researchTrace: { events: [
      { stage: 'provider_queued', run_id: 'current', request_id: 'request-a', estimated_wait_seconds: 12.2 },
    ] } }
    const shell = await loadAnswerShell(workspace)
    const tree = shell.render('MessageAnswerShell', { message: { pending: true, turn_depth: depth } })
    assert.match(visibleText(tree), /等待模型服务额度/)
    assert.match(visibleText(tree), /预计等待约 13 秒.*可取消/)
  })
}

test('聊天等待按请求清除，并发放行不抹掉另一个请求，历史运行与已结束卡不显示等待', async () => {
  const events = [
    { stage: 'provider_queued', run_id: 'old', request_id: 'old-request' },
    { stage: 'provider_queued', run_id: 'current', request_id: 'a', estimated_wait_seconds: 12 },
    { stage: 'provider_queued', run_id: 'current', request_id: 'b', estimated_wait_seconds: 24 },
    { stage: 'provider_admitted', run_id: 'current', request_id: 'a' },
  ]
  const shell = await loadAnswerShell({ activeRun: { run_id: 'current' }, researchTrace: { events } })
  const message = { pending: true, turn_depth: 'minimal' }
  assert.match(visibleText(shell.render('MessageAnswerShell', { message })), /等待模型服务额度/)
  for (const stage of ['provider_admitted', 'provider_cancelled', 'provider_queue_failed']) {
    events.push({ stage, run_id: 'current', request_id: 'b' })
    assert.doesNotMatch(visibleText(shell.render('MessageAnswerShell', { message })), /等待模型服务额度/)
    events.pop()
  }
  assert.doesNotMatch(visibleText(shell.render('MessageAnswerShell', { message: { ...message, pending: false } })), /等待模型服务额度/)
})

test('明确未保存的候选不显示查询重试操作，补存后的同份答案恢复候选查询', async () => {
  const loaded = await loadModule('../components/chat/MessageMemoryWritebackRetry.jsx', {
    '../../lib/api/rag-runs': { retryRagRunMemoryWriteback: async () => assert.fail('渲染不得触发模型') },
    '../../stores/workspace-store': { useWorkspaceStore: () => ({}) },
  })
  for (const persistence_status of ['pending', 'not_staged']) {
    for (const memory_writeback of [{ status: 'pending' }, { status: 'failed', retryable: true }]) {
      const tree = loaded.render('MessageMemoryWritebackRetry', { message: { run_id: 'run', persistence_status, memory_writeback } })
      assert.equal(nodes(tree, (node) => node.type === 'button').length, 0)
    }
  }
  const saved = loaded.render('MessageMemoryWritebackRetry', { message: {
    run_id: 'run', persistence_status: 'saved', memory_writeback: { status: 'pending' },
  } })
  assert.match(visibleText(saved), /查询候选状态/)
})

test('真实回合摘要在保存失败时不声称已保存，补存后按保存事实显示', async () => {
  const loaded = await loadModule('../components/chat/MessageTurnSummary.jsx', {
    '../../lib/research-turn.mjs': { isProjectResearchTurn: () => true, summarizeTurnCapabilities },
  })
  const base = { turn_depth: 'minimal', memory_writeback: { status: 'pending' } }
  for (const persistence_status of ['pending', 'not_staged']) {
    const text = visibleText(loaded.render('MessageTurnSummary', { message: { ...base, persistence_status } }))
    assert.doesNotMatch(text, /答案已保存|记忆候选处理中/)
    assert.match(text, /先保存答案/)
  }
  assert.match(visibleText(loaded.render('MessageTurnSummary', { message: { ...base, persistence_status: 'saved' } })), /答案已保存/)
  assert.doesNotMatch(visibleText(loaded.render('MessageTurnSummary', { message: base })), /答案已保存/)
})

test('未保存预览即使携带候选ID也不展示候选详情入口', async () => {
  const loaded = await loadModule('../components/chat/MessageMemoryCandidates.jsx', {
    '../project/MemoryDetailDrawer': { MemoryDetailDrawer: () => null },
    '../../lib/api/research-memory': { getMemory: async () => assert.fail('未保存不读候选'), acceptDraft: () => {}, dismissDraft: () => {} },
    '../../lib/research-turn.mjs': { describeMemoryWriteback },
    '../../stores/workspace-store': { useWorkspaceStore: () => ({ projectId: 'project' }) },
  })
  const base = { memory_writeback: { status: 'success', candidate_ids: ['candidate'] } }
  assert.equal(loaded.render('MessageMemoryCandidates', { message: { ...base, persistence_status: 'pending' } }), null)
  assert.notEqual(loaded.render('MessageMemoryCandidates', { message: { ...base, persistence_status: 'saved' } }), null)
})

test('编辑后纳入提交修改正文和读取版本，不先保存旧正文', async () => {
  const calls = []
  const loaded = await loadModule('../components/project/MemoryEditForm.jsx', {
    '../../lib/api/research-memory': { acceptDraft: async (payload) => calls.push(payload),
      updateMemory: async () => assert.fail('待审纳入必须走单次接纳') },
  })
  const props = { projectId: 'project', item: { memory_id: 'memory', title: '标题', body: '旧判断',
    version: 7, review_state: 'pending_review' } }
  let tree = loaded.render('MemoryEditForm', props)
  nodes(tree, (node) => node.type === 'textarea')[0].props.onChange({ target: { value: '修改后的判断' } })
  tree = loaded.render('MemoryEditForm', props)
  await nodes(tree, (node) => node.type === 'button')[1].props.onClick()
  assert.deepEqual(plain(calls), [{ projectId: 'project', memoryId: 'memory', expectedVersion: 7,
    patch: { title: '标题', body: '修改后的判断' } }])
})

test('绑定原文包含确认和版本，歧义选点后传递精确位置', async () => {
  const calls = []
  const location = { chunk_id: 'chunk-a', source_field: 'content', source_start: 2, source_end: 6 }
  const loaded = await loadModule('../components/project/MemoryEvidenceBindForm.jsx', {
    '../../lib/api/projects': { getProjectPapers: async () => assert.fail('测试不运行数据加载effect') },
    '../../lib/api/research-memory': { addEvidence: async (payload) => {
      calls.push(payload)
      if (calls.length === 1) throw { message: '请选择位置', body: { detail: { candidates: [location] } } }
    } },
  })
  const props = { projectId: 'project', item: { memory_id: 'memory', version: 3 } }
  let tree = loaded.render('MemoryEvidenceBindForm', props)
  nodes(tree, (node) => node.type?.name === 'PaperPicker')[0].props.onChange({ target: { value: 'paper' } })
  nodes(tree, (node) => node.type === 'textarea')[0].props.onChange({ target: { value: '  原文😀 ' } })
  tree = loaded.render('MemoryEvidenceBindForm', props)
  await nodes(tree, (node) => node.type === 'button')[0].props.onClick()
  tree = loaded.render('MemoryEvidenceBindForm', props)
  assert.equal(nodes(tree, (node) => node.type === 'button')[0].props.disabled, true)
  nodes(tree, (node) => node.type?.name === 'SourceChoices')[0].props.onSelect(0)
  tree = loaded.render('MemoryEvidenceBindForm', props)
  await nodes(tree, (node) => node.type === 'button')[0].props.onClick()
  assert.deepEqual(plain(calls[1].payload), { paper_id: 'paper', quote_text: '  原文😀 ',
    expected_version: 3, confirm: true, ...location })
})

test('记忆API序列化版本、编辑正文与删除条件', async () => {
  const calls = []
  const { api } = await loadModule('./api/research-memory.js', { './core': {
    requestJson: async (request) => { calls.push(request); return {} },
  } })
  await api.acceptDraft({ projectId: 'p/x', memoryId: 'm/x', expectedVersion: 8, patch: { body: '修改' } })
  await api.removeEvidence({ projectId: 'p/x', memoryId: 'm/x', refId: 'ref/x', expectedVersion: 8 })
  await api.confirmMemory({ projectId: 'p/x', memoryId: 'm/x', expectedVersion: 9 })
  assert.deepEqual(JSON.parse(calls[0].init.body), { body: '修改', expected_version: 8 })
  assert.match(calls[0].path, /projects\/p%2Fx\/memory\/m%2Fx\/accept$/)
  assert.match(calls[1].path, /ref%2Fx\?expected_version=8$/)
  assert.deepEqual(JSON.parse(calls[2].init.body), { expected_version: 9 })
})

test('研究重试与补存API保留同次请求标识及命名超时', async () => {
  const calls = []
  const ports = { './core': { requestJson: async (request) => { calls.push(request); return {} } },
    '../study-assistant-workspace.mjs': { createId: () => 'new-action' } }
  const research = await loadModule('./api/research.js', ports)
  await research.api.retryResearchSession({ sessionId: 'session', idempotencyKey: 'same-action' })
  const runs = await loadModule('./api/rag-runs.js', ports)
  await runs.api.retryRagRunSave({ runId: 'run', idempotencyKey: 'same-save' })
  assert.equal(calls[0].timeoutMs, 600_000)
  assert.equal(calls[0].init.headers['Idempotency-Key'], 'same-action')
  assert.equal(calls[1].init.headers['Idempotency-Key'], 'same-save')
  assert.equal(calls[1].path, '/api/v1/rag-runs/run/retry-save')
})

test('候选重试按钮仅调用独立候选接口并透传原答案版本', async () => {
  const applied = []
  const result = { run_id: 'run', delivery_revision: 4, memory_writeback: { status: 'success', candidate_ids: ['memory'] } }
  const loaded = await loadModule('../components/chat/MessageMemoryWritebackRetry.jsx', {
    '../../lib/api/rag-runs': { retryRagRunMemoryWriteback: async ({ runId }) => {
      assert.equal(runId, 'run')
      return result
    } },
    '../../stores/workspace-store': { useWorkspaceStore: () => ({ conversationId: 'conversation',
      applyMemoryWriteback: (payload) => applied.push(payload), reloadCandidates: (runId) => assert.equal(runId, 'run') }) },
  })
  const props = { message: { run_id: 'run', memory_writeback: { status: 'failed', retryable: true } } }
  const tree = loaded.render('MessageMemoryWritebackRetry', props)
  await nodes(tree, (node) => node.type === 'button')[0].props.onClick()
  assert.deepEqual(plain(applied), [{ conversationId: 'conversation', result }])
  assert.equal(loaded.render('MessageMemoryWritebackRetry', { message: { memory_writeback: { status: 'failed', retryable: false } } }), null)
})
