import assert from 'node:assert/strict'
import { test } from 'node:test'
import { applyCurrentResult, hasPendingCandidates, candidateRecoveryError, applyDeliveryPatch, deliveryResponsePatch, describeDelivery, describePersistence, sessionDeliveryValues } from './result-delivery.mjs'
import { pickMessageExtras } from './study-assistant-workspace.mjs'

test('迟到的旧答案不能覆盖新版本，也不能移除并发追加的消息', () => {
  const messages = [{ id: 'a', run_id: 'r', content: '新版', delivery_revision: 3 }, { id: 'b', content: '新问题' }]
  const patch = { run_id: 'r', content: '旧版', delivery_revision: 2 }
  assert.deepEqual(applyDeliveryPatch(messages, { messageId: 'a', runId: 'r', patch }), messages)
})

test('提交替换整份来源和正文，原始空白与Unicode引用位置不变', () => {
  const messages = [{ id: 'a', run_id: 'r', content: '旧版', delivery_revision: 1, evidence_records: [{ id: 'old' }], memory_writeback: { status: 'old' } }]
  const patch = { run_id: 'r', content: '  😀 [E1]\n', delivery_revision: 2, final_generation_attempt: { attempt_id: 'new' }, persistence_status: 'saved' }
  const [next] = applyDeliveryPatch(messages, { messageId: 'a', runId: 'r', patch })
  assert.equal(next.content, patch.content)
  assert.equal(next.evidence_records, undefined)
  assert.equal(next.memory_writeback, undefined)
  assert.equal(pickMessageExtras(next).delivery_revision, 2)
  assert.equal(pickMessageExtras(next).persistence_status, 'saved')
})

test('不属于目标的补丁与缺版本补丁明确拒绝', () => {
  assert.throws(() => applyDeliveryPatch([], { runId: 'r', patch: { run_id: 'other', delivery_revision: 1, content: '' } }), /同一次研究/)
  assert.throws(() => applyDeliveryPatch([], { runId: 'r', patch: { run_id: 'r', content: '' } }), /版本/)
})

test('新合同未提交的checkpoint正文不得展示，历史结果和澄清保留', () => {
  assert.deepEqual(sessionDeliveryValues({ current_run_id: 'r', values: { answer: '未提交' } }), {})
  assert.deepEqual(sessionDeliveryValues({ current_run_id: 'r', current_result: { answer: '已提交' }, values: { answer: '未提交' } }), { answer: '已提交' })
  assert.equal(sessionDeliveryValues({ values: { answer: '旧答案' } }).answer, '旧答案')
  assert.equal(sessionDeliveryValues({ current_run_id: 'r', status: 'awaiting_clarification', values: { clarification_question: '补充范围' } }).clarification_question, '补充范围')
})

test('四种交付和独立保存失败不显示统一成功，仅服务端允许才可补存', () => {
  const labels = ['complete', 'partial', 'insufficient', 'failed'].map((status) => describeDelivery({ status }).label)
  assert.equal(new Set(labels).size, 4)
  assert.equal(describePersistence({ persistence_status: 'pending' }).canRetry, false)
  assert.equal(describePersistence({ persistence_status: 'pending', can_retry_save: true }).canRetry, true)
  assert.equal(describePersistence({ persistence_status: 'unavailable', can_retry_save: true }).canRetry, false)
})

test('最终重试保存失败保留真实新正文和补存动作，但不伪造新提交版本', () => {
  const patch = deliveryResponsePatch({ response: { run_id: 'r', answer: '  未保存😀\n',
    delivery_revision: 2, persistence_status: 'pending', can_retry_save: true } })
  const messages = [{ id: 'message', run_id: 'r', delivery_revision: 2, content: '旧正文' }]
  const [preview] = applyDeliveryPatch(messages, { messageId: 'message', runId: 'r', patch })
  assert.equal(preview.content, '  未保存😀\n')
  assert.equal(preview.delivery_revision, 2)
  assert.equal(describePersistence(preview).canRetry, true)
  const newer = [{ ...messages[0], delivery_revision: 3 }]
  assert.deepEqual(applyDeliveryPatch(newer, { messageId: 'message', runId: 'r', patch }), newer)
})

test('当前结果同版本只更新候选，高版本整份替换，旧版本不覆盖', () => {
  const messages = [{ id: 'a', run_id: 'r', delivery_revision: 1, content: '原正文', final_delivery_report: { status: 'partial' } }]
  const current = { run_id: 'r', delivery_revision: 1, answer: '禁止覆盖', memory_writeback: { status: 'empty' } }
  const same = applyCurrentResult(messages, current)
  assert.equal(same[0].content, '原正文')
  assert.equal(same[0].memory_writeback.status, 'empty')
  assert.deepEqual(applyCurrentResult(same, { ...current, delivery_revision: 0 }), same)
  const newer = applyCurrentResult(same, { ...current, delivery_revision: 2, answer: '新正文' })
  assert.equal(newer[0].content, '新正文')
  assert.equal(newer[0].final_delivery_report, undefined)
})

test('旧terminal过渡标识会跟进查询，停止未定性只报恢复异常', () => {
  const result = { memory_writeback: { status: 'skipped', reason_codes: ['awaiting_result_commit'] } }
  assert.equal(hasPendingCandidates(result), true)
  assert.equal(candidateRecoveryError(result), '')
  assert.equal(candidateRecoveryError({ ...result, execution_stopped_at: 'now' }), '')
  assert.equal(result.memory_writeback.status, 'skipped')
})


import fs from 'node:fs'
import { runInNewContext } from 'node:vm'

/** @param {object} options 隔离端口 @returns {object} 执行真实工作区监听effect的调度句柄 */
function mountResultListener(options = {}) {
  const source = fs.readFileSync(new URL('../stores/workspace-store.js', import.meta.url), 'utf8')
  const start = source.indexOf('  // 1. 【研究交付】【当前结果】')
  const end = source.indexOf('  useEffect(() => {', source.indexOf('queryToken])', start))
  const timers = new Map()
  const actions = []
  let cleanup
  let currentAccount = true
  let calls = 0
  const context = {
    candidateRunId: 'r', candidateRevision: 1, pathname: '/chat', queryToken: 0,
    state: { conversationId: 'c', projectId: 'p' }, auth: { sessionContext: 'account-a' },
    AbortController, captureAuthContext: () => 'account-a', isCurrentAuthContext: () => currentAccount,
    hasPendingCandidates, candidateRecoveryError, getError: (error) => error.message,
    dispatch: (action) => actions.push(action),
    getRagRun: async (args) => { calls++; return options.getRun(args, calls) },
    RESULT_POLL_INTERVAL_MS: 2000,
    setTimeout: (fn) => { timers.set(1, fn); return 1 }, clearTimeout: (id) => timers.delete(id),
    useEffect: (effect) => { cleanup = effect() },
  }
  runInNewContext(source.slice(start, end), context)
  return { actions, timers, get calls() { return calls }, cleanup: () => cleanup(),
    switchAccount: () => { currentAccount = false },
    async tick() { const callback = timers.get(1); timers.delete(1); await callback?.() },
  }
}

/** @returns {Promise<void>} 等真实异步effect完成当前微任务 */
async function flushListener() { await new Promise((resolve) => setImmediate(resolve)) }

test('工作区单监听done后查当前结果，pending继续，最终状态停止', async () => {
  const listener = mountResultListener({ getRun: (_args, calls) => ({ current_result: {
    run_id: 'r', delivery_revision: 1, memory_writeback: { status: calls === 1 ? 'pending' : 'empty' },
  } }) })
  await flushListener()
  assert.equal(listener.calls, 1)
  assert.equal(listener.timers.size, 1)
  await listener.tick()
  assert.equal(listener.actions.at(-1).result.memory_writeback.status, 'empty')
  assert.equal(listener.timers.size, 0)
  listener.cleanup()
})

test('当前结果查询失败与停止未定性明确显示，不自动重试模型或查询', async () => {
  const failed = mountResultListener({ getRun: () => { throw new Error('网络中断') } })
  await flushListener()
  assert.equal(failed.actions.at(-1).patch.candidateQueryError, '网络中断')
  assert.equal(failed.timers.size, 0)
  const stopped = mountResultListener({ getRun: () => ({ current_result: {
    run_id: 'r', delivery_revision: 1, execution_stopped_at: 'now', memory_writeback: {
      status: 'pending', processing_source: 'delivery_run', processing_run_id: 'r' },
  } }) })
  await flushListener()
  assert.match(stopped.actions.at(-1).error, /恢复异常/)
  assert.equal(stopped.timers.size, 0)
  failed.cleanup(); stopped.cleanup()
})

test('页面离开和账号切换拒绝迟到的当前结果', async () => {
  for (const boundary of ['cleanup', 'switchAccount']) {
    let resolve
    const response = new Promise((done) => { resolve = done })
    const listener = mountResultListener({ getRun: () => response })
    listener[boundary]()
    resolve({ current_result: { run_id: 'r', delivery_revision: 1, memory_writeback: { status: 'pending' } } })
    await flushListener()
    assert.equal(listener.actions.length, 0)
    assert.equal(listener.timers.size, 0)
    listener.cleanup()
  }
})

test('明确未保存的答案不触发候选跟进，同份答案补存后允许继续查询', () => {
  for (const persistence_status of ['pending', 'not_staged']) {
    for (const memory_writeback of [{ status: 'pending' }, { status: 'skipped', reason_codes: ['awaiting_result_commit'] }]) {
      assert.equal(hasPendingCandidates({ persistence_status, memory_writeback }), false)
      assert.equal(hasPendingCandidates({ persistence_status: 'saved', memory_writeback }), true)
    }
  }
})

test('另一标签页的HTTP候选重试不继承原答案停止事实', async () => {
  const listener = mountResultListener({ getRun: (_args, calls) => ({ current_result: {
    run_id: 'r', delivery_revision: 1, operation_run_id: 'old', execution_stopped_at: 'old-stop',
    memory_writeback: calls === 1 ? { status: 'pending', processing_source: 'http_retry', processing_run_id: null }
      : { status: 'empty' },
  } }) })
  await flushListener()
  assert.equal(listener.actions.at(-1).error, '')
  assert.equal(listener.timers.size, 1)
  await listener.tick()
  assert.equal(listener.actions.at(-1).result.memory_writeback.status, 'empty')
  assert.equal(listener.timers.size, 0)
  listener.cleanup()
})

test('候选owner与迟到的旧操作停止事实不匹配时不能判恢复异常', () => {
  const pending = { run_id: 'r', operation_run_id: 'old', execution_stopped_at: 'old-stop',
    memory_writeback: { status: 'pending', processing_source: 'delivery_run', processing_run_id: 'new' } }
  assert.equal(candidateRecoveryError(pending), '')
  assert.match(candidateRecoveryError({ ...pending, operation_run_id: 'new' }), /恢复异常/)
})
