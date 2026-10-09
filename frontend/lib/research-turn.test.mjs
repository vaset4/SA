import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TURN_DEPTH_MINIMAL,
  TURN_DEPTH_LOOKUP,
  TURN_DEPTH_RESEARCH,
  isProjectResearchTurn,
  turnDepthView,
  describeMemoryRead,
  describeMemoryWriteback,
  summarizeTurnCapabilities,
} from './research-turn.mjs'

import {
  MEMORY_STATE_CONFIRMED,
  MEMORY_STATE_PENDING,
  MEMORY_STATE_REBIND,
  MEMORY_STATE_INACTIVE,
  classifyMemoryState,
  groupMemoryByState,
  pendingActionKind,
  isAiDraftSource,
} from './research-memory.mjs'

import { MESSAGE_EXTRA_KEYS, pickMessageExtras } from './study-assistant-workspace.mjs'

// ─── 回合深度：判定与差异化露出 ───

test('缺 turn_depth 的普通问答不算项目科研回合，入口按老样子全露出', () => {
  assert.equal(isProjectResearchTurn({}), false)
  assert.equal(isProjectResearchTurn({ turn_depth: 'unknown_depth' }), false)
  const view = turnDepthView(undefined)
  assert.deepEqual(view, { showEvidenceToggle: true, showProcessToggle: true, showProgressSteps: true })
})

test('直接协助档不露依据入口与进度条，但保留过程入口', () => {
  const view = turnDepthView(TURN_DEPTH_MINIMAL)
  assert.equal(view.showEvidenceToggle, false)
  assert.equal(view.showProgressSteps, false)
  assert.equal(view.showProcessToggle, true)
})

test('轻量查证与复杂研究两档都露依据入口与进度条', () => {
  for (const depth of [TURN_DEPTH_LOOKUP, TURN_DEPTH_RESEARCH]) {
    const view = turnDepthView(depth)
    assert.equal(view.showEvidenceToggle, true, depth)
    assert.equal(view.showProgressSteps, true, depth)
  }
})

// ─── 记忆读取四态：empty 与 failed 不得归并 ───

test('后端没下发记忆读取终态时返回 null，调用方不渲染，也不显示成未使用', () => {
  assert.equal(describeMemoryRead(null), null)
  assert.equal(describeMemoryRead({}), null)
})

test('记忆读取四态各自独立，empty 与 failed 色调不同', () => {
  const success = describeMemoryRead({ status: 'success', items: [{ memory_id: 'm1' }, { memory_id: 'm2' }] })
  assert.equal(success.itemCount, 2)
  assert.equal(success.tone, 'ok')
  assert.match(success.label, /2 条/)

  const empty = describeMemoryRead({ status: 'empty', items: [] })
  const failed = describeMemoryRead({ status: 'failed', items: [], error_kind: 'repository_error' })
  assert.equal(empty.tone, 'neutral')
  assert.equal(failed.tone, 'danger')
  assert.notEqual(empty.label, failed.label)
  assert.match(failed.detail, /repository_error/)

  const skipped = describeMemoryRead({ status: 'skipped', items: [] })
  assert.notEqual(skipped.label, empty.label)
})

test('读到记忆不说成「已引用」，措辞要点明记忆不是引用来源', () => {
  const success = describeMemoryRead({ status: 'success', items: [{ memory_id: 'm1' }] })
  assert.doesNotMatch(success.label, /已引用/)
  assert.match(success.detail, /不是论文原文|不会把它们当引用来源/)
})

// ─── 候选写回四态 ───

test('重新打开已保存答案时，候选处理中不显示失败或允许重复启动', () => {
  const result = describeMemoryWriteback({ status: 'pending', candidate_ids: [], retryable: true })
  assert.equal(result.tone, 'neutral')
  assert.match(result.label, /处理中/)
  assert.equal(result.retryable, false)
  assert.deepEqual(result.candidateIds, [])
  assert.doesNotMatch(result.detail, /AI 整理|失败/)
})

test('写回成功只说生成待审候选，绝不说已纳入记忆', () => {
  const result = describeMemoryWriteback({ status: 'success', candidate_ids: ['c1', 'c2'] })
  assert.deepEqual(result.candidateIds, ['c1', 'c2'])
  assert.doesNotMatch(result.label, /已纳入/)
  assert.match(result.label, /待审候选/)
  assert.match(result.detail, /确认后/)
})

test('skipped 与 empty 分开表述，且都不产出候选卡', () => {
  const skipped = describeMemoryWriteback({ status: 'skipped', reason_codes: ['depth_not_research'] })
  const empty = describeMemoryWriteback({ status: 'empty', reason_codes: ['no_long_term_value'] })
  assert.notEqual(skipped.label, empty.label)
  assert.deepEqual(skipped.candidateIds, [])
  assert.deepEqual(empty.candidateIds, [])
  assert.match(skipped.detail, /depth_not_research/)
  assert.match(empty.detail, /no_long_term_value/)
})

test('写回失败按 retryable 区分文案，不可重试的不谎称可重试', () => {
  const retryable = describeMemoryWriteback({ status: 'failed', retryable: true, error_kind: 'storage_unavailable' })
  const permanent = describeMemoryWriteback({ status: 'failed', retryable: false, error_kind: 'contract_violation' })
  assert.equal(retryable.retryable, true)
  assert.match(retryable.label, /可重试/)
  assert.doesNotMatch(retryable.detail, /AI 整理/)
  assert.equal(permanent.retryable, false)
  assert.doesNotMatch(permanent.label, /可重试/)
})

// ─── 本轮能力汇总 ───

test('非项目科研回合汇总为空，不给它编一份终态', () => {
  assert.deepEqual(summarizeTurnCapabilities({ citations: [{ paper_id: 'p1' }] }), [])
})

test('需要论文证据的深度没命中引用时，明确显示无检索结果', () => {
  const rows = summarizeTurnCapabilities({ turn_depth: TURN_DEPTH_LOOKUP, citations: [] })
  const retrieval = rows.find((row) => row.key === 'retrieval')
  assert.equal(retrieval.label, '无检索结果')
  assert.equal(retrieval.tone, 'danger')
})

test('直接协助档不摆检索行：这一档压根没发检索', () => {
  const rows = summarizeTurnCapabilities({ turn_depth: TURN_DEPTH_MINIMAL, citations: [] })
  assert.equal(rows.find((row) => row.key === 'retrieval'), undefined)
})

test('证据缺口与写回失败都进汇总，不藏', () => {
  const rows = summarizeTurnCapabilities({
    turn_depth: TURN_DEPTH_RESEARCH,
    citations: [{ paper_id: 'p1' }],
    missing_evidence: [{ claim: 'c1' }, { claim: 'c2' }],
    memory_read_result: { status: 'empty', items: [] },
    memory_writeback: { status: 'failed', retryable: true, error_kind: 'storage_unavailable' },
  })
  const keys = rows.map((row) => row.key)
  assert.deepEqual(keys, ['depth', 'memory_read', 'retrieval', 'evidence_gap', 'memory_writeback'])
  assert.match(rows.find((row) => row.key === 'evidence_gap').label, /2 处/)
})

// ─── 记忆状态三分档：互斥且完备 ───

const CONFIRMED_ITEM = { review_state: 'accepted', status: 'confirmed', stale_state: 'fresh' }

test('只有四道资格门都满足的条目进已确认档', () => {
  assert.equal(classifyMemoryState(CONFIRMED_ITEM), MEMORY_STATE_CONFIRMED)
  // 逐轴打破一次，都必须掉出已确认档
  assert.notEqual(classifyMemoryState({ ...CONFIRMED_ITEM, status: 'draft' }), MEMORY_STATE_CONFIRMED)
  assert.notEqual(classifyMemoryState({ ...CONFIRMED_ITEM, review_state: 'pending_review' }), MEMORY_STATE_CONFIRMED)
  assert.notEqual(classifyMemoryState({ ...CONFIRMED_ITEM, stale_state: 'stale' }), MEMORY_STATE_CONFIRMED)
  assert.notEqual(classifyMemoryState({ ...CONFIRMED_ITEM, stale_state: 'needs_rebind' }), MEMORY_STATE_CONFIRMED)
})

test('状态轴缺值或不认识的值不算已确认，已确认档必须逐值匹配', () => {
  // 已确认档对用户的承诺是「Agent 现在读得到这些」，而后端那道门要的是 status 字面等于
  // confirmed。缺了这个字段或来了个没见过的值时归进已确认，用户看到的生效范围就比真的大
  assert.equal(classifyMemoryState({ review_state: 'accepted', stale_state: 'fresh' }), MEMORY_STATE_INACTIVE)
  assert.equal(classifyMemoryState({ ...CONFIRMED_ITEM, status: 'some_new_status' }), MEMORY_STATE_INACTIVE)
  assert.equal(classifyMemoryState({ ...CONFIRMED_ITEM, review_state: 'some_new_review' }), MEMORY_STATE_INACTIVE)
  assert.equal(classifyMemoryState({ ...CONFIRMED_ITEM, stale_state: '' }), MEMORY_STATE_INACTIVE)
  assert.equal(classifyMemoryState({}), MEMORY_STATE_INACTIVE)
})

test('失效轴优先级最高：待审的条目一旦失效归到待重新核对', () => {
  const staleePending = { review_state: 'pending_review', status: 'draft', stale_state: 'needs_rebind' }
  assert.equal(classifyMemoryState(staleePending), MEMORY_STATE_REBIND)
})

test('已否决与已归档不混进待审档，避免让用户再审一次已处理的条目', () => {
  assert.equal(classifyMemoryState({ review_state: 'dismissed', status: 'draft', stale_state: 'fresh' }), MEMORY_STATE_INACTIVE)
  assert.equal(classifyMemoryState({ review_state: 'accepted', status: 'archived', stale_state: 'fresh' }), MEMORY_STATE_INACTIVE)
})

test('分组互斥且完备：每条恰好落一档，总数守恒', () => {
  const items = [
    CONFIRMED_ITEM,
    { review_state: 'pending_review', status: 'draft', stale_state: 'fresh' },
    { review_state: 'accepted', status: 'draft', stale_state: 'fresh', evidence_mode: 'evidence_backed' },
    { review_state: 'accepted', status: 'confirmed', stale_state: 'stale' },
    { review_state: 'dismissed', status: 'draft', stale_state: 'fresh' },
  ]
  const groups = groupMemoryByState(items)
  const total = Object.values(groups).reduce((sum, group) => sum + group.length, 0)
  assert.equal(total, items.length)
  assert.equal(groups[MEMORY_STATE_CONFIRMED].length, 1)
  assert.equal(groups[MEMORY_STATE_PENDING].length, 2)
  assert.equal(groups[MEMORY_STATE_REBIND].length, 1)
  assert.equal(groups[MEMORY_STATE_INACTIVE].length, 1)
})

// ─── 待审档内按来源分操作 ───

test('AI 草稿走接纳，手写的文献背书草稿走挂证据', () => {
  assert.equal(pendingActionKind({ review_state: 'pending_review', source_kind: 'agent_draft' }), 'accept_draft')
  assert.equal(
    pendingActionKind({ review_state: 'accepted', status: 'draft', evidence_mode: 'evidence_backed', source_kind: 'manual' }),
    'bind_evidence',
  )
  // 纯笔记草稿不需要挂证据，直接确认即可
  assert.equal(
    pendingActionKind({ review_state: 'accepted', status: 'draft', evidence_mode: 'user_note_only', source_kind: 'manual' }),
    'confirm',
  )
})

test('AI 草稿即使是文献背书也不走挂证据入口，它该先被接纳', () => {
  assert.equal(isAiDraftSource('agent_draft'), true)
  assert.equal(isAiDraftSource('system_draft'), true)
  assert.equal(isAiDraftSource('manual'), false)
  assert.equal(
    pendingActionKind({ review_state: 'pending_review', evidence_mode: 'evidence_backed', source_kind: 'agent_draft' }),
    'accept_draft',
  )
})

// ─── 回合级字段必须过前端白名单，否则历史消息重现不出本轮结果 ───

test('四个回合级键在 MESSAGE_EXTRA_KEYS 里，且能被 pickMessageExtras 取出', () => {
  const turnKeys = ['turn_depth', 'depth_transitions', 'memory_read_result', 'memory_writeback']
  for (const key of turnKeys) {
    assert.ok(MESSAGE_EXTRA_KEYS.includes(key), `${key} 不在白名单里`)
  }
  const extras = pickMessageExtras({
    metadata: {
      turn_depth: TURN_DEPTH_RESEARCH,
      memory_read_result: { status: 'success', items: [{ memory_id: 'm1' }] },
      memory_writeback: { status: 'success', candidate_ids: ['c1'] },
      depth_transitions: [],
    },
  })
  assert.equal(extras.turn_depth, TURN_DEPTH_RESEARCH)
  assert.equal(extras.memory_read_result.status, 'success')
  assert.deepEqual(extras.memory_writeback.candidate_ids, ['c1'])
  assert.deepEqual(extras.depth_transitions, [])
})
