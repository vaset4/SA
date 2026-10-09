'use client'

import { useEffect, useState } from 'react'
import { MemoryDetailDrawer } from '../project/MemoryDetailDrawer'

import { getMemory, acceptDraft, dismissDraft } from '../../lib/api/research-memory'
import { describeMemoryWriteback } from '../../lib/research-turn.mjs'
import { getError } from '../../lib/study-assistant-workspace.mjs'
import { useWorkspaceStore } from '../../stores/workspace-store'

/**
 * 单张候选卡：正文 + 来源回答 + Evidence 数量 + 重复提示 + 纳入/编辑/忽略。
 * @param {{ detail: object, projectId: string, memoryHref: string, busy: boolean, onAction: Function }} props
 *   detail 是 getMemory 的详情载荷（item + evidence_refs）
 */
function CandidateCard({ detail, projectId, onOpen, busy, onAction }) {
  const item = detail?.item || {}
  const evidenceCount = Array.isArray(detail?.evidence_refs) ? detail.evidence_refs.length : 0
  // 已被处理过的候选不再给操作：接纳后 review_state 变 accepted，否决后变 dismissed
  const settled = String(item.review_state || '') !== 'pending_review'

  return (
    <article className="sa-list-card" style={{ display: 'grid', gap: 6 }}>
      <div style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.6 }}>{item.title || '(无标题候选)'}</div>
      {item.body ? (
        <div style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{item.body}</div>
      ) : null}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <span className="sa-chip">Evidence {evidenceCount}</span>
        {item.source_answer_id ? <span className="sa-chip">来源回答 {item.source_answer_id}</span> : null}
        {item.possible_duplicate ? (
          <span className="sa-chip" style={{ background: 'var(--sa-warning-soft, #fef3c7)' }}>疑似与已有记忆重复</span>
        ) : null}
      </div>
      {settled ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>这条已处理过，去记忆页查看当前状态。</p>
      ) : (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button type="button" className="sa-button" disabled={busy} onClick={() => onAction(() => acceptDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '纳入失败')}>
            纳入
          </button>
          <button type="button" onClick={() => onOpen(item.memory_id)} className="sa-button sa-button--ghost" style={{ display: 'inline-flex', alignItems: 'center' }}>
            编辑后纳入
          </button>
          <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => onAction(() => dismissDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '忽略失败')}>
            忽略
          </button>
        </div>
      )}
    </article>
  )
}

// 候选列表：按 memory_id 逐张摆开
function CandidateList({ details, projectId, onOpen, busy, onAction }) {
  return details.map((detail) => (
    <CandidateCard
      key={detail?.item?.memory_id}
      detail={detail}
      projectId={projectId}
      onOpen={onOpen}
      busy={busy}
      onAction={onAction}
    />
  ))
}

// 候选区标题：数量 + 一句身份说明，避免用户以为候选已经生效
function CandidatesHeader({ count }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'baseline' }}>
      <p className="sa-section-title" style={{ margin: 0 }}>本轮候选（{count}）</p>
      <span style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>确认后才会进入记忆库并被 Agent 读到</span>
    </div>
  )
}

/**
 * 回答卡里的候选区：本轮写回产出的待审候选，就地纳入或忽略。
 *
 * 候选只在写回真的产出了 candidate_ids 时才渲染。评估过没产出、本档不评估、
 * 写回失败这三种情况由「本轮结果」那一栏说明，这里不摆空卡冒充有候选。
 *
 * @param {{ message: object }} props
 */
export function MessageMemoryCandidates({ message }) {
  const workspace = useWorkspaceStore()
  const projectId = String(workspace.projectId || '').trim()
  const writeback = describeMemoryWriteback(message?.memory_writeback, message?.persistence_status)
  const candidateIds = writeback?.candidateIds || []
  const [state, setState] = useState({ loading: false, error: '', details: [] })
  const [busy, setBusy] = useState(false)
  const [activeMemoryId, setActiveMemoryId] = useState(null)

  // 候选正文与 Evidence 数量不在 done 载荷里，按 id 逐条取详情
  const load = async () => {
    if (!projectId || !candidateIds.length) return
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const details = await Promise.all(
        candidateIds.map((memoryId) => getMemory({ projectId, memoryId })),
      )
      setState({ loading: false, error: '', details: details.filter(Boolean) })
    } catch (reason) {
      setState({ loading: false, error: getError(reason, '候选加载失败'), details: [] })
    }
  }

  useEffect(() => {
    load().catch(() => {})
  }, [projectId, candidateIds.join(',')])

  if (!projectId || !candidateIds.length) return null

  // 纳入/忽略后重取详情，卡上的状态跟着变，不靠前端猜结果
  const runAction = async (action, failMessage) => {
    if (busy) return
    setBusy(true)
    try {
      await action()
      await load()
    } catch (reason) {
      setState((current) => ({ ...current, error: getError(reason, failMessage) }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sa-stack" style={{ marginTop: 10, display: 'grid', gap: 8 }}>
      <CandidatesHeader count={candidateIds.length} />
      {state.error ? <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-danger)' }}>{state.error}</p> : null}
      {state.loading ? <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>候选加载中...</p> : null}
      <CandidateList details={state.details} projectId={projectId} onOpen={setActiveMemoryId} busy={busy} onAction={runAction} />
      {activeMemoryId ? <MemoryDetailDrawer key={activeMemoryId} projectId={projectId} memoryId={activeMemoryId} onClose={() => setActiveMemoryId(null)} onChanged={load} /> : null}
    </div>
  )
}
