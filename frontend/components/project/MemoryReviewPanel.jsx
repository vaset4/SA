'use client'

import { useEffect, useState } from 'react'

import { EmptyState } from '../shared/EmptyState'
import { MemoryDetailDrawer } from './MemoryDetailDrawer'
import {
  formatMaturity,
  formatMemoryType,
  formatEvidenceMode,
} from '../../lib/research-memory.mjs'
import {
  listReviewQueue,
  acceptDraft,
  dismissDraft,
} from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'

// 低置信度阈值：低于此值高亮"建议重点核对"
const LOW_CONFIDENCE = 0.5

// 单条草稿的置信度徽标：低置信度换成醒目底色并提示重点核对
function ConfidenceBadge({ score }) {
  const value = typeof score === 'number' ? score : 0
  const low = value < LOW_CONFIDENCE
  return (
    <span
      className="sa-chip"
      style={low ? { background: 'var(--sa-danger-soft, #fee2e2)', color: 'var(--sa-danger, #b91c1c)' } : undefined}
      title={low ? '低置信度，建议重点核对' : '置信度'}
    >
      置信度 {(value * 100).toFixed(0)}%{low ? ' · 重点核对' : ''}
    </span>
  )
}

/** @param {object} props 待审条目及版本化动作 @returns {JSX.Element} 单条审核卡 */
function ReviewItem({ item, projectId, busy, runAction, onEdit }) {
  return (
            <article className="sa-list-card" style={{ display: 'grid', gap: 8 }}>
              <div style={{ fontSize: 14, fontWeight: 500, lineHeight: 1.6 }}>{item.title}</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <ConfidenceBadge score={item.confidence_score} />
                <span className="sa-chip">{formatMaturity(item.maturity)}</span>
                <span className="sa-chip">{formatMemoryType(item.memory_type)}</span>
                <span className="sa-chip">{formatEvidenceMode(item.evidence_mode)}</span>
                {item.possible_duplicate ? <span className="sa-chip" style={{ background: 'var(--sa-warning-soft, #fef3c7)' }}>疑似重复</span> : null}
              </div>
              {item.body ? <div style={{ fontSize: 13, color: 'var(--sa-text-soft)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{item.body}</div> : null}
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="button" className="sa-button" disabled={busy} onClick={() => runAction(() => acceptDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '接纳失败')}>接纳</button>
                <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => onEdit(item.memory_id)}>编辑后纳入</button>
                <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => runAction(() => dismissDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '否决失败')}>否决</button>
              </div>
            </article>
  )
}

/** @param {object} props 当前项目 @returns {JSX.Element} 待审列表与详情操作 */
export function MemoryReviewPanel({ projectStore }) {
  const projectId = projectStore.activeProjectId
  const [state, setState] = useState({ loading: true, error: '', items: [] })
  const [busy, setBusy] = useState(false)
  const [activeMemoryId, setActiveMemoryId] = useState(null)

  const refresh = async () => {
    if (!projectId) {
      setState({ loading: false, error: '', items: [] })
      return
    }
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const items = await listReviewQueue({ projectId })
      setState({ loading: false, error: '', items: items || [] })
    } catch (reason) {
      setState({ loading: false, error: getError(reason, '待审区加载失败'), items: [] })
    }
  }

  useEffect(() => {
    refresh().catch(() => {})
  }, [projectId])

  // 执行一次接纳/否决动作，成功后刷新待审区
  const runAction = async (action, failMessage) => {
    if (busy) return
    setBusy(true)
    try {
      await action()
      await refresh()
    } catch (reason) {
      setState((current) => ({ ...current, error: getError(reason, failMessage) }))
    } finally {
      setBusy(false)
    }
  }

  if (!projectId) {
    return <EmptyState title="请先选择项目" description="选择项目后可查看 AI 整理待审区。" />
  }

  return (
    <section className="sa-panel-card sa-section-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <p className="sa-section-card__eyebrow">AI Review</p>
          <h3 className="sa-section-card__title">AI 整理待审区</h3>
          <p className="sa-section-card__description">AI 草稿按置信度排序，低置信度高亮。接纳前可核对，不自动进正式库。</p>
        </div>
        <span className="sa-chip">待审 {state.items.length}</span>
      </div>

      {state.error ? <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{state.error}</p> : null}
      {state.loading ? <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>加载中...</p> : null}

      {!state.loading && !state.items.length ? (
        <EmptyState title="待审区为空" description="用知识页的「AI 整理」在圈定范围内生成草稿后，会出现在这里等待核对。" />
      ) : null}

      {state.items.length ? (
        <div style={{ display: 'grid', gap: 10 }}>
          {state.items.map((item) => (
            <ReviewItem key={item.memory_id} item={item} projectId={projectId} busy={busy} runAction={runAction} onEdit={setActiveMemoryId} />
          ))}
        </div>
      ) : null}
      {activeMemoryId ? <MemoryDetailDrawer key={activeMemoryId} projectId={projectId} memoryId={activeMemoryId} onClose={() => setActiveMemoryId(null)} onChanged={refresh} /> : null}
    </section>
  )
}
