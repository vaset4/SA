'use client'

import { useState } from 'react'

import { EmptyState } from '../shared/EmptyState'
import { MemoryEvidenceBindForm } from './MemoryEvidenceBindForm'
import {
  MEMORY_STATE_CONFIRMED,
  MEMORY_STATE_PENDING,
  MEMORY_STATE_REBIND,
  MEMORY_STATE_INACTIVE,
  MEMORY_STATE_META,
  formatMaturity,
  formatMemoryType,
  formatStale,
  formatStatus,
  formatEvidenceMode,
  groupMemoryByState,
  pendingActionKind,
} from '../../lib/research-memory.mjs'
import { acceptDraft, dismissDraft } from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'

// 三档的呈现顺序：先看 Agent 现在真能读到什么，再看等我动手的，最后看已经失效的
const BOARD_STATES = [MEMORY_STATE_CONFIRMED, MEMORY_STATE_PENDING, MEMORY_STATE_REBIND]

// 一条记忆的状态徽标组：只显示这条记忆真实带着的状态值，不补「已引用」之类没发生的动作
function StateChips({ item }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      <span className="sa-chip">{formatMaturity(item.maturity)}</span>
      <span className="sa-chip">{formatMemoryType(item.memory_type)}</span>
      <span className="sa-chip">{formatStatus(item.status)}</span>
      <span className="sa-chip">{formatEvidenceMode(item.evidence_mode)}</span>
      {item.stale_state && item.stale_state !== 'fresh' ? (
        <span className="sa-chip" style={{ background: 'var(--sa-danger-soft, #fee2e2)', color: 'var(--sa-danger, #b91c1c)' }}>
          {formatStale(item.stale_state)}
        </span>
      ) : null}
      <span className="sa-chip">v{item.version}</span>
    </div>
  )
}

// 已确认档的一行：这一档就是 Agent 每轮读得到的记忆，点开进详情抽屉
function ConfirmedRow({ item, onOpen }) {
  return (
    <article className="sa-list-card" style={{ display: 'grid', gap: 6, cursor: 'pointer' }} onClick={onOpen}>
      <div style={{ fontSize: 14, lineHeight: 1.6, fontWeight: 500 }}>{item.title}</div>
      <StateChips item={item} />
    </article>
  )
}

// 待重新核对档的一行：把失效原因摊开，用户要靠它判断该重建还是该删
function RebindRow({ item, onOpen }) {
  const reasons = Array.isArray(item.stale_reasons) ? item.stale_reasons.filter(Boolean) : []
  return (
    <article className="sa-list-card" style={{ display: 'grid', gap: 6, cursor: 'pointer' }} onClick={onOpen}>
      <div style={{ fontSize: 14, lineHeight: 1.6 }}>{item.title}</div>
      <StateChips item={item} />
      {reasons.length ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>失效原因：{reasons.join('；')}</p>
      ) : null}
    </article>
  )
}

// AI 草稿的三个出口：直接纳入、进详情改完再纳入、忽略
function AcceptActions({ projectId, item, actions }) {
  const { busy, onAction, onOpen } = actions
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <button type="button" className="sa-button" disabled={busy} onClick={() => onAction(() => acceptDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '纳入失败')}>
        纳入
      </button>
      <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={onOpen}>
        编辑后纳入
      </button>
      <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => onAction(() => dismissDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version }), '忽略失败')}>
        忽略
      </button>
    </div>
  )
}

/**
 * 待审档的一行：按来源分两套操作。
 * AI 草稿给纳入/编辑后纳入/忽略；手写的文献背书草稿给挂证据入口。
 * @param {{ projectId: string, item: object, busy: boolean, onAction: Function, onOpen: Function, onRefresh: Function }} props
 */
function PendingRow({ projectId, item, actions }) {
  const { busy, onAction, onOpen, onRefresh } = actions
  const [binding, setBinding] = useState(false)
  const actionKind = pendingActionKind(item)

  return (
    <article className="sa-list-card" style={{ display: 'grid', gap: 8 }}>
      <div style={{ fontSize: 14, lineHeight: 1.6, fontWeight: 500 }}>{item.title}</div>
      <StateChips item={item} />
      {item.body ? (
        <div style={{ fontSize: 13, color: 'var(--sa-text-soft)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{item.body}</div>
      ) : null}

      {actionKind === 'accept_draft' ? <AcceptActions projectId={projectId} item={item} actions={actions} /> : null}

      {actionKind === 'bind_evidence' ? (
        <>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button type="button" className="sa-button" disabled={busy} onClick={() => setBinding((value) => !value)}>
              {binding ? '收起' : '挂证据'}
            </button>
            <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={onOpen}>
              打开详情
            </button>
          </div>
          {binding ? (
            <MemoryEvidenceBindForm
              projectId={projectId}
              item={item}
              onDone={() => { setBinding(false); onRefresh() }}
            />
          ) : null}
        </>
      ) : null}

      {actionKind === 'confirm' ? (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>这条还在草稿位，去详情里确认生效。</p>
          <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={onOpen}>
            打开详情
          </button>
        </div>
      ) : null}
    </article>
  )
}

// 档位切换条：每档带实时计数，用户不点进去就能看出哪档积压
function StateTabs({ groups, activeState, onSelect }) {
  return (
    <div className="sa-tab-strip" style={{ marginBottom: 10 }}>
      {BOARD_STATES.map((state) => (
        <button
          key={state}
          type="button"
          className="sa-tab-button"
          data-active={activeState === state}
          onClick={() => onSelect(state)}
        >
          {MEMORY_STATE_META[state].label}（{groups[state].length}）
        </button>
      ))}
    </div>
  )
}

// 按当前档位选行形态：待审档要操作，待重新核对档要失效原因，已确认档只需一行陈述
function StateRow({ activeState, item, projectId, actions }) {
  if (activeState === MEMORY_STATE_PENDING) {
    return <PendingRow projectId={projectId} item={item} actions={actions} />
  }
  if (activeState === MEMORY_STATE_REBIND) {
    return <RebindRow item={item} onOpen={actions.onOpen} />
  }
  return <ConfirmedRow item={item} onOpen={actions.onOpen} />
}

// 列表加载失败与操作失败分开报：一个是这批记忆没取到，一个是刚才那次纳入/忽略没成
function BoardNotices({ error, actionError, loading }) {
  return (
    <>
      {error ? <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{error}</p> : null}
      {actionError ? <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{actionError}</p> : null}
      {loading ? <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>加载中...</p> : null}
    </>
  )
}

// 一档的空态：说清这一档为空意味着什么，别让用户以为加载失败
const EMPTY_HINTS = {
  [MEMORY_STATE_CONFIRMED]: '还没有已确认的判断。Agent 本轮读不到任何项目历史认识。',
  [MEMORY_STATE_PENDING]: '没有等你处理的条目。',
  [MEMORY_STATE_REBIND]: '没有需要重新核对的条目，现有证据都还有效。',
}

/**
 * 记忆状态三分视图：已确认 / 待审核 / 待重新核对。
 *
 * 这三档走审核与新鲜度轴，回答的是「Agent 现在读得到什么、什么在等我、什么已经失效」；
 * 论证台账走成熟度轴，回答的是「研究进展到哪一步」。两个视图看同一批记忆，互不替代。
 *
 * @param {{ items: Array<object>, projectId: string, loading: boolean, error: string,
 *   onOpen: Function, onRefresh: Function }} props
 */
export function MemoryStateBoard({ items, projectId, loading, error, onOpen, onRefresh }) {
  const [activeState, setActiveState] = useState(MEMORY_STATE_CONFIRMED)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')

  const groups = groupMemoryByState(items || [])
  const inactiveCount = groups[MEMORY_STATE_INACTIVE].length
  const visible = groups[activeState] || []

  // 执行一次纳入/忽略，成功后让父组件重载列表 —— 分档结果直接取决于这些状态字段
  const runAction = async (action, failMessage) => {
    if (busy) return
    setBusy(true)
    setActionError('')
    try {
      await action()
      onRefresh()
    } catch (reason) {
      setActionError(getError(reason, failMessage))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <StateTabs groups={groups} activeState={activeState} onSelect={setActiveState} />

      <p style={{ margin: '0 0 10px', fontSize: 12, color: 'var(--sa-text-soft)' }}>{MEMORY_STATE_META[activeState].hint}</p>
      <BoardNotices error={error} actionError={actionError} loading={loading} />

      {!loading && !visible.length ? (
        <EmptyState title={`${MEMORY_STATE_META[activeState].label}为空`} description={EMPTY_HINTS[activeState]} />
      ) : null}

      {visible.length ? (
        <div style={{ display: 'grid', gap: 10 }}>
          {visible.map((item) => (
            <StateRow
              key={item.memory_id}
              activeState={activeState}
              item={item}
              projectId={projectId}
              actions={{ busy, onAction: runAction, onOpen: () => onOpen(item.memory_id), onRefresh }}
            />
          ))}
        </div>
      ) : null}

      {inactiveCount ? (
        <p style={{ margin: '12px 0 0', fontSize: 12, color: 'var(--sa-text-faint)' }}>
          另有 {inactiveCount} 条已否决或已归档，不参与上面三档。
        </p>
      ) : null}
    </>
  )
}
