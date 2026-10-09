'use client'

import { useState } from 'react'
import { useWorkspaceStore } from '../../stores/workspace-store'

// planner 四阶段的固定顺序与用户可读标签；node 名与后端 _emit_rag_stage_event 对齐
const STEP_ORDER = [
  { node: 'planner_analyze', label: '分析问题' },
  { node: 'planner_validate', label: '校验计划' },
  { node: 'planner_execute', label: '检索证据' },
  { node: 'planner_synthesize', label: '综合答案' },
]

const STEP_INDEX = STEP_ORDER.reduce((acc, step, index) => {
  acc[step.node] = index
  return acc
}, {})

// 把阶段事件数组归约成「每个阶段的最新细节」，事件里带的 unit_count/evidence_count 等覆盖式合并
function reduceStageDetails(events) {
  const details = {}
  let reachedIndex = -1
  for (const event of Array.isArray(events) ? events : []) {
    if (!event || typeof event !== 'object') continue
    const node = event.stage || event.node
    if (!(node in STEP_INDEX)) continue
    reachedIndex = Math.max(reachedIndex, STEP_INDEX[node])
    details[node] = {
      ...details[node],
      detail: event.detail || details[node]?.detail || '',
      unit_count: event.unit_count ?? details[node]?.unit_count,
      evidence_count: event.evidence_count ?? details[node]?.evidence_count,
      claim_count: event.claim_count ?? details[node]?.claim_count,
      evidence_sufficient: event.evidence_sufficient ?? details[node]?.evidence_sufficient,
    }
  }
  return { details, reachedIndex }
}

function StepDetailChips({ node, detail }) {
  if (!detail) return null
  const chips = []
  if (node === 'planner_validate' && typeof detail.unit_count === 'number') {
    chips.push(`${detail.unit_count} 个研究单元`)
  }
  if (node === 'planner_execute') {
    if (typeof detail.evidence_count === 'number') chips.push(`证据 ${detail.evidence_count} 条`)
    if (typeof detail.claim_count === 'number') chips.push(`主张 ${detail.claim_count} 条`)
    if (typeof detail.evidence_sufficient === 'boolean') {
      chips.push(detail.evidence_sufficient ? '证据充分' : '证据待补')
    }
  }
  if (!chips.length) return null
  return (
    <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap', marginLeft: 8 }}>
      {chips.map((text, index) => (
        <span key={index} className="sa-chip" style={{ fontSize: 11 }}>{text}</span>
      ))}
    </span>
  )
}

function StepMarker({ state }) {
  // 已完成打勾、进行中转圈、未到置灰点
  const base = {
    width: 16,
    height: 16,
    borderRadius: '50%',
    flexShrink: 0,
    fontSize: 11,
    lineHeight: '16px',
    textAlign: 'center',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  }
  if (state === 'done') {
    return <span style={{ ...base, background: 'var(--sa-accent, #2563eb)', color: '#fff' }}>✓</span>
  }
  if (state === 'active') {
    return (
      <span
        aria-label="进行中"
        style={{
          ...base,
          border: '2px solid var(--sa-accent, #2563eb)',
          borderTopColor: 'transparent',
          boxSizing: 'border-box',
          animation: 'sa-step-spin 0.8s linear infinite',
        }}
      />
    )
  }
  return <span style={{ ...base, background: 'var(--sa-border, #e5e7eb)' }} />
}

// 决定某阶段相对当前进度的状态：running 时最高已达阶段为 active，其余已达为 done；completed 时全 done
function stepState(index, reachedIndex, running) {
  if (!running) return index <= reachedIndex ? 'done' : 'pending'
  // 运行中尚无事件：第一项视为 active（转圈），其余 pending
  if (reachedIndex < 0) return index === 0 ? 'active' : 'pending'
  if (index < reachedIndex) return 'done'
  if (index === reachedIndex) return 'active'
  return 'pending'
}

function StepList({ details, reachedIndex, running }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {STEP_ORDER.map((step, index) => {
        const state = stepState(index, reachedIndex, running)
        const detail = details[step.node]
        return (
          <div key={step.node} style={{ display: 'flex', alignItems: 'center', gap: 8, opacity: state === 'pending' ? 0.5 : 1 }}>
            <StepMarker state={state} />
            <span style={{ fontSize: 13, color: state === 'active' ? 'var(--sa-accent, #2563eb)' : 'var(--sa-text, #111)' }}>
              {detail?.detail || step.label}
            </span>
            <StepDetailChips node={step.node} detail={detail} />
          </div>
        )
      })}
    </div>
  )
}

/**
 * 研究步骤。forceExpand 用于答案主卡「过程」抽屉内强制展开。
 * 结束后默认不渲染在主卡表面（由 MessageAnswerShell 控制挂载）。
 * @param {{ message: object, forceExpand?: boolean }} props
 */
export function MessageResearchSteps({ message, forceExpand = false }) {
  const store = useWorkspaceStore()
  const running = Boolean(message?.pending)
  // 结束后默认折叠；forceExpand 时展开
  const [open, setOpen] = useState(false)
  // 运行中读 store 的 live events（响应式）；已落地读消息里的 research_steps 快照
  const events = running
    ? (store.researchTrace?.events || [])
    : (Array.isArray(message?.research_steps) ? message.research_steps : [])

  const { details, reachedIndex } = reduceStageDetails(events)
  // 运行中即便还没有任何阶段事件也展示骨架
  if (!running && reachedIndex < 0) return null

  const doneCount = running
    ? Math.max(0, reachedIndex)
    : (reachedIndex >= 0 ? reachedIndex + 1 : 0)
  const total = STEP_ORDER.length
  const expanded = running || forceExpand || open

  // 非 forceExpand 且结束后未展开：极简一行（兼容独立挂载）
  if (!running && !forceExpand && !open) {
    return (
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          研究完成 · {doneCount}/{total} 步
        </span>
        <button
          type="button"
          className="sa-button sa-button--ghost"
          style={{ fontSize: 12, minHeight: 28 }}
          onClick={() => setOpen(true)}
        >
          展开步骤
        </button>
      </div>
    )
  }

  return (
    <section
      style={{
        marginBottom: running ? 12 : 10,
        border: running ? '1px solid var(--sa-border, #e5e7eb)' : '0',
        borderRadius: running ? 10 : 0,
        padding: running ? 12 : 0,
        background: running ? 'var(--sa-surface-soft, #fafafa)' : 'transparent',
      }}
    >
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
        <p className="sa-section-title" style={{ margin: 0 }}>
          {running ? '研究过程（进行中）' : '步骤'}
        </p>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          {doneCount}/{total}
        </span>
        {!running && !forceExpand ? (
          <button
            type="button"
            className="sa-button sa-button--ghost"
            style={{ fontSize: 12, minHeight: 28 }}
            onClick={() => setOpen(false)}
          >
            收起
          </button>
        ) : null}
      </div>
      {expanded ? <StepList details={details} reachedIndex={reachedIndex} running={running} /> : null}
    </section>
  )
}
