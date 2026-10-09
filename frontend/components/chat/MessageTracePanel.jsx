'use client'

import { useState } from 'react'
import { useAuthStore } from '../../stores/auth-store'

const DEVELOPER_ROLES = new Set(['developer', 'admin'])

function TraceField({ label, value }) {
  if (value === null || value === undefined || value === '') return null
  const display = typeof value === 'object' ? JSON.stringify(value) : String(value)
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 12, lineHeight: '18px' }}>
      <span style={{ color: 'var(--sa-text-faint)', minWidth: 80, flexShrink: 0 }}>{label}</span>
      <span style={{ wordBreak: 'break-all' }}>{display}</span>
    </div>
  )
}

/**
 * 研究轨迹。embedded 时由答案主卡打开，默认展开内容。
 * @param {{ message: object, embedded?: boolean }} props
 */
export function MessageTracePanel({ message, embedded = false }) {
  const [open, setOpen] = useState(Boolean(embedded))
  const auth = useAuthStore()
  const isDeveloper = DEVELOPER_ROLES.has(auth.user?.role)
  const trace = message?.trace_summary
  const requestId = message?.request_id

  if (!trace && !requestId) return null

  const mode = trace?.mode || ''
  const graphMode = trace?.graph_mode || ''
  const dialogueMode = trace?.dialogue_mode || ''
  const runtimeChain = trace?.runtime_chain || message?.runtime_chain || ''
  const sourceStatus = trace?.source_status || {}
  const layerStatus = trace?.evidence_layer_status || {}
  const latency = trace?.latency_ms || {}
  const evidenceCount = trace?.evidence_count ?? 0
  const citationCount = trace?.citation_count ?? 0
  const modelCallCount = trace?.model_call_count ?? 0

  const fields = open ? (
    <div style={{ marginTop: embedded ? 0 : 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
      <TraceField label="证据" value={`${evidenceCount} 条线索，${citationCount} 条引用`} />
      <TraceField label="范围" value={[mode, graphMode, dialogueMode].filter(Boolean).join(' / ')} />
      {isDeveloper ? (
        <>
          <TraceField label="request_id" value={requestId || trace?.request_id} />
          <TraceField label="session_id" value={trace?.session_id} />
          <TraceField label="chain" value={runtimeChain} />
          <TraceField label="source_status" value={Object.entries(sourceStatus).map(([k, v]) => `${k}:${v}`).join(', ')} />
          <TraceField
            label="layers"
            value={Object.entries(layerStatus).map(([k, v]) => `${k}:${v?.called ? v?.status : 'not_called'}`).join(', ')}
          />
          <TraceField label="model_calls" value={modelCallCount} />
          <TraceField label="latency" value={Object.entries(latency).map(([k, v]) => `${k}:${v}ms`).join(', ')} />
        </>
      ) : null}
      {isDeveloper && requestId ? (
        <a
          href={`/observability/${requestId}`}
          style={{ fontSize: 11, color: 'var(--sa-link, #2563eb)', marginTop: 4 }}
        >
          查看完整链路
        </a>
      ) : null}
    </div>
  ) : null

  if (embedded) {
    return fields
  }

  return (
    <section style={{ marginTop: 10, border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 10, padding: 12, background: 'var(--sa-surface-soft, #fafafa)' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="sa-button sa-button--ghost"
        style={{ fontSize: 12, minHeight: 24, display: 'flex', alignItems: 'center', gap: 6 }}
      >
        {open ? '收起' : '展开'} 研究轨迹与证据
        {evidenceCount > 0 ? <span style={{ color: 'var(--sa-text-faint)' }}>{evidenceCount} 条证据</span> : null}
      </button>
      {fields}
    </section>
  )
}
