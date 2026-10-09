'use client'

import { useEffect, useState } from 'react'
import { Icon } from '@iconify/react'
import { useParams } from 'next/navigation'
import { getRagTrace } from '../../../../lib/api/observability'
import { DeveloperAccessGate } from '../../../../components/auth/DeveloperAccessGate'

const STAGE_LABELS = {
  request_overview: '请求概览',
  routing: '路由 / 范围',
  retrieval: '检索',
  evidence: '证据整理',
  model_calls: '模型调用',
  generation: '生成',
}

function Badge({ children, color = '#6b7280' }) {
  return (
    <span style={{ display: 'inline-block', fontSize: 11, padding: '2px 8px', borderRadius: 10, background: `${color}18`, color, fontWeight: 500 }}>
      {children}
    </span>
  )
}

function KV({ label, value, mono }) {
  if (value === null || value === undefined || value === '') return null
  const display = typeof value === 'object' ? JSON.stringify(value) : String(value)
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 6, fontSize: 13, lineHeight: 1.5 }}>
      <span style={{ color: 'var(--sa-text-faint, #9ca3af)', minWidth: 130, flexShrink: 0 }}>{label}</span>
      <span style={{ wordBreak: 'break-word', fontFamily: mono ? 'monospace' : 'inherit', fontSize: mono ? 12 : 13 }}>{display}</span>
    </div>
  )
}

function SummaryCard({ summary, latency }) {
  if (!summary) return null
  const lat = latency || summary.latency_ms || {}
  const subLatencies = ['retrieval', 'rerank', 'grading', 'generation', 'evaluation']
    .map((k) => [k, lat?.[k]])
    .filter(([, v]) => typeof v === 'number' && v > 0)
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, padding: 16, background: 'var(--sa-surface, #f9fafb)', borderRadius: 10 }}>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>模式</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.mode || '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>运行链</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.runtime_chain || '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>图模式</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.graph_mode || '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>对话模式</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.dialogue_mode || '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>总耗时</div><div style={{ fontSize: 14, fontWeight: 500 }}>{lat?.total ?? '-'} ms</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>证据数</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.evidence_count ?? '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>引用数</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.citation_count ?? '-'}</div></div>
        <div><div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>模型调用</div><div style={{ fontSize: 14, fontWeight: 500 }}>{summary.model_call_count ?? '-'} 次</div></div>
      </div>
      {subLatencies.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8, fontSize: 11, color: 'var(--sa-text-faint)' }}>
          {subLatencies.map(([k, v]) => <span key={k}>{k}: {v}ms</span>)}
        </div>
      )}
    </div>
  )
}

function SourceStatusBar({ status }) {
  if (!status || typeof status !== 'object') return null
  const colors = { ok: '#10b981', unused: '#9ca3af', failed: '#ef4444', ready: '#10b981', disabled: '#d1d5db' }
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
      {Object.entries(status).map(([k, v]) => (
        <Badge key={k} color={colors[v] || '#6b7280'}>{k}: {v}</Badge>
      ))}
    </div>
  )
}

function EvidenceLayerStatusBar({ status }) {
  if (!status || typeof status !== 'object') return null
  const colors = { ok: '#10b981', not_called: '#9ca3af', failed: '#ef4444', insufficient: '#f59e0b' }
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
      {Object.entries(status).map(([k, v]) => {
        const called = Boolean(v?.called)
        const label = `${k}: ${called ? v?.status || 'called' : 'not_called'}`
        const color = colors[called ? v?.status : 'not_called'] || '#6b7280'
        return <Badge key={k} color={color}>{label}</Badge>
      })}
    </div>
  )
}

function StageCard({ stage, debugMap }) {
  const [open, setOpen] = useState(false)
  if (!stage) return null
  const label = STAGE_LABELS[stage.name] || stage.name
  const entries = Object.entries(stage).filter(([k]) => k !== 'name' && k !== 'calls')
  const hasContent = entries.length > 0 || stage.calls?.length > 0
  return (
    <div style={{ border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 10, marginBottom: 10, overflow: 'hidden' }}>
      <button
        type="button"
        onClick={() => hasContent && setOpen((v) => !v)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '12px 16px', background: open ? 'var(--sa-surface, #f9fafb)' : 'transparent', border: 'none', cursor: hasContent ? 'pointer' : 'default', fontSize: 14, fontWeight: 500, textAlign: 'left' }}
      >
        {hasContent ? (
          <Icon icon={open ? 'solar:alt-arrow-down-linear' : 'solar:alt-arrow-right-linear'} width={16} aria-hidden="true" />
        ) : (
          <span style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>·</span>
        )}
        {label}
        {stage.latency_ms != null && <Badge color="var(--sa-accent)">{stage.latency_ms}ms</Badge>}
        {stage.calls?.length > 0 && <Badge color="#8b5cf6">{stage.calls.length} 次调用</Badge>}
      </button>
      {open && (
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--sa-border, #e5e7eb)' }}>
          {entries.map(([k, v]) => <KV key={k} label={k} value={v} />)}
          {stage.calls?.length > 0 && (
            <div style={{ marginTop: 12 }}>
              {stage.calls.map((call, i) => (
                <ModelCallCard key={call.call_id || i} call={call} debug={debugMap?.[call.call_id]} index={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ModelCallCard({ call, debug, index }) {
  const hasDebug = !!(debug?.input_full || debug?.output_full)
  const [showFull, setShowFull] = useState(hasDebug)
  const inputFull = debug?.input_full
  const outputFull = debug?.output_full
  const outputText = showFull && outputFull ? outputFull : (call.output?.text_preview || '')
  return (
    <div style={{ border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 8, padding: 12, marginBottom: 8, background: 'var(--sa-surface, #f9fafb)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, flexWrap: 'wrap', gap: 6 }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          {index != null && <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>#{index}</span>}
          <Badge color="#8b5cf6">{call.node || 'unknown'}</Badge>
          <Badge color="#6366f1">{call.stage || '-'}</Badge>
          {call.cache_hit && <Badge color="#10b981">cache</Badge>}
          {call.retry_count > 0 && <Badge color="#f59e0b">retry {call.retry_count}</Badge>}
          {call.error && <Badge color="#ef4444">error</Badge>}
        </div>
        <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>{call.latency_ms}ms</span>
      </div>
      <div style={{ fontSize: 12, color: 'var(--sa-text-faint)', marginBottom: 6 }}>
        输入 {call.input?.total_chars ?? '?'} chars · 输出 {call.output?.char_count ?? '?'} chars
        {(inputFull || outputFull) && (
          <button type="button" onClick={() => setShowFull(v => !v)} style={{ marginLeft: 10, background: 'none', border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 4, padding: '1px 8px', fontSize: 11, cursor: 'pointer' }}>
            {showFull ? '收起完整 I/O' : '展开完整 I/O'}
          </button>
        )}
      </div>

      {showFull && inputFull?.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          {inputFull.map((msg, mi) => (
            <div key={mi} style={{ marginBottom: 6 }}>
              <Badge color={msg.role === 'system' ? '#6366f1' : msg.role === 'assistant' ? '#10b981' : '#f59e0b'}>{msg.role}</Badge>
              <pre style={{ fontSize: 11, marginTop: 4, padding: 10, background: '#fff', borderRadius: 6, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 300, overflow: 'auto', border: '1px solid var(--sa-border, #e5e7eb)', fontFamily: 'monospace' }}>{msg.content}</pre>
            </div>
          ))}
        </div>
      )}

      {!showFull && call.input?.messages?.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--sa-text-faint)', marginBottom: 6 }}>
          {call.input.messages.map((m, i) => (
            <span key={i} style={{ marginRight: 8 }}>{m.role}: {m.char_count} chars</span>
          ))}
        </div>
      )}

      {outputText && (
        <div>
          {showFull && <Badge color="#10b981">output</Badge>}
          <pre style={{ fontSize: 12, color: 'var(--sa-text, #374151)', marginTop: showFull ? 4 : 0, padding: 10, background: '#fff', borderRadius: 6, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: showFull ? 400 : 200, overflow: 'auto', border: '1px solid var(--sa-border, #e5e7eb)', fontFamily: showFull ? 'monospace' : 'inherit' }}>{outputText}</pre>
        </div>
      )}
    </div>
  )
}

// --- PLACEHOLDER_MAIN ---

function TraceDetailContent() {
  const params = useParams()
  const requestId = params?.requestId
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!requestId) return
    setLoading(true)
    getRagTrace(requestId)
      .then((d) => setData(d))
      .catch((e) => setError(e?.message || 'Failed to load trace'))
      .finally(() => setLoading(false))
  }, [requestId])

  function handleExport() {
    if (!data?.trace) return
    const blob = new Blob([JSON.stringify(data.trace, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `trace-${requestId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return <div style={{ padding: 24 }}>Loading...</div>
  if (error) return <div style={{ padding: 24, color: 'var(--sa-error, red)' }}>{error}</div>
  if (!data) return <div style={{ padding: 24 }}>No data</div>

  const trace = data.trace || {}
  const stages = trace.stages || []
  const debug = trace.debug || {}
  const summary = trace.summary || {}
  const debugMap = {}
  for (const item of debug.full_model_inputs_outputs || []) {
    if (item?.call_id) debugMap[item.call_id] = item
  }

  return (
    <div style={{ padding: 24, maxWidth: 960, overflowY: 'auto', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>Trace 详情</h2>
        <button type="button" className="sa-button" onClick={handleExport} style={{ fontSize: 12 }}>导出 JSON</button>
      </div>

      <div style={{ fontSize: 12, color: 'var(--sa-text-faint)', marginBottom: 16, display: 'flex', flexWrap: 'wrap', gap: '4px 16px' }}>
        <span>request: <code>{data.request_id?.slice(0, 12)}...</code></span>
        <span>session: <code>{data.session_id?.slice(0, 12)}...</code></span>
        {data.user_id && <span>user: <code>{data.user_id}</code></span>}
        <span>{data.created_at?.slice(0, 19)}</span>
      </div>

      <div style={{ fontSize: 14, padding: '10px 14px', background: 'var(--sa-surface, #f9fafb)', borderRadius: 8, marginBottom: 20, borderLeft: '3px solid var(--sa-primary, #6366f1)' }}>
        {data.query}
      </div>

      <SummaryCard summary={summary} latency={trace.latency_ms} />
      <SourceStatusBar status={summary.source_status} />
      <EvidenceLayerStatusBar status={summary.evidence_layer_status} />

      <h3 style={{ fontSize: 15, marginBottom: 10 }}>阶段</h3>
      {stages.map((stage, i) => <StageCard key={stage.name || i} stage={stage} debugMap={debugMap} />)}

      {debug.truncated && (
        <p style={{ fontSize: 12, color: 'var(--sa-warning, #d97706)', marginTop: 16 }}>完整 I/O 数据因体积超限被截断 ({debug.reason})</p>
      )}

      <details style={{ marginTop: 24 }}>
        <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 500 }}>原始 JSON</summary>
        <pre style={{ fontSize: 11, marginTop: 8, padding: 12, background: 'var(--sa-surface, #f9fafb)', borderRadius: 8, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 600, overflow: 'auto', border: '1px solid var(--sa-border, #e5e7eb)' }}>
          {JSON.stringify(trace, null, 2)}
        </pre>
      </details>
    </div>
  )
}

export default function TraceDetailPage() {
  return (
    <DeveloperAccessGate
      title="观测详情"
      description="查看单次检索与生成链路，仅 developer/admin 可访问。"
    >
      <TraceDetailContent />
    </DeveloperAccessGate>
  )
}
