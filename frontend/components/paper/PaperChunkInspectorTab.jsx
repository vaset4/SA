'use client'

import { useEffect, useMemo, useState } from 'react'
import { Icon } from '@iconify/react'
import { Button } from '@heroui/react'
import { getChunkDetail, getPaperChunkAudit, getPaperChunkMap } from '../../lib/api/papers'

const INSPECTOR_TABS = [
  { id: 'overview', label: 'Overview', icon: 'lucide:gauge' },
  { id: 'map', label: 'Chunk Map', icon: 'lucide:git-branch' },
  { id: 'audit', label: 'Audit', icon: 'lucide:list-checks' },
]
const SEVERITIES = ['ALL', 'ERROR', 'WARNING', 'INFO']
const FULL_TEXT_LIMIT = 12000

function compactText(value, fallback = '-') {
  const text = String(value || '').trim()
  return text || fallback
}

function statusTone(status) {
  if (status === 'ok') return 'success'
  if (status === 'warning') return 'warning'
  if (status === 'failed') return 'danger'
  return 'neutral'
}

function severityColor(severity) {
  if (severity === 'ERROR') return 'var(--sa-danger)'
  if (severity === 'WARNING') return '#9a5c34'
  return 'var(--sa-text-soft)'
}

function softPanelStyle() {
  return {
    border: '1px solid var(--sa-border)',
    borderRadius: 14,
    background: 'rgba(255, 253, 249, 0.72)',
    padding: 14,
  }
}

function MetricTile({ label, value, tone = 'neutral' }) {
  return (
    <div style={{ ...softPanelStyle(), display: 'grid', gap: 4 }} data-tone={tone}>
      <span style={{ fontSize: 11, color: 'var(--sa-text-faint)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {label}
      </span>
      <strong style={{ fontSize: 20, color: tone === 'danger' ? 'var(--sa-danger)' : 'var(--sa-text)' }}>
        {value}
      </strong>
    </div>
  )
}

function InspectorTabs({ activeTab, onChange }) {
  return (
    <div className="sa-tab-strip" role="tablist" aria-label="Chunk Inspector views">
      {INSPECTOR_TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className="sa-tab-button"
          data-active={activeTab === tab.id}
          onClick={() => onChange(tab.id)}
          role="tab"
          aria-selected={activeTab === tab.id}
        >
          <Icon icon={tab.icon} width={16} height={16} aria-hidden="true" />
          {tab.label}
        </button>
      ))}
    </div>
  )
}

function OverviewView({ chunkMap, audit }) {
  const summary = audit?.summary || {}
  const stats = audit?.stats || {}
  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
        <MetricTile label="Vector Rows" value={chunkMap?.vector_row_count ?? 0} />
        <MetricTile label="Audit Status" value={compactText(summary.status, 'unknown')} tone={statusTone(summary.status)} />
        <MetricTile label="Errors" value={summary.error_count ?? 0} tone={summary.error_count ? 'danger' : 'neutral'} />
        <MetricTile label="Warnings" value={summary.warning_count ?? 0} tone={summary.warning_count ? 'warning' : 'neutral'} />
      </div>
      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
        <Distribution title="Block Types" items={stats.block_types} />
        <Distribution title="Parser Modes" items={stats.parser_modes} />
        <Distribution title="Counts" items={stats.counts} />
        <Distribution title="Page Range" items={stats.page_range} />
      </div>
      <div style={softPanelStyle()}>
        <div className="sa-chip-list">
          <span className="sa-chip">map_type={chunkMap?.map_type || 'metadata_grouped'}</span>
          <span className="sa-chip">tree_quality={chunkMap?.tree_quality || 'derived_from_section_path'}</span>
          <span className="sa-chip">paper_db={chunkMap?.paper?.exists ? 'present' : 'missing'}</span>
        </div>
      </div>
    </div>
  )
}

function Distribution({ title, items }) {
  const entries = Object.entries(items || {})
  return (
    <div style={{ ...softPanelStyle(), display: 'grid', gap: 8 }}>
      <strong style={{ fontSize: 13 }}>{title}</strong>
      {entries.length ? entries.map(([key, value]) => (
        <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, fontSize: 12 }}>
          <span style={{ color: 'var(--sa-text-soft)' }}>{key}</span>
          <span>{String(value)}</span>
        </div>
      )) : <span style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>No data</span>}
    </div>
  )
}

function ChunkMapView({ chunkMap, onSelectChunk, selectedChunkId }) {
  const sections = chunkMap?.sections || []
  if (!sections.length) {
    return <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>Milvus 当前没有可分组的 chunk row。</p>
  }
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {sections.map((section) => (
        <SectionGroup
          key={section.section_path || '(missing)'}
          section={section}
          onSelectChunk={onSelectChunk}
          selectedChunkId={selectedChunkId}
        />
      ))}
    </div>
  )
}

function sectionContainsChunk(section, chunkId) {
  if (!chunkId) return false
  return (section.parents || []).some((parent) => (
    (parent.chunks || []).some((chunk) => chunk.chunk_id === chunkId)
  ))
}

function SectionGroup({ section, onSelectChunk, selectedChunkId }) {
  const hasSelectedChunk = sectionContainsChunk(section, selectedChunkId)
  const [open, setOpen] = useState(hasSelectedChunk)
  useEffect(() => {
    if (hasSelectedChunk) setOpen(true)
  }, [hasSelectedChunk])

  return (
    <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)} style={softPanelStyle()}>
      <summary style={{ cursor: 'pointer', fontWeight: 700 }}>
        {compactText(section.section_path, '(missing section_path)')}
        <span style={{ marginLeft: 8, color: 'var(--sa-text-soft)', fontSize: 12 }}>
          {section.parent_count} parents / {section.child_count} chunks
        </span>
      </summary>
      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        {(section.parents || []).map((parent) => (
          <ParentGroup
            key={parent.parent_id || '(missing-parent)'}
            parent={parent}
            onSelectChunk={onSelectChunk}
            selectedChunkId={selectedChunkId}
          />
        ))}
      </div>
    </details>
  )
}

function ParentGroup({ parent, onSelectChunk, selectedChunkId }) {
  return (
    <div style={{ display: 'grid', gap: 8, paddingLeft: 10, borderLeft: '2px solid var(--sa-border)' }}>
      <div className="sa-chip-list">
        <span className="sa-chip">parent={compactText(parent.parent_id, '(missing)')}</span>
        <span className="sa-chip">{parent.child_count} child chunks</span>
        <span className="sa-chip">full_text={parent.full_text_length}</span>
        {!parent.full_text_consistent ? <span className="sa-chip" style={{ color: 'var(--sa-danger)' }}>inconsistent full_text</span> : null}
      </div>
      {(parent.chunks || []).map((chunk, index) => (
        <ChunkRow
          key={chunk.chunk_id || `${parent.parent_id || 'parent'}-${index}`}
          chunk={chunk}
          onSelectChunk={onSelectChunk}
          selected={chunk.chunk_id === selectedChunkId}
        />
      ))}
    </div>
  )
}

function ChunkRow({ chunk, onSelectChunk, selected = false }) {
  return (
    <button
      type="button"
      onClick={() => onSelectChunk(chunk.chunk_id)}
      className="sa-button sa-button--ghost"
      data-active={selected}
      style={{
        justifyContent: 'flex-start',
        minHeight: 0,
        padding: 10,
        textAlign: 'left',
        whiteSpace: 'normal',
        borderColor: selected ? 'var(--sa-border-strong)' : 'var(--sa-border)',
        background: selected ? 'rgba(42, 122, 138, 0.08)' : undefined,
      }}
    >
      <span style={{ display: 'grid', gap: 5 }}>
        <strong style={{ fontSize: 12 }}>{compactText(chunk.chunk_id, '(missing chunk_id)')}</strong>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
          p.{chunk.page_num || 0} / {compactText(chunk.block_type, 'missing block')} / {compactText(chunk.parser_mode, 'missing parser')}
        </span>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{compactText(chunk.display_preview, 'No display preview')}</span>
      </span>
    </button>
  )
}

function AuditView({ audit, severity, onSeverityChange, onSelectChunk }) {
  const issues = Array.isArray(audit?.issues) ? audit.issues : []
  const filtered = severity === 'ALL' ? issues : issues.filter((issue) => issue.severity === severity)
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div className="sa-chip-list" aria-label="Audit severity filters">
        {SEVERITIES.map((item) => (
          <button
            key={item}
            type="button"
            className="sa-button sa-button--ghost"
            onClick={() => onSeverityChange(item)}
            data-active={severity === item}
            style={{ minHeight: 32, fontSize: 12, borderColor: severity === item ? 'var(--sa-border-strong)' : 'var(--sa-border)' }}
          >
            {item}
          </button>
        ))}
      </div>
      {filtered.length ? filtered.map((issue) => (
        <AuditIssue key={issue.issue_id || issue.code} issue={issue} onSelectChunk={onSelectChunk} />
      )) : <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>当前筛选下没有 audit issue。</p>}
    </div>
  )
}

function AuditIssue({ issue, onSelectChunk }) {
  return (
    <div style={{ ...softPanelStyle(), display: 'grid', gap: 8 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Icon icon={issue.severity === 'ERROR' ? 'lucide:circle-alert' : 'lucide:info'} width={16} height={16} color={severityColor(issue.severity)} aria-hidden="true" />
        <strong style={{ color: severityColor(issue.severity), fontSize: 13 }}>{issue.severity}</strong>
        <code style={{ fontSize: 12 }}>{issue.code}</code>
      </div>
      <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>{issue.message}</p>
      <div className="sa-chip-list">
        {issue.chunk_id ? <button type="button" className="sa-button sa-button--ghost" style={{ minHeight: 28, fontSize: 11 }} onClick={() => onSelectChunk(issue.chunk_id)}>chunk={issue.chunk_id}</button> : null}
        {issue.parent_id ? <span className="sa-chip">parent={issue.parent_id}</span> : null}
        {issue.section_path ? <span className="sa-chip">{issue.section_path}</span> : null}
        {issue.page_num ? <span className="sa-chip">p.{issue.page_num}</span> : null}
      </div>
    </div>
  )
}

function ChunkDetailPanel({ detail, loading, error, onLoadFullText, fullTextLoading }) {
  if (loading) return <aside style={softPanelStyle()}>Chunk detail 加载中...</aside>
  if (error) return <aside style={softPanelStyle()} role="alert">{error}</aside>
  if (!detail) {
    return (
      <aside style={{ ...softPanelStyle(), display: 'grid', gap: 8 }}>
        <p className="sa-section-card__eyebrow">Chunk Detail</p>
        <p style={{ margin: 0, color: 'var(--sa-text-soft)', fontSize: 13 }}>
          从 Chunk Map 或 Audit issue 选择一个 chunk 后，这里会显示字段、长度和文本预览。
        </p>
      </aside>
    )
  }
  return (
    <aside style={{ ...softPanelStyle(), display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
        <div>
          <p className="sa-section-card__eyebrow">Chunk Detail</p>
          <h3 style={{ margin: 0, fontSize: 16 }}>{detail.chunk_id}</h3>
        </div>
        <Button size="sm" variant="flat" onPress={onLoadFullText} isDisabled={fullTextLoading}>
          {fullTextLoading ? '加载中' : '加载全文'}
        </Button>
      </div>
      <div className="sa-chip-list">
        <span className="sa-chip">paper={compactText(detail.paper_id)}</span>
        <span className="sa-chip">parent={compactText(detail.parent_id)}</span>
        <span className="sa-chip">section={compactText(detail.metadata?.section_path || detail.metadata?.section)}</span>
        <span className="sa-chip">p.{detail.metadata?.page_num || 0}</span>
      </div>
      {detail.missing_fields?.length ? <p role="alert" style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>Missing: {detail.missing_fields.join(', ')}</p> : null}
      {(detail.text_fields || []).map((field) => <TextField key={field.name} field={field} />)}
    </aside>
  )
}

function TextField({ field }) {
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12 }}>
        <strong>{field.name}</strong>
        <span style={{ color: 'var(--sa-text-soft)' }}>
          {field.returned_chars ? `${field.returned_chars}/` : ''}{field.total_chars || field.length} chars{field.truncated ? ' truncated' : ''}
        </span>
      </div>
      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, lineHeight: 1.65, color: 'var(--sa-text-soft)' }}>
        {field.text || field.preview || '(empty)'}
      </pre>
    </div>
  )
}

function ActiveInspectorView({ state }) {
  if (state.loading) {
    return <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>Chunk Inspector 加载中...</p>
  }
  if (state.activeTab === 'overview') {
    return <OverviewView chunkMap={state.chunkMap} audit={state.audit} />
  }
  if (state.activeTab === 'map') {
    return (
      <ChunkMapView
        chunkMap={state.chunkMap}
        onSelectChunk={state.selectChunk}
        selectedChunkId={state.selectedChunkId}
      />
    )
  }
  if (state.activeTab === 'audit') {
    return (
      <AuditView
        audit={state.audit}
        severity={state.severity}
        onSeverityChange={state.setSeverity}
        onSelectChunk={state.selectChunk}
      />
    )
  }
  return null
}

async function loadMapAndAudit({ paperId, setChunkMap, setAudit, setLoading, setError }) {
  if (!paperId) return
  setLoading(true)
  setError('')
  try {
    const [nextMap, nextAudit] = await Promise.all([getPaperChunkMap({ paperId }), getPaperChunkAudit({ paperId })])
    setChunkMap(nextMap)
    setAudit(nextAudit)
  } catch (reason) {
    setError(reason instanceof Error ? reason.message : 'Chunk Inspector 加载失败')
  } finally {
    setLoading(false)
  }
}

async function loadChunkPreview({ chunkId, setDetail, setDetailLoading, setDetailError }) {
  if (!chunkId) return
  setDetailLoading(true)
  setDetailError('')
  try {
    setDetail(await getChunkDetail({ chunkId }))
  } catch (reason) {
    setDetailError(reason instanceof Error ? reason.message : 'Chunk detail 加载失败')
  } finally {
    setDetailLoading(false)
  }
}

async function loadChunkFullText({ chunkId, setDetail, setFullTextLoading, setDetailError }) {
  if (!chunkId) return
  setFullTextLoading(true)
  setDetailError('')
  try {
    setDetail(await getChunkDetail({ chunkId, includeText: 'full', maxTextChars: FULL_TEXT_LIMIT }))
  } catch (reason) {
    setDetailError(reason instanceof Error ? reason.message : '全文加载失败')
  } finally {
    setFullTextLoading(false)
  }
}

function useChunkInspectorState(paperId) {
  const [activeTab, setActiveTab] = useState('overview')
  const [chunkMap, setChunkMap] = useState(null)
  const [audit, setAudit] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [severity, setSeverity] = useState('ALL')
  const [detail, setDetail] = useState(null)
  const [selectedChunkId, setSelectedChunkId] = useState('')
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [fullTextLoading, setFullTextLoading] = useState(false)

  const summary = useMemo(() => audit?.summary || {}, [audit])
  const loadInspector = async () => {
    await loadMapAndAudit({ paperId, setChunkMap, setAudit, setLoading, setError })
  }
  useEffect(() => {
    loadInspector()
  }, [paperId])
  const selectChunk = async (chunkId) => {
    setSelectedChunkId(chunkId)
    await loadChunkPreview({ chunkId, setDetail, setDetailLoading, setDetailError })
  }
  const loadFullText = async () => {
    await loadChunkFullText({ chunkId: detail?.chunk_id, setDetail, setFullTextLoading, setDetailError })
  }
  return {
    activeTab,
    audit,
    chunkMap,
    detail,
    detailError,
    detailLoading,
    error,
    fullTextLoading,
    loading,
    severity,
    selectedChunkId,
    setActiveTab,
    setSeverity,
    loadFullText,
    loadInspector,
    selectChunk,
    summary,
  }
}

export function PaperChunkInspectorTab({ paperId }) {
  const state = useChunkInspectorState(paperId)

  return (
    <section className="sa-section-card" style={{ gap: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'start' }}>
        <div className="sa-section-card__header">
          <p className="sa-section-card__eyebrow">Developer Diagnostics</p>
          <h2 className="sa-section-card__title">Chunk Inspector</h2>
          <p className="sa-section-card__description">
            这里按 Milvus v2 当前行展示 chunk map 与结构 audit；它是开发诊断入口，不代表语义章节树。
          </p>
        </div>
        <Button size="sm" variant="flat" onPress={state.loadInspector} isDisabled={state.loading} startContent={<Icon icon="lucide:refresh-cw" width={16} height={16} />}>
          {state.loading ? '刷新中' : '刷新'}
        </Button>
      </div>

      <InspectorTabs activeTab={state.activeTab} onChange={state.setActiveTab} />
      {state.error ? <p role="alert" style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 13 }}>{state.error}</p> : null}

      {state.summary.status ? (
        <div className="sa-chip-list">
          <span className="sa-chip">status={state.summary.status}</span>
          <span className="sa-chip">errors={state.summary.error_count || 0}</span>
          <span className="sa-chip">warnings={state.summary.warning_count || 0}</span>
        </div>
      ) : null}

      <div
        style={{
          display: 'grid',
          gap: 16,
          alignItems: 'start',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 360px), 1fr))',
        }}
      >
        <div style={{ minWidth: 0 }}>
          <ActiveInspectorView state={state} />
        </div>
        <div style={{ minWidth: 0, position: 'sticky', top: 88 }}>
          <ChunkDetailPanel
            detail={state.detail}
            loading={state.detailLoading}
            error={state.detailError}
            onLoadFullText={state.loadFullText}
            fullTextLoading={state.fullTextLoading}
          />
        </div>
      </div>
    </section>
  )
}
