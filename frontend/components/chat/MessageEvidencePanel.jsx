'use client'

import { useState } from 'react'
import { Icon } from '@iconify/react'
import { GraphEvidenceCard } from './GraphEvidenceCard'
import { evidenceSourceLabel } from '../../lib/evidence-source.mjs'
import { finalCitationRecords, hasFinalDelivery } from '../../lib/final-citations.mjs'

const EVIDENCE_LAYER = {
  DIRECT_TEXT: 'direct_text',
  GRAPH_HINT: 'graph_hint',
  STRUCTURE_HINT: 'structure_hint',
  METADATA: 'metadata',
}

function safeAnchorSegment(value = '') {
  return String(value || '').replace(/[^A-Za-z0-9_.:-]/g, '_')
}

/** @param {object} props 证据记录、展开状态与消息身份 @returns {JSX.Element} 保留来源范围的证据卡片 */
function EvidenceCard({ record, defaultOpen = false, messageId = '' }) {
  const [expanded, setExpanded] = useState(defaultOpen)
  const excerpt = record.excerpt || record.summary || ''
  const label = evidenceSourceLabel(record)
  const evidenceId = String(record.evidence_id || '').trim()
  const anchorId = evidenceId
    ? `evidence-${safeAnchorSegment(messageId || 'message')}-${evidenceId}`
    : undefined

  const handleCopy = () => {
    navigator.clipboard.writeText(excerpt).catch(() => {})
  }

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `evidence-${record.evidence_id || 'unknown'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div id={anchorId} className="sa-tool-card" style={{ padding: 12, scrollMarginTop: 96 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>
          {evidenceId ? <strong style={{ color: 'var(--sa-text)' }}>[{record.citation_alias || evidenceId}] </strong> : null}
          {record.paper_id || ''} {label ? `· ${label}` : ''}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button type="button" className="sa-button sa-button--ghost" style={{ fontSize: 11, padding: '2px 6px' }} onClick={handleCopy}>复制</button>
          <button type="button" className="sa-button sa-button--ghost" style={{ fontSize: 11, padding: '2px 6px' }} onClick={handleExport}>导出</button>
        </div>
      </div>
      {excerpt.length > 120 && !expanded ? (
        <p style={{ margin: '6px 0 0', cursor: 'pointer' }} onClick={() => setExpanded(true)}>
          {excerpt.slice(0, 120)}...
        </p>
      ) : (
        <p style={{ margin: '6px 0 0', whiteSpace: 'pre-wrap' }}>{excerpt || '无摘录'}</p>
      )}
    </div>
  )
}

function LayerSection({ title, records, defaultOpen = false, badge = null, messageId = '' }) {
  const [open, setOpen] = useState(defaultOpen)
  if (!records.length) return null

  return (
    <div style={{ marginTop: 8 }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="sa-button sa-button--ghost"
        style={{ fontSize: 13, minHeight: 28, display: 'flex', alignItems: 'center', gap: 6 }}
      >
        <Icon icon={open ? 'solar:alt-arrow-down-linear' : 'solar:alt-arrow-right-linear'} width={16} aria-hidden="true" />
        {title} ({records.length})
        {badge ? <span style={{ fontSize: 11, color: 'var(--sa-warning, #d97706)', marginLeft: 4 }}>{badge}</span> : null}
      </button>
      {open ? (
        <div className="sa-stack" style={{ marginTop: 6 }}>
          {records.map((rec, i) => (
            <EvidenceCard key={rec.evidence_id || i} record={rec} defaultOpen={defaultOpen && i < 3} messageId={messageId} />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function GraphHintSection({ preview = '', full = '' }) {
  const previewText = String(preview || '').trim()
  const fullText = String(full || previewText).trim()
  const [open, setOpen] = useState(Boolean(fullText && fullText !== previewText))
  if (!previewText && !fullText) return null
  const collapsedText = previewText || fullText.slice(0, 500)

  return (
    <div style={{ marginTop: 8 }}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="sa-button sa-button--ghost"
        style={{ fontSize: 13, minHeight: 28, display: 'flex', alignItems: 'center', gap: 6 }}
      >
        <Icon icon={open ? 'solar:alt-arrow-down-linear' : 'solar:alt-arrow-right-linear'} width={16} aria-hidden="true" />
        图谱线索
        <span style={{ fontSize: 11, color: 'var(--sa-warning, #d97706)', marginLeft: 4 }}>
          线索，不可作为事实引用
        </span>
      </button>
      <pre
        style={{
          margin: '6px 0 0',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontFamily: 'inherit',
          fontSize: 12,
          color: 'var(--sa-text-soft)',
          lineHeight: 1.6,
        }}
      >
        {open ? fullText : collapsedText}
      </pre>
    </div>
  )
}

function citationEvidenceId(item, index) {
  const current = String(item?.evidence_id || '').trim()
  return current || `E${index + 1}`
}

function normalizeCitationRecord(item, index) {
  return {
    ...item,
    evidence_id: citationEvidenceId(item, index),
  }
}

function UnifiedEvidenceView({ evidenceRecords, citations, missingEvidence, graphHintSection, graphHintFull, messageId = '' }) {
  const directText = evidenceRecords.filter((r) => r.layer === EVIDENCE_LAYER.DIRECT_TEXT)
  const graphHint = evidenceRecords.filter((r) => r.layer === EVIDENCE_LAYER.GRAPH_HINT)
  const structureHint = evidenceRecords.filter((r) => r.layer === EVIDENCE_LAYER.STRUCTURE_HINT)
  const metadata = evidenceRecords.filter((r) => r.layer === EVIDENCE_LAYER.METADATA)
  const citationRecords = citations.map(normalizeCitationRecord)

  return (
    <div className="sa-stack">
      {citationRecords.length ? (
        <div className="sa-stack">
          <p className="sa-section-title">原文引用</p>
          {citationRecords.map((item, index) => (
            <EvidenceCard
              key={`${item.evidence_id || item.paper_id || 'citation'}-${index}`}
              record={item}
              defaultOpen={false}
              messageId={messageId}
            />
          ))}
        </div>
      ) : null}
      {/* 原文证据默认折叠，避免回答落地后 7 条证据平铺占满主视图 */}
      <LayerSection title="原文证据" records={directText} defaultOpen={false} messageId={messageId} />
      <GraphHintSection preview={graphHintSection} full={graphHintFull} />
      <LayerSection title="图线索" records={graphHint} badge="线索，不可作为事实引用" messageId={messageId} />
      <LayerSection title="结构线索" records={structureHint} badge="待验证，不可作为事实引用" messageId={messageId} />
      <LayerSection title="清单" records={metadata} messageId={messageId} />
      {missingEvidence?.length ? (
        <div style={{ marginTop: 8 }}>
          <p className="sa-section-title" style={{ color: 'var(--sa-warning, #d97706)' }}>缺失证据</p>
          {missingEvidence.map((m, i) => (
            <div key={i} style={{ fontSize: 12, color: 'var(--sa-text-soft)', marginTop: 4 }}>
              {m.paper_id || '未知论文'} · {m.dimension || ''} · {m.reason || ''}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function LegacyEvidenceView({ citations, graphEvidence, paperSources, summary, evidenceSummary, graphHintSection, graphHintFull, messageId = '' }) {
  const citationRecords = citations.map(normalizeCitationRecord)
  return (
    <div className="sa-stack">
      {summary ? <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>{summary}</p> : null}
      {evidenceSummary ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
          直接证据 {evidenceSummary.has_direct_evidence ? '已命中' : '未命中'}，
          结构化引用 {evidenceSummary.citation_count ?? 0} 条。
        </p>
      ) : null}
      {paperSources.length ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
          论文来源: {paperSources.map((item) => item.paper_id).join(', ')}
        </p>
      ) : null}
      {citationRecords.length ? (
        <div className="sa-stack">
          <p className="sa-section-title">原文引用</p>
          {citationRecords.map((item, index) => (
            <EvidenceCard
              key={`${item.evidence_id || item.paper_id || 'citation'}-${index}`}
              record={item}
              defaultOpen={false}
              messageId={messageId}
            />
          ))}
        </div>
      ) : null}
      <GraphHintSection preview={graphHintSection} full={graphHintFull} />
      {graphEvidence.length ? (
        <div className="sa-stack">
          <p className="sa-section-title">Graph Evidence</p>
          {graphEvidence.map((item) => (
            <GraphEvidenceCard key={item.path_id} item={item} />
          ))}
        </div>
      ) : null}
    </div>
  )
}

/**
 * 证据面板。embedded 时由答案主卡工具条打开，默认直接展开内容、去掉外层重复卡片边框。
 * @param {{ message: object, embedded?: boolean }} props
 */
export function MessageEvidencePanel({ message, embedded = false }) {
  const evidenceRecords = Array.isArray(message?.evidence_records) ? message.evidence_records : []
  // 答案优先：独立挂载时默认折叠；主卡内打开时直接展示
  const [open, setOpen] = useState(Boolean(embedded))
  const missingEvidence = Array.isArray(message?.missing_evidence) ? message.missing_evidence : []
  const useUnified = evidenceRecords.length > 0

  const chunkCitations = Array.isArray(message?.chunk_citations) ? message.chunk_citations : []
  const answerCitations = Array.isArray(message?.citations) ? message.citations : []
  const citations = hasFinalDelivery(message)
    ? finalCitationRecords(message)
    : (chunkCitations.length ? chunkCitations : answerCitations)
  const graphEvidence = message?.graph_evidence || []
  const graphHintSection = message?.graph_hint_section || ''
  const graphHintFull = message?.graph_hint_full || ''
  const paperSources = message?.paper_sources || []
  const summary = message?.evidence_summary || ''
  const synthesis = message?.synthesis_state && typeof message.synthesis_state === 'object' ? message.synthesis_state : null
  const evidenceSummary = synthesis?.evidence_summary && typeof synthesis.evidence_summary === 'object'
    ? synthesis.evidence_summary
    : null

  const hasContent = useUnified || citations.length || graphEvidence.length || graphHintSection || graphHintFull || paperSources.length || summary || evidenceSummary
  if (!hasContent) return null

  const body = open ? (
    useUnified
      ? (
        <UnifiedEvidenceView
          evidenceRecords={evidenceRecords}
          citations={citations}
          missingEvidence={missingEvidence}
          graphHintSection={graphHintSection}
          graphHintFull={graphHintFull}
          messageId={message?.id}
        />
      )
      : (
        <LegacyEvidenceView
          citations={citations}
          graphEvidence={graphEvidence}
          paperSources={paperSources}
          summary={summary}
          evidenceSummary={evidenceSummary}
          graphHintSection={graphHintSection}
          graphHintFull={graphHintFull}
          messageId={message?.id}
        />
      )
  ) : null

  // 主卡内：不再套一层「展开证据」按钮卡
  if (embedded) {
    return <div className="sa-stack">{body}</div>
  }

  return (
    <section
      style={{
        marginTop: 10,
        border: '1px solid var(--sa-border)',
        borderRadius: 10,
        padding: 12,
        background: 'var(--sa-surface-soft)',
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="sa-button sa-button--ghost"
        style={{ minHeight: 32 }}
      >
        {open ? '隐藏证据' : `展开证据${useUnified ? ` (${evidenceRecords.length})` : ''}`}
      </button>
      {body}
    </section>
  )
}
