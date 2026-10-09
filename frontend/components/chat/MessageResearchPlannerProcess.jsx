'use client'

import { useState } from 'react'

const STATUS_LABELS = {
  planned: '已规划',
  evidence_found: '已找到证据',
  inferred: '基于证据推断',
  insufficient_evidence: '证据不足',
  external_needed: '需要外部背景',
  blocked: '无法执行',
}

const CLAIM_LABELS = {
  paper_supported: '论文证据',
  evidence_grounded_inference: '证据推断',
  model_prior_hypothesis: '待验证假设',
  external_needed: '外部背景',
}

function safeAnchorSegment(value = '') {
  return String(value || '').replace(/[^A-Za-z0-9_.:-]/g, '_')
}

function textValue(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return `${value}`
  return ''
}

function normalizeUnits(plan, results) {
  const rawUnits = Array.isArray(plan?.units) ? plan.units : []
  const resultMap = new Map(
    (Array.isArray(results) ? results : [])
      .filter((item) => item && typeof item === 'object')
      .map((item) => [textValue(item.unit_id), item]),
  )
  return rawUnits
    .filter((unit) => unit && typeof unit === 'object' && textValue(unit.unit_id))
    .map((unit) => ({ unit, result: resultMap.get(textValue(unit.unit_id)) || null }))
}

function normalizeClaims(result) {
  const claims = Array.isArray(result?.claims) ? result.claims : []
  return claims
    .filter((claim) => claim && typeof claim === 'object' && textValue(claim.text))
    .slice(0, 3)
}

function normalizeEvidenceRecordIds(records) {
  if (!Array.isArray(records)) return new Set()
  return new Set(records.map((record) => textValue(record?.evidence_id)).filter(Boolean))
}

function normalizeEvidenceIds(raw, knownIds) {
  if (!Array.isArray(raw)) return []
  const ids = raw.map(textValue).filter(Boolean)
  const filtered = knownIds.size ? ids.filter((id) => knownIds.has(id)) : ids
  return [...new Set(filtered)]
}

function claimEvidenceIds(claim, knownIds) {
  return normalizeEvidenceIds([
    ...(Array.isArray(claim?.supporting_evidence_ids) ? claim.supporting_evidence_ids : []),
    ...(Array.isArray(claim?.related_evidence_ids) ? claim.related_evidence_ids : []),
  ], knownIds)
}

function evidenceAnchor(messageId, evidenceId) {
  return `#evidence-${safeAnchorSegment(messageId || 'message')}-${evidenceId}`
}

function statusLabel(status) {
  return STATUS_LABELS[textValue(status)] || ''
}

function claimLabel(type) {
  return CLAIM_LABELS[textValue(type)] || '研究结论'
}

function EvidenceLinks({ ids, messageId, prefix = '关联证据' }) {
  if (!ids.length) return null
  return (
    <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
      <span>{prefix}</span>
      {ids.map((evidenceId) => (
        <a key={evidenceId} href={evidenceAnchor(messageId, evidenceId)} className="sa-chip" style={{ fontSize: 11 }}>
          {evidenceId}
        </a>
      ))}
    </span>
  )
}

function PlannerClaimLine({ claim, knownEvidenceIds, messageId }) {
  const ids = claimEvidenceIds(claim, knownEvidenceIds)
  return (
    <div style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--sa-text-soft)' }}>
      {claimLabel(claim.claim_type)}：{claim.text}
      {ids.length ? <span> <EvidenceLinks ids={ids} messageId={messageId} prefix="证据" /></span> : null}
    </div>
  )
}

function PlannerUnitCard({ unit, result, knownEvidenceIds, messageId }) {
  const unitEvidenceIds = normalizeEvidenceIds(result?.evidence_ids, knownEvidenceIds)
  return (
    <div className="sa-tool-card" style={{ padding: 10 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 600 }}>{unit.goal || unit.research_action}</div>
        {statusLabel(result?.status || 'planned') ? (
          <span className="sa-chip" style={{ fontSize: 12 }}>
            {statusLabel(result?.status || 'planned')}
          </span>
        ) : null}
      </div>
      {unit.research_action ? (
        <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)' }}>{unit.research_action}</p>
      ) : null}
      {result?.unresolved_reason ? (
        <p style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--sa-text-faint)' }}>
          {result.unresolved_reason}
        </p>
      ) : null}
      {unitEvidenceIds.length ? (
        <p style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--sa-text-faint)' }}>
          <EvidenceLinks ids={unitEvidenceIds} messageId={messageId} />
        </p>
      ) : null}
      {normalizeClaims(result).map((claim) => (
        <PlannerClaimLine
          key={claim.claim_id || claim.text}
          claim={claim}
          knownEvidenceIds={knownEvidenceIds}
          messageId={messageId}
        />
      ))}
    </div>
  )
}

export function MessageResearchPlannerProcess({ plan, unitResults, coverageReview, evidenceRecords = [], messageId = '' }) {
  // 研究过程细节默认折叠：概览由上方 step 卡承担，units/claims 按需展开
  const [open, setOpen] = useState(false)
  const units = normalizeUnits(plan, unitResults)
  const knownEvidenceIds = normalizeEvidenceRecordIds(evidenceRecords)
  const coveredCount = Array.isArray(coverageReview?.covered_facet_ids)
    ? coverageReview.covered_facet_ids.length
    : 0

  if (!units.length) return null

  return (
    <div className="sa-stack" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>研究过程</p>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          {units.length} 个子问题
        </span>
        {coveredCount ? (
          <span className="sa-chip" style={{ fontSize: 12 }}>
            覆盖 {coveredCount} 个请求点
          </span>
        ) : null}
        <button
          type="button"
          className="sa-button sa-button--ghost"
          style={{ fontSize: 12, minHeight: 28 }}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? '收起细节' : `查看细节（${units.length}）`}
        </button>
      </div>
      {open ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {units.map(({ unit, result }) => (
            <PlannerUnitCard
              key={unit.unit_id}
              unit={unit}
              result={result}
              knownEvidenceIds={knownEvidenceIds}
              messageId={messageId}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}
