'use client'

import { MessageEvidenceLayerChips, summarizeEvidenceLayers } from './MessageEvidenceLayerChips'

function hasObjectFields(raw) {
  return raw && typeof raw === 'object' && Object.keys(raw).length > 0
}

function textValue(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return `${value}`
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean).join('、')
  if (typeof value === 'object') {
    return textValue(value.label || value.title || value.question || value.reason || value.notes || value.name)
  }
  return ''
}

function normalizeTextList(raw) {
  if (!Array.isArray(raw)) return []
  return raw.map(textValue).filter(Boolean)
}

function normalizeDialogueMode(value, fallback = '') {
  const normalized = `${value || fallback || ''}`.trim().toLowerCase()
  return ['chat', 'lookup', 'research'].includes(normalized) ? normalized : ''
}

function formatDialogueMode(mode) {
  if (mode === 'lookup') return '查证'
  if (mode === 'research') return '研究'
  if (mode === 'chat') return '对话'
  return ''
}

function formatSynthesisStatus(status) {
  const normalized = `${status || ''}`.trim().toLowerCase()
  if (normalized === 'ready') return '整理完成'
  if (normalized === 'insufficient') return '证据不足'
  return ''
}

export function normalizeSynthesisState(raw) {
  if (!hasObjectFields(raw)) return null
  const hasEvidenceSummary = hasObjectFields(raw.evidence_summary)
  const hasEvidenceLayers = summarizeEvidenceLayers(raw.evidence_layers).length > 0
  const hasDimensions = Boolean(
    raw.status ||
    hasEvidenceSummary ||
    hasEvidenceLayers ||
    normalizeTextList(raw.supported_dimensions).length ||
    normalizeTextList(raw.unsupported_dimensions).length ||
    normalizeTextList(raw.open_questions).length
  )
  return hasDimensions ? raw : null
}

export function MessageResearchSynthesisSummary({ synthesis, message, hasTrailingContent = false }) {
  if (!synthesis) return null
  const dialogueMode = normalizeDialogueMode(message?.dialogue_mode, synthesis?.dialogue_mode)
  const synthesisStatus = formatSynthesisStatus(synthesis?.status)
  const supportedDimensions = normalizeTextList(synthesis?.supported_dimensions)
  const unsupportedDimensions = normalizeTextList(synthesis?.unsupported_dimensions)
  const openQuestions = normalizeTextList(synthesis?.open_questions)
  const evidenceSummary = synthesis?.evidence_summary && typeof synthesis.evidence_summary === 'object'
    ? synthesis.evidence_summary
    : null

  return (
    <div className="sa-stack" style={{ marginBottom: hasTrailingContent ? 12 : 0 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>研究整理</p>
        {dialogueMode ? <span className="sa-chip" style={{ fontSize: 12 }}>{formatDialogueMode(dialogueMode)}</span> : null}
        {synthesisStatus ? <span className="sa-chip" style={{ fontSize: 12 }}>{synthesisStatus}</span> : null}
      </div>
      {evidenceSummary ? (
        <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>
          已整理 {evidenceSummary.paper_count ?? 0} 篇论文，结构化引用 {evidenceSummary.citation_count ?? 0} 条，
          当前置信度 {evidenceSummary.answer_confidence || '未标注'}。
        </p>
      ) : null}
      <MessageEvidenceLayerChips roles={synthesis?.evidence_layers} status={message?.trace_summary?.evidence_layer_status} />
      {supportedDimensions.length ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
          已覆盖维度: {supportedDimensions.join('、')}
        </p>
      ) : null}
      {unsupportedDimensions.length ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
          待补维度: {unsupportedDimensions.join('、')}
        </p>
      ) : null}
      {openQuestions.length ? (
        <div className="sa-stack" style={{ gap: 6 }}>
          {openQuestions.slice(0, 3).map((item, index) => (
            <p key={`${item}-${index}`} style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
              {item}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  )
}
