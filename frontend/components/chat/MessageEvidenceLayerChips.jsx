'use client'

function hasObjectFields(raw) {
  return raw && typeof raw === 'object' && Object.keys(raw).length > 0
}

function textValue(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return `${value}`
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean).join('、')
  if (typeof value === 'object') {
    return textValue(value.label || value.title || value.reason || value.notes || value.name)
  }
  return ''
}

export function summarizeEvidenceLayers(raw) {
  const entries = Array.isArray(raw)
    ? raw.map((item, index) => [item?.layer || `layer-${index}`, item])
    : hasObjectFields(raw)
      ? Object.entries(raw)
      : []
  return entries.map(([key, value]) => evidenceLayerSummary(key, value)).filter(Boolean)
}

function evidenceLayerSummary(key, value) {
  if (typeof value === 'string') {
    const label = value.trim()
    return label ? { key, label } : null
  }
  if (!value || typeof value !== 'object') return null
  const layer = textValue(value.layer || key)
  const role = textValue(value.role)
  const output = textValue(value.output)
  const notes = textValue(value.notes)
  const label = [layer, role || output, notes].filter(Boolean).join('：')
  return label ? { key: layer || key, label } : null
}

function summarizeEvidenceLayerStatus(raw) {
  if (!hasObjectFields(raw)) return []
  return Object.entries(raw).map(([key, value]) => evidenceLayerStatusSummary(key, value)).filter(Boolean)
}

function evidenceLayerStatusSummary(key, value) {
  if (!value || typeof value !== 'object') return null
  const called = Boolean(value.called)
  const status = textValue(value.status) || (called ? 'called' : 'not_called')
  const label = `${formatEvidenceLayerName(key)}：${called ? status : '未调用'}`
  return { key, label, called }
}

function formatEvidenceLayerName(key) {
  const normalized = `${key || ''}`.trim().toLowerCase()
  if (normalized === 'milvus') return 'Milvus'
  if (normalized === 'lightrag') return 'LightRAG'
  if (normalized === 'neo4j') return 'Neo4j'
  if (normalized === 'planner') return 'Planner'
  if (normalized === 'inventory') return '清单'
  return key
}

function EvidenceChipGroup({ label, items, dimNotCalled = false }) {
  if (!items.length) return null
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
      <span style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>{label}</span>
      {items.map((item) => (
        <span
          key={item.key}
          className="sa-chip"
          style={{ fontSize: 12, opacity: dimNotCalled && !item.called ? 0.68 : 1 }}
        >
          {item.label}
        </span>
      ))}
    </div>
  )
}

export function MessageEvidenceLayerChips({ roles, status }) {
  const roleItems = summarizeEvidenceLayers(roles)
  const statusItems = summarizeEvidenceLayerStatus(status)
  if (!roleItems.length && !statusItems.length) return null
  return (
    <>
      <EvidenceChipGroup label="证据层角色" items={roleItems} />
      <EvidenceChipGroup label="本次调用" items={statusItems} dimNotCalled />
    </>
  )
}

