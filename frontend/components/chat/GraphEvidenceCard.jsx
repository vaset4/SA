'use client'

export function GraphEvidenceCard({ item }) {
  if (!item) return null
  return (
    <article className="sa-tool-card" style={{ padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <strong>{item.path_type || 'support'}</strong>
        <span style={{ color: 'var(--sa-text-faint)', fontSize: 12 }}>
          confidence {(item.confidence ?? 0).toFixed(2)}
        </span>
      </div>
      <p style={{ margin: '8px 0', color: 'var(--sa-text-soft)' }}>{item.summary || '无摘要'}</p>
      <div style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>
        论文: {(item.support_paper_ids || []).join(', ') || '无'}
      </div>
    </article>
  )
}
