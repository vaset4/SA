'use client'

export function EmptyState({
  title,
  description = '',
  action = null,
}) {
  return (
    <div
      style={{
        border: '1px dashed var(--sa-border)',
        borderRadius: 14,
        background: 'var(--sa-surface-soft)',
        padding: 28,
        textAlign: 'center',
      }}
    >
      <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
      {description ? (
        <p style={{ margin: '10px auto 0', color: 'var(--sa-text-soft)', maxWidth: 560, lineHeight: 1.6, fontSize: 13 }}>
          {description}
        </p>
      ) : null}
      {action ? <div style={{ marginTop: 14 }}>{action}</div> : null}
    </div>
  )
}
