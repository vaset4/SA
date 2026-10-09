'use client'

export function ConfirmDialog({
  open,
  title,
  description = '',
  confirmLabel = '确认',
  onConfirm,
  onCancel,
}) {
  if (!open) return null
  return (
    <div className="sa-dialog-backdrop" role="presentation">
      <section className="sa-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title">
        <h3 id="confirm-dialog-title" style={{ margin: 0, fontSize: 16 }}>{title}</h3>
        {description ? <p className="sa-note" style={{ margin: 0 }}>{description}</p> : null}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, flexWrap: 'wrap' }}>
          <button type="button" className="sa-button sa-button--ghost" onClick={onCancel}>取消</button>
          <button type="button" className="sa-button sa-button--primary" onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </section>
    </div>
  )
}
