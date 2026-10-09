'use client'

import { EmptyState } from './EmptyState'

export function ErrorState({ title, description = '', onRetry = null }) {
  const action = onRetry ? (
    <button type="button" className="sa-button sa-button--ghost" onClick={onRetry}>
      重新加载
    </button>
  ) : null

  return (
    <div className="sa-error-state">
      <EmptyState title={title} description={description} action={action} />
    </div>
  )
}
