'use client'

import { ErrorState } from '../../components/shared/ErrorState'

export default function WorkspaceError({ error, reset }) {
  const message = error?.message || '当前工作区加载失败，请重试。'
  return (
    <main className="sa-workbench">
      <ErrorState title="工作区加载失败" description={message} onRetry={reset} />
    </main>
  )
}
