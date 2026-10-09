'use client'

import Link from 'next/link'
import { PageHeader } from '../shared/PageHeader'
import { StatusBadge } from '../shared/StatusBadge'

function shortSessionId(sessionId) {
  if (!sessionId) return '首轮提问后生成'
  return sessionId.length > 12 ? sessionId.slice(0, 12) : sessionId
}

function resolveScopeLabel(projectName, papers) {
  if (papers.length) return '对话论文优先'
  if (projectName) return '当前项目论文池'
  return '全局知识库'
}

export function ChatHeader({
  conversation,
  papers,
  busy,
  currentSessionId = '',
  projectName = '',
  projectId = '',
}) {
  const resolvedSessionId = currentSessionId || conversation?.sessionId || ''
  const title = conversation?.title || '新对话'
  const scopeLabel = resolveScopeLabel(projectName, papers)

  return (
    <>
      <PageHeader
        eyebrow="Dialogue"
        variant="workspace"
        title={title}
        status={(
          <StatusBadge tone={busy ? 'warning' : 'success'}>
            {busy ? '研究处理中' : '就绪'}
          </StatusBadge>
        )}
        actions={(
          <div className="sa-chip-list">
            <span className="sa-chip" title={resolvedSessionId || undefined}>会话 {shortSessionId(resolvedSessionId)}</span>
            <span className="sa-chip">当前范围：{scopeLabel}</span>
            <span className="sa-chip">关联论文：{papers.length} 篇</span>
            {projectName ? (
              <Link href={projectId ? `/projects/${encodeURIComponent(projectId)}` : '/projects'} className="sa-chip">
                当前项目：{projectName}
              </Link>
            ) : null}
          </div>
        )}
      />
    </>
  )
}
