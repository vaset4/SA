'use client'

import Link from 'next/link'
import { useEffect, useMemo } from 'react'
import { useSearchParams } from 'next/navigation'
import { ResearchSessionList } from '../../../components/research/ResearchSessionList'
import { ResearchLauncher } from '../../../components/research/ResearchLauncher'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'
import { externalResearchModeLabel, normalizeExternalResearchMode } from '../../../lib/external-research-mode'
import { formatResearchTraceDetail, formatResearchTraceTitle } from '../../../lib/research-trace'
import { useProjectStore } from '../../../stores/project-store'
import { useWorkspaceStore } from '../../../stores/workspace-store'

function modeLabel(mode) {
  return externalResearchModeLabel(normalizeExternalResearchMode(mode))
}

function formatTraceTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString('zh-CN')
}

function CurrentResearchPanel({ workspace }) {
  const trace = workspace.researchTrace

  if (!trace?.sessionId && !trace?.events?.length) return null

  const keyFindings = Array.isArray(trace.keyFindings) ? trace.keyFindings : []
  const events = Array.isArray(trace.events) ? trace.events.slice(-4).reverse() : []

  return (
    <section
      className="sa-section-card sa-section-card--compact sa-research-summary-card"
      style={{
        background: 'linear-gradient(145deg, rgba(217,244,242,0.5), rgba(255,255,255,0.96) 42%, rgba(217,164,98,0.14))',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <div>
            <h3 className="sa-section-card__title">当前研究</h3>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <StatusBadge tone={trace.status === 'failed' ? 'danger' : trace.status === 'completed' ? 'success' : 'warning'}>
            {trace.status || 'idle'}
          </StatusBadge>
          <StatusBadge tone="neutral">{modeLabel(trace.researchMode)}</StatusBadge>
          {trace.sessionId ? (
            <Link href={`/research/${encodeURIComponent(trace.sessionId)}`} className="sa-button sa-button--ghost">
              打开详情
            </Link>
          ) : null}
        </div>
      </div>

      <div className="sa-research-summary-card__stats">
        <span className="sa-chip">论文 {trace.paperIds?.length || 0}</span>
        <span className="sa-chip">证据 {trace.evidenceCount || 0}</span>
        <span className="sa-chip">发现 {keyFindings.length}</span>
      </div>

      {keyFindings.length ? (
        <div className="sa-research-summary-card__findings">
          {keyFindings.slice(0, 2).map((item) => (
            <div key={item} className="sa-list-card" style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.6 }}>
              {item}
            </div>
          ))}
        </div>
      ) : null}

      {events.length ? (
        <div className="sa-research-summary-card__events">
          <strong style={{ fontSize: 14 }}>最近轨迹</strong>
          <div style={{ display: 'grid', gap: 8 }}>
            {events.map((event, index) => (
              <article
                key={`${event.at || 'event'}-${index}`}
                className="sa-list-card"
                style={{ padding: 10, gap: 3 }}
              >
                <strong style={{ fontSize: 13 }}>{formatResearchTraceTitle(event)}</strong>
                <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                  {formatResearchTraceDetail(event)}
                </span>
                {event.at ? (
                  <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>{formatTraceTime(event.at)}</span>
                ) : null}
              </article>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}

export default function ResearchHomePage() {
  const searchParams = useSearchParams()
  const workspace = useWorkspaceStore()
  const projectStore = useProjectStore()
  const prefill = useMemo(() => ({
    prompt: String(searchParams?.get('prompt') || '').trim(),
    paperId: String(searchParams?.get('paperId') || '').trim(),
    projectId: String(searchParams?.get('projectId') || '').trim(),
    source: String(searchParams?.get('source') || '').trim(),
    focus: String(searchParams?.get('focus') || '').trim(),
  }), [searchParams])
  useEffect(() => {
    if (prefill.projectId && projectStore.activeProjectId !== prefill.projectId) {
      projectStore.setActiveProjectId(prefill.projectId)
    }
  }, [prefill.projectId, projectStore])
  const hasPrefill = Boolean(prefill.prompt || prefill.paperId || prefill.source)
  const headerTone = workspace.error
    ? 'danger'
    : workspace.busy
      ? 'warning'
      : hasPrefill
        ? 'neutral'
        : 'success'
  const headerLabel = workspace.error
    ? '研究异常'
    : workspace.busy
      ? '研究进行中'
      : hasPrefill
        ? '待启动'
        : '工作区已就绪'

  return (
    <div className="sa-research-page">
      <PageHeader
        eyebrow="External Research"
        variant="workspace"
        title="外部研究"
        status={(
          <>
            <StatusBadge tone={headerTone}>
              {headerLabel}
            </StatusBadge>
            <span className="sa-chip">模式 {workspace.researchTrace?.researchMode ? modeLabel(workspace.researchTrace.researchMode) : '外部研究'}</span>
            <span className="sa-chip">独立研究会话</span>
          </>
        )}
        actions={(
          <>
            <Link href="/chat" className="sa-button sa-button--ghost">
              去内部研究
            </Link>
            {workspace.researchTrace?.sessionId ? (
              <Link href={`/research/${encodeURIComponent(workspace.researchTrace.sessionId)}`} className="sa-button sa-button--ghost">
                查看当前研究记录
              </Link>
            ) : null}
            {hasPrefill ? (
              <Link href="/research" className="sa-button sa-button--ghost">
                清空预填
              </Link>
            ) : null}
          </>
        )}
      />

      <main className="sa-workbench sa-research-console">
        <div className="sa-research-console__hero sa-research-console__hero--single">
          <ResearchLauncher workspace={workspace} projectStore={projectStore} prefill={prefill} />
        </div>
        <div className="sa-workbench__grid sa-research-console__body">
          <CurrentResearchPanel workspace={workspace} />
          <ResearchSessionList
            projectStore={projectStore}
            activeSessionId={workspace.researchTrace?.sessionId || ''}
          />
        </div>
      </main>
    </div>
  )
}
