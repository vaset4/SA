'use client'

import Link from 'next/link'
import { buildInternalResearchHref, buildResearchWorkspaceHref } from '../../lib/workspace-routes'

function buildResearchPrompt(projectStore) {
  const projectName = projectStore.activeProject?.name || '当前项目'
  const scopeCount = projectStore.summary?.analyzedPapers || 0
  return `请围绕项目《${projectName}》发起一次内部研究，优先基于当前已完成分析的 ${scopeCount} 篇论文，整理研究问题、方法差异和下一步验证方向。`
}

function latestPaperScope(session) {
  const paperIds = Array.isArray(session?.paper_ids) ? session.paper_ids.filter(Boolean) : []
  if (!paperIds.length) return '当前还没有明确挂到论文范围。'
  return `本次研究基于 ${paperIds.length} 篇论文。`
}

export function ProjectResearchDesk({ projectStore }) {
  const latestSession = projectStore.latestSession
  const readyCount = projectStore.summary?.analyzedPapers || 0
  const projectName = projectStore.activeProject?.name || '当前项目'
  const canStartResearch = Boolean(projectStore.activeProjectId && readyCount > 0)
  const startHref = buildInternalResearchHref({
    prompt: buildResearchPrompt(projectStore),
    mode: 'chat',
  })
  const externalHref = buildResearchWorkspaceHref({
    prompt: `请围绕项目《${projectName}》补充外部研究背景，优先说明最近值得关注的方法、代表论文和需要回到项目内继续验证的问题。`,
    projectId: projectStore.activeProjectId,
    source: 'project',
    focus: 'external',
  })

  return (
    <section className="sa-section-card sa-section-card--accent">
      <div className="sa-section-card__header">
        <span className="sa-section-card__eyebrow">
          Research Actions
        </span>
        <div>
          <h3 className="sa-section-card__title">研究推进</h3>
          <p className="sa-section-card__description">
            这里不再跳到统一研究工作区，而是直接决定是进入内部研究，还是先补一轮外部研究。
          </p>
        </div>
      </div>

      <section
        className="sa-list-card"
        style={{
          padding: 18,
          borderColor: 'color-mix(in srgb, var(--sa-accent) 22%, white)',
          background: 'linear-gradient(180deg, rgba(255, 251, 245, 0.98), rgba(222, 238, 232, 0.88))',
          gap: 12,
        }}
      >
        <div style={{ display: 'grid', gap: 6 }}>
          <strong style={{ fontSize: 18 }}>开始新的内部研究</strong>
          <span className="sa-note">项目主线仍然是内部研究，优先围绕当前项目内已经准备好的研究材料展开。</span>
        </div>
        <p className="sa-note">
          {canStartResearch
            ? `当前已有 ${readyCount} 篇论文具备正式研究材料，可以直接发起内部研究。`
            : '当前还没有足够的已分析论文。先补齐 AI 分析，再回来发起内部研究会更稳。'}
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {canStartResearch ? (
            <Link href={startHref} className="sa-button sa-button--primary">
              开始新的内部研究
            </Link>
          ) : (
            <Link href={`/papers${projectStore.activeProjectId ? `?projectId=${encodeURIComponent(projectStore.activeProjectId)}` : ''}`} className="sa-button sa-button--primary">
              先去准备论文材料
            </Link>
          )}
          <Link href={externalHref} className="sa-button sa-button--ghost">
            补一轮外部研究
          </Link>
        </div>
      </section>

      <section
        className="sa-list-card sa-progress-accent-card"
        style={{ padding: 18, gap: 12 }}
      >
        <div style={{ display: 'grid', gap: 6 }}>
          <strong style={{ fontSize: 16 }}>继续最近研究</strong>
          <span className="sa-note">这里优先回看最近一次研究基于哪些论文，以及当前已经走到了哪一步。</span>
        </div>
        {latestSession ? (
          <>
            <div style={{ display: 'grid', gap: 6 }}>
              <span style={{ fontSize: 14 }}>{latestSession.topic || '未命名研究'}</span>
              <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                {latestPaperScope(latestSession)}
              </span>
              <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                {Array.isArray(latestSession.key_findings) && latestSession.key_findings.length
                  ? latestSession.key_findings[0]
                  : latestSession.preview || '最近一次研究还没有结构化摘要。'}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Link href="/chat" className="sa-button sa-button--ghost">去内部研究页继续</Link>
              <span className="sa-chip">最近一次内部研究</span>
            </div>
          </>
        ) : (
          <p className="sa-note">
            还没有内部研究记录。第一次进入时，建议先把问题写清楚，再明确要基于哪些论文继续问。
          </p>
        )}
      </section>
    </section>
  )
}
