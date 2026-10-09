'use client'

import { ProjectScopePanel } from './ProjectScopePanel'
import { ProjectSwitcher } from './ProjectSwitcher'

function scopeLabel(scope) {
  const hasExplicitScope = Boolean(
    (scope?.paper_ids || []).length
    || (scope?.task_filters || []).length
    || (scope?.method_filters || []).length
    || scope?.year_from
    || scope?.year_to,
  )
  return hasExplicitScope ? '研究范围已收束' : '研究范围待细化'
}

function recentSessionLabel(session) {
  if (!session) return '还没有内部研究'
  return session.topic || '最近一次内部研究'
}

export function ProjectHeroPanel({ projectStore, onSelectProject = null }) {
  const project = projectStore.activeProject
  const summary = projectStore.summary || { totalPapers: 0, analyzedPapers: 0, pendingPapers: 0 }
  const readinessRatio = summary.totalPapers ? Math.round((summary.analyzedPapers / summary.totalPapers) * 100) : 0
  const latestSession = projectStore.latestSession

  return (
    <section
      className="sa-panel-card sa-section-card sa-section-card--accent"
      style={{ position: 'relative', overflow: 'hidden' }}
    >
      <div className="sa-detail-grid sa-detail-grid--split">
        <div className="sa-workbench__stack" style={{ minWidth: 0 }}>
          <div className="sa-section-card__header">
            <p className="sa-section-card__eyebrow">项目研究桌面</p>
            <div style={{ display: 'grid', gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: 'clamp(2rem, 3vw, 3rem)', lineHeight: 0.98, fontFamily: 'var(--sa-font-display)' }}>
                {project?.name || '当前项目'}
              </h2>
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.85, color: 'var(--sa-text-soft)', maxWidth: 760 }}>
                {project?.description || '项目页现在先承接研究材料、研究入口和项目动态，再决定下一步是补论文、补分析，还是继续内部研究。'}
              </p>
            </div>
          </div>

          <div className="sa-chip-list">
            <span className="sa-chip">{scopeLabel(projectStore.scope)}</span>
            <span className="sa-chip">最近一次内部研究 · {recentSessionLabel(latestSession)}</span>
            <span className="sa-chip">研究准备度 {readinessRatio}%</span>
          </div>

          <section className="sa-progress-accent-card">
            <h3 className="sa-progress-accent-title">研究准备度</h3>
            <p className="sa-note">材料、分析和检索状态决定当前是否适合继续内部研究。</p>
            <div className="sa-progress-accent-track" aria-label={`研究准备度 ${readinessRatio}%`}>
              <span style={{ width: `${readinessRatio}%` }} />
            </div>
          </section>

          <div className="sa-metric-grid">
            <article className="sa-metric-tile">
              <p className="sa-metric-tile__label">总论文数</p>
              <p className="sa-metric-tile__value">{summary.totalPapers}</p>
              <p className="sa-metric-tile__hint">先看项目范围里一共积累了多少材料。</p>
            </article>
            <article className="sa-metric-tile">
              <p className="sa-metric-tile__label">已完成分析</p>
              <p className="sa-metric-tile__value">{summary.analyzedPapers}</p>
              <p className="sa-metric-tile__hint">这些论文已经可以进入正式研究主线。</p>
            </article>
            <article className="sa-metric-tile">
              <p className="sa-metric-tile__label">待补分析</p>
              <p className="sa-metric-tile__value">{summary.pendingPapers}</p>
              <p className="sa-metric-tile__hint">这一项决定下一步更像补材料还是继续研究。</p>
            </article>
            <article className="sa-metric-tile">
              <p className="sa-metric-tile__label">最近一次内部研究</p>
              <p className="sa-metric-tile__value" style={{ fontSize: '1.25rem', lineHeight: 1.35 }}>
                {recentSessionLabel(latestSession)}
              </p>
              <p className="sa-metric-tile__hint">
                {latestSession?.updated_at || latestSession?.created_at ? '可以直接从中间工作区继续推进。' : '还没有研究记录，第一次研究建议先基于已分析论文发起。'}
              </p>
            </article>
          </div>

          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: '1px solid var(--sa-border)',
              background: 'rgba(255, 252, 248, 0.78)',
              display: 'grid',
              gap: 10,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <strong style={{ fontSize: 15 }}>当前研究范围</strong>
              <span className="sa-chip">论文范围 {projectStore.scope?.paper_ids?.length || 0} 篇</span>
            </div>
            <p className="sa-note">这里主要确认当前项目到底在研究什么、范围收得够不够窄，以及接下来内部研究会基于哪些论文继续推进。</p>
            <ProjectScopePanel scope={projectStore.scope} />
          </div>
        </div>

        <div className="sa-workbench__stack" style={{ minWidth: 0 }}>
          <section
            style={{
              padding: 18,
              borderRadius: 20,
              border: '1px solid var(--sa-border)',
              background: 'linear-gradient(180deg, rgba(255, 253, 249, 0.98), rgba(246, 238, 229, 0.9))',
              display: 'grid',
              gap: 16,
            }}
          >
            <div className="sa-section-card__header">
              <p className="sa-section-card__eyebrow">Workspace Focus</p>
              <h3 className="sa-section-card__title">这轮项目现在最适合做什么</h3>
              <p className="sa-section-card__description">
                如果 `已完成分析` 还不够，先补研究材料；如果已经具备基础材料，就直接从中间列发起内部研究。
              </p>
            </div>
            <div className="sa-detail-grid" style={{ gap: 12 }}>
              <article className="sa-list-card">
                <strong style={{ fontSize: 14 }}>研究主线</strong>
                <span className="sa-note">内部研究会把论文比较、追问和结论沉淀重新带回项目页。</span>
              </article>
              <article className="sa-list-card">
                <strong style={{ fontSize: 14 }}>补充动作</strong>
                <span className="sa-note">外部深度研究和图谱继续保留，但不再挤占第一页主动作。</span>
              </article>
            </div>
          </section>

          <ProjectSwitcher
            project={project}
            projects={projectStore.projects}
            onSelect={onSelectProject || projectStore.setActiveProjectId}
          />
        </div>
      </div>
    </section>
  )
}
