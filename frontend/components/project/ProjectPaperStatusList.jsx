'use client'

import Link from 'next/link'
import { useMemo } from 'react'
import { PaperReadinessTimeline } from './PaperReadinessTimeline'

function isAnalysisReady(paper) {
  const status = String(paper?.analysis_status || '').trim()
  if (['analysis_ready', 'succeeded', 'ready', 'completed'].includes(status)) return true
  return Boolean(paper?.analysis?.problem || paper?.analysis?.method || paper?.analysis?.tldr)
}

function PaperRow({ paper, action, tone = 'default' }) {
  return (
    <article
      className="sa-list-card"
      style={tone === 'success' ? { background: 'rgba(240, 248, 244, 0.92)' } : undefined}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
        <strong style={{ fontSize: 14, lineHeight: 1.5 }}>{paper.title || paper.paper_id}</strong>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{paper.year || '年份待补'}</span>
      </div>
      <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
        {paper.analysis?.method || paper.analysis?.tldr || '当前还没有结构化分析，可先进入论文详情页补材料。'}
      </span>
      <PaperReadinessTimeline paper={paper} />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <Link href={`/papers/${encodeURIComponent(paper.paper_id)}`} className="sa-button sa-button--ghost">
          查看论文
        </Link>
        {action}
      </div>
    </article>
  )
}

export function ProjectPaperStatusList({ projectStore }) {
  const analyzedPapers = useMemo(
    () => projectStore.papers.filter((paper) => isAnalysisReady(paper)),
    [projectStore.papers],
  )
  const pendingPapers = useMemo(
    () => projectStore.papers.filter((paper) => !isAnalysisReady(paper)),
    [projectStore.papers],
  )

  return (
    <section className="sa-section-card">
      <div className="sa-section-card__header">
        <span className="sa-section-card__eyebrow">
          Paper Scope
        </span>
        <div>
          <h3 className="sa-section-card__title">论文范围与分析状态</h3>
          <p className="sa-section-card__description">
            左侧先判断哪些论文已经能进入正式研究范围，哪些还只是待分析材料。
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span className="sa-chip">已完成分析 {analyzedPapers.length}</span>
        <span className="sa-chip">待补分析 {pendingPapers.length}</span>
        <span className="sa-chip">总计 {projectStore.summary?.totalPapers || 0}</span>
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
        <strong style={{ fontSize: 14 }}>待分析论文</strong>
        {!pendingPapers.length ? (
          <p className="sa-note">
            当前没有待分析论文，正式研究范围可以继续往内部研究里推进。
          </p>
        ) : pendingPapers.map((paper) => (
          <PaperRow
            key={paper.paper_id}
            paper={paper}
            action={
              <span className="sa-chip">待补分析</span>
            }
          />
        ))}
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
        <strong style={{ fontSize: 14 }}>已完成分析</strong>
        {!analyzedPapers.length ? (
          <p className="sa-note">
            还没有论文进入正式研究范围。先补齐 `AI 分析`，项目页才会形成稳定研究材料区。
          </p>
        ) : analyzedPapers.map((paper) => (
          <PaperRow
            key={paper.paper_id}
            paper={paper}
            tone="success"
            action={
              <span className="sa-chip">可进入内部研究</span>
            }
          />
        ))}
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
        <strong style={{ fontSize: 14 }}>材料补充</strong>
        <p className="sa-note">
          这里不再直接做“绑定论文到项目”的操作。需要补材料时，先去知识库查看和整理，再回到这个项目页确认哪些论文已经准备好。
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link href={`/papers${projectStore.activeProjectId ? `?projectId=${encodeURIComponent(projectStore.activeProjectId)}` : ''}`} className="sa-button sa-button--ghost">
            去知识库看论文
          </Link>
          <span className="sa-chip">在知识库补材料</span>
        </div>
      </div>
    </section>
  )
}
