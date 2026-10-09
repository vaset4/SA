'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { Button } from '@heroui/react'
import { useConversationStore } from '../../stores/conversation-store'
import { getProjectPapers, getProjectSessions } from '../../lib/api/projects'
import { notifyError } from '../../lib/notify'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { EmptyState } from '../shared/EmptyState'
import { ErrorState } from '../shared/ErrorState'
import { PageHeader } from '../shared/PageHeader'
import { Skeleton } from '../shared/Skeleton'
import { StatusBadge } from '../shared/StatusBadge'

function ProjectMetric({ label, value, hint }) {
  return (
    <div className="sa-metric-tile">
      <p className="sa-metric-tile__label">{label}</p>
      <p className="sa-metric-tile__value">{value}</p>
      <p className="sa-metric-tile__hint">{hint}</p>
    </div>
  )
}

export function ProjectWorkspaceShell({ projectStore }) {
  const conversations = useConversationStore()
  const [projectMetrics, setProjectMetrics] = useState({})
  const [confirmProject, setConfirmProject] = useState(null)

  const conversationSessionIds = useMemo(
    () => new Set((conversations.conversations || []).map((item) => item.sessionId).filter(Boolean)),
    [conversations.conversations],
  )

  useEffect(() => {
    let active = true
    if (!projectStore.projects.length) {
      setProjectMetrics({})
      return () => {
        active = false
      }
    }

    Promise.all(
      projectStore.projects.map(async (project) => {
        const [paperResult, sessionResult] = await Promise.allSettled([
          getProjectPapers({ projectId: project.project_id }),
          getProjectSessions({ projectId: project.project_id }),
        ])
        const papers = paperResult.status === 'fulfilled' ? paperResult.value?.papers ?? [] : []
        const sessions = sessionResult.status === 'fulfilled' ? sessionResult.value?.sessions ?? [] : []
        const conversationCount = sessions.filter((session) => conversationSessionIds.has(session.session_id)).length
        return [
          project.project_id,
          {
            paperCount: papers.length,
            conversationCount,
            researchCount: Math.max(0, sessions.length - conversationCount),
          },
        ]
      }),
    ).then((entries) => {
      if (!active) return
      setProjectMetrics(Object.fromEntries(entries))
    })

    return () => {
      active = false
    }
  }, [conversationSessionIds, projectStore.projects])

  const handleConfirmDelete = async () => {
    if (!confirmProject) return
    const target = confirmProject
    setConfirmProject(null)
    try {
      await projectStore.delete(target.project_id)
    } catch (error) {
      notifyError('项目删除失败', error?.message || '请稍后重试。')
    }
  }

  return (
    <div style={{ minHeight: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      <ConfirmDialog
        open={Boolean(confirmProject)}
        title="删除项目"
        description={`确认删除项目「${confirmProject?.name || confirmProject?.project_id || ''}」吗？`}
        confirmLabel="删除"
        onConfirm={handleConfirmDelete}
        onCancel={() => setConfirmProject(null)}
      />
      <PageHeader
        eyebrow="Projects"
        variant="workspace"
        title="项目"
        description="项目页现在只保留两件事：整理研究主题，和进入具体项目工作面。列表页不再承担研究桌面或旧入口分发。"
        status={(
          <>
            <StatusBadge tone={projectStore.error ? 'danger' : 'neutral'}>
              {projectStore.error ? '状态异常' : '项目空间可用'}
            </StatusBadge>
            <span className="sa-chip">项目 {projectStore.projects.length}</span>
            <span className="sa-chip">当前项目 {projectStore.activeProject?.name || '未选择'}</span>
          </>
        )}
      />

      <main className="sa-workbench">
        <section className="sa-section-card sa-section-card--accent">
          <div className="sa-section-card__header">
            <span className="sa-section-card__eyebrow">Project Overview</span>
            <div>
              <h3 className="sa-section-card__title">先定研究主题，再进入项目工作面</h3>
              <p className="sa-section-card__description">
                这里不再放研究入口分发。先选好项目，再去项目详情页继续总览、论文、研究笔记和活动。
              </p>
            </div>
          </div>
          <div className="sa-metric-grid">
            <ProjectMetric label="项目数" value={projectStore.projects.length} hint="统一管理研究主题和材料范围。" />
            <ProjectMetric label="当前项目" value={projectStore.activeProject?.name || '未选择'} hint="切换后会影响默认研究范围。" />
            <ProjectMetric label="创建状态" value={projectStore.busy ? '进行中' : '待命'} hint="项目创建和删除都在这里处理。" />
          </div>
        </section>

        <section className="sa-workbench__grid" style={{ gridTemplateColumns: 'minmax(0, 1.45fr) minmax(320px, 0.75fr)' }}>
          <section className="sa-section-card">
            <div className="sa-section-card__header">
              <span className="sa-section-card__eyebrow">Project Index</span>
              <div>
                <h3 className="sa-section-card__title">项目列表</h3>
                <p className="sa-section-card__description">
                  保留项目卡片和进入详情的动作，删除旧的中间跳转型入口。
                </p>
              </div>
            </div>

            {projectStore.loading ? (
              <Skeleton rows={5} height={22} />
            ) : projectStore.error ? (
              <ErrorState
                title="项目列表加载失败"
                description={projectStore.error}
                onRetry={projectStore.refresh}
              />
            ) : !projectStore.projects.length ? (
              <EmptyState
                title="还没有项目"
                description="先创建一个项目，再把论文、对话和研究沉淀都挂到这个主题下面。"
              />
            ) : (
              <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
                {projectStore.projects.map((project) => {
                  const metrics = projectMetrics[project.project_id] || {}
                  const isActive = project.project_id === projectStore.activeProjectId
                  return (
                    <article
                      key={project.project_id}
                      className="sa-list-card"
                      style={isActive
                        ? {
                          gap: 12,
                          background: 'linear-gradient(145deg, rgba(222,238,232,0.72), rgba(255,255,255,0.96))',
                          borderColor: 'color-mix(in srgb, var(--sa-accent) 38%, white)',
                        }
                        : { gap: 12 }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
                        <div style={{ display: 'grid', gap: 6, minWidth: 0 }}>
                          <strong style={{ fontSize: 16, lineHeight: 1.5 }}>{project.name || project.project_id}</strong>
                          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                            {project.description || '暂无项目描述'}
                          </span>
                        </div>
                        {isActive ? <StatusBadge tone="success">当前项目</StatusBadge> : null}
                      </div>

                      <div className="sa-chip-list">
                        <span className="sa-chip">论文 {metrics.paperCount ?? 0}</span>
                        <span className="sa-chip">对话 {metrics.conversationCount ?? 0}</span>
                        <span className="sa-chip">研究 {metrics.researchCount ?? 0}</span>
                      </div>

                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <Button
                          size="sm"
                          color={isActive ? 'primary' : 'default'}
                          variant={isActive ? 'flat' : 'light'}
                          onPress={() => projectStore.setActiveProjectId(project.project_id)}
                        >
                          {isActive ? '当前项目' : '切换为当前'}
                        </Button>
                        <Link href={`/projects/${encodeURIComponent(project.project_id)}`} className="sa-button sa-button--ghost">
                          进入项目详情
                        </Link>
                        <Button
                          size="sm"
                          color="danger"
                          variant="light"
                          onPress={() => setConfirmProject(project)}
                          isDisabled={projectStore.busy}
                        >
                          删除项目
                        </Button>
                      </div>
                    </article>
                  )
                })}
              </div>
            )}
          </section>

          <section className="sa-section-card sa-section-card--accent">
            <div className="sa-section-card__header">
              <span className="sa-section-card__eyebrow">Create Project</span>
              <div>
                <h3 className="sa-section-card__title">新建项目</h3>
                <p className="sa-section-card__description">
                  这里先写清楚项目名和研究主题。创建后，论文范围、研究笔记和项目动态都在项目详情页继续展开。
                </p>
              </div>
            </div>
            <div className="sa-workbench__stack">
              <input
                value={projectStore.draftName}
                onChange={(event) => projectStore.setDraftName(event.target.value)}
                placeholder="例如：多模态 RAG 研究"
                className="sa-input"
              />
              <textarea
                value={projectStore.draftDescription}
                onChange={(event) => projectStore.setDraftDescription(event.target.value)}
                placeholder="用一句话写清这个项目现在要研究什么。"
                className="sa-textarea"
                style={{ minHeight: 120 }}
              />
              <div className="sa-chip-list">
                <span className="sa-chip">创建后可立即进入项目详情</span>
                <span className="sa-chip">后续论文和研究会继续沉淀到项目下</span>
              </div>
              <Button size="sm" color="primary" onPress={projectStore.create} isDisabled={!projectStore.draftName.trim() || projectStore.busy}>
                {projectStore.busy ? '处理中' : '创建项目'}
              </Button>
            </div>
          </section>
        </section>
      </main>
    </div>
  )
}
