'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useParams, useRouter } from 'next/navigation'
import { WorkspaceLayout } from '../../../../components/layout/WorkspaceLayout'
import { PageHeader } from '../../../../components/shared/PageHeader'
import { EmptyState } from '../../../../components/shared/EmptyState'
import { ProjectHeroPanel } from '../../../../components/project/ProjectHeroPanel'
import { ProjectPaperStatusList } from '../../../../components/project/ProjectPaperStatusList'
import { ProjectResearchDesk } from '../../../../components/project/ProjectResearchDesk'
import { ProjectActivityFeed } from '../../../../components/project/ProjectActivityFeed'
import { ProjectNotesPanel } from '../../../../components/project/ProjectNotesPanel'
import { ProjectMemoryPanel } from '../../../../components/project/ProjectMemoryPanel'
import { useProjectStore } from '../../../../stores/project-store'
import { updateProject } from '../../../../lib/api/projects'
import { notifyError } from '../../../../lib/notify'

const PROJECT_TABS = [
  { id: 'overview', label: '总览' },
  { id: 'papers', label: '论文' },
  { id: 'notes', label: '笔记' },
  { id: 'memory', label: '研究记忆' },
  { id: 'activity', label: '活动' },
]

function InlineEdit({ value, onSave, tag: Tag = 'h2', style }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value || '')

  if (!editing) {
    return (
      <Tag style={{ ...style, cursor: 'pointer' }} onClick={() => { setDraft(value || ''); setEditing(true) }} title="点击编辑">
        {value || '未命名'}
      </Tag>
    )
  }

  return (
    <input
      className="sa-input"
      style={{ ...style, minHeight: 36 }}
      value={draft}
      autoFocus
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => { if (draft !== value) onSave(draft); setEditing(false) }}
      onKeyDown={(e) => { if (e.key === 'Enter') { if (draft !== value) onSave(draft); setEditing(false) } if (e.key === 'Escape') setEditing(false) }}
    />
  )
}

export default function ProjectDetailPage() {
  const params = useParams()
  const router = useRouter()
  const projectStore = useProjectStore()
  const projectId = String(params?.projectId || '')
  const [refreshingProject, setRefreshingProject] = useState(false)
  const [refreshingActivities, setRefreshingActivities] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    if (!projectId || projectStore.activeProjectId === projectId) return
    projectStore.setActiveProjectId(projectId)
  }, [projectId, projectStore.activeProjectId, projectStore.setActiveProjectId])

  const project = useMemo(
    () => projectStore.projects.find((item) => item.project_id === projectId) || null,
    [projectId, projectStore.projects],
  )

  const handleSelectProject = (nextProjectId) => {
    if (!nextProjectId) { router.push('/projects'); return }
    projectStore.setActiveProjectId(nextProjectId)
    router.push(`/projects/${encodeURIComponent(nextProjectId)}`)
  }

  const handleRefreshProject = async () => {
    if (!projectId || refreshingProject) return
    setRefreshingProject(true)
    try { await projectStore.refreshProjectData(projectId) } finally { setRefreshingProject(false) }
  }

  const handleRefreshActivities = async () => {
    if (!projectId || refreshingActivities) return
    setRefreshingActivities(true)
    try { await projectStore.refreshActivities(projectId) } finally { setRefreshingActivities(false) }
  }

  const handleUpdateProject = async (field, value) => {
    try {
      await updateProject({ projectId, patch: { [field]: value } })
      await handleRefreshProject()
    } catch (error) {
      notifyError('项目保存失败', error?.message || '请稍后重试。')
    }
  }

  const tabContent = useMemo(() => {
    if (activeTab === 'papers') {
      return <ProjectPaperStatusList projectStore={projectStore} />
    }
    if (activeTab === 'notes') {
      return <ProjectNotesPanel projectStore={projectStore} />
    }
    if (activeTab === 'memory') {
      return <ProjectMemoryPanel projectStore={projectStore} />
    }
    if (activeTab === 'activity') {
      return (
        <ProjectActivityFeed
          activities={projectStore.activities}
          onRefresh={handleRefreshActivities}
          refreshing={refreshingActivities}
        />
      )
    }
    return (
      <div className="sa-detail-grid sa-detail-grid--split">
        <div className="sa-workbench__stack">
          <ProjectHeroPanel projectStore={projectStore} onSelectProject={handleSelectProject} />
          <ProjectResearchDesk projectStore={projectStore} />
        </div>
        <div className="sa-workbench__stack">
          <ProjectActivityFeed
            activities={projectStore.activities.slice(0, 5)}
            onRefresh={handleRefreshActivities}
            refreshing={refreshingActivities}
          />
        </div>
      </div>
    )
  }, [
    activeTab,
    handleRefreshActivities,
    handleSelectProject,
    projectStore,
    refreshingActivities,
  ])

  const content = !projectStore.loading && !project ? (
    <div className="sa-workbench">
      <EmptyState
        title="未找到项目"
        action={<Link href="/projects" className="sa-button sa-button--ghost">返回</Link>}
      />
    </div>
  ) : (
    <div className="sa-workbench">
      {projectStore.error ? (
        <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{projectStore.error}</p>
      ) : null}

      <section className="sa-section-card sa-section-card--accent">
        <div className="sa-tab-strip">
          {PROJECT_TABS.map((tab) => (
            <button key={tab.id} type="button" className="sa-tab-button" data-active={activeTab === tab.id} onClick={() => setActiveTab(tab.id)}>
              {tab.label}
            </button>
          ))}
        </div>
      </section>

      {tabContent}
    </div>
  )

  return (
    <WorkspaceLayout
      focusMode
      header={(
        <PageHeader
          eyebrow="Project"
          variant="workspace"
          title={(
            <InlineEdit
              value={project?.name}
              onSave={(v) => handleUpdateProject('name', v)}
              tag="span"
              style={{ margin: 0 }}
            />
          )}
          description={project?.description || '项目工作面'}
          actions={(
            <>
              <button type="button" className="sa-button sa-button--ghost" onClick={() => handleRefreshProject()} disabled={refreshingProject}>
                {refreshingProject ? '刷新中' : '刷新项目工作面'}
              </button>
              <Link href="/projects" className="sa-button sa-button--ghost">返回</Link>
            </>
          )}
        />
      )}
    >
      {content}
    </WorkspaceLayout>
  )
}
