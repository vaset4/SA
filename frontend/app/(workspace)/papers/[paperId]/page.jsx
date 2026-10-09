'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { WorkspaceLayout } from '../../../../components/layout/WorkspaceLayout'
import { getPaperDetail, triggerPaperAnalysisTask } from '../../../../lib/api/papers'
import { PaperDetailHeader } from '../../../../components/paper/PaperDetailHeader'
import { PaperProcessingPanel } from '../../../../components/paper/PaperProcessingPanel'
import { PaperOverviewTab } from '../../../../components/paper/PaperOverviewTab'
import { PaperAnalysisTab } from '../../../../components/paper/PaperAnalysisTab'
import { PaperOriginalTab } from '../../../../components/paper/PaperOriginalTab'
import { PaperChunkInspectorTab } from '../../../../components/paper/PaperChunkInspectorTab'
import { useConversationStore } from '../../../../stores/conversation-store'
import { useProjectStore } from '../../../../stores/project-store'
import { useLibraryStore } from '../../../../stores/library-store'
import { getTask } from '../../../../lib/api/tasks'
import { notifyError } from '../../../../lib/notify'
import { buildResearchWorkspaceHref } from '../../../../lib/workspace-routes'

const TAB_OPTIONS = [
  { id: 'overview', label: '总览' },
  { id: 'analysis', label: 'AI 分析' },
  { id: 'original', label: '原文' },
  { id: 'inspector', label: 'Chunk Inspector' },
]
const ANALYSIS_POLL_INTERVAL_MS = 1200
const TERMINAL_TASK_STATUS = new Set(['completed', 'failed', 'cancelled'])

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export default function PaperDetailPage() {
  const params = useParams()
  const router = useRouter()
  const paperId = String(params?.paperId || '')
  const [paper, setPaper] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('overview')
  const [analysisBusy, setAnalysisBusy] = useState(false)
  const [analysisMessage, setAnalysisMessage] = useState('')

  const conversations = useConversationStore()
  const projectStore = useProjectStore()
  const libraryStore = useLibraryStore()

  const syncPaper = (nextPaper) => {
    setPaper(nextPaper)
    libraryStore.upsertPaper(nextPaper)
  }

  useEffect(() => {
    if (!paperId) return
    let active = true
    setLoading(true)
    setError('')
    getPaperDetail({ paperId })
      .then((data) => {
        if (!active) return
        syncPaper(data)
      })
      .catch((reason) => {
        if (!active) return
        setError(reason instanceof Error ? reason.message : '论文详情加载失败')
        setPaper(null)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [paperId])

  const addToProject = async (nextPaperId) => {
    if (!projectStore.activeProjectId || !nextPaperId) return
    await projectStore.addPaper(nextPaperId)
  }

  const ensureConversation = async (title = `论文问答 · ${paper?.title || paperId}`) => {
    if (conversations.activeId) return conversations.activeId
    return conversations.createNew(title)
  }

  const startChat = async () => {
    const targetConversationId = await ensureConversation(`论文问答 · ${paper?.title || paperId}`)
    router.push(`/chat/${targetConversationId}?paperId=${encodeURIComponent(paperId)}`)
  }

  const startResearch = () => {
    const prompt = `请围绕论文《${paper?.title || paperId}》发起一次内部研究，优先说明研究问题、核心方法、主要结论，以及这篇论文在当前项目里的作用。`
    router.push(buildResearchWorkspaceHref({
      prompt,
      paperId,
      projectId: projectStore.activeProjectId || '',
      source: 'paper',
      focus: 'internal',
    }))
  }

  const triggerAnalysis = async () => {
    if (!paperId || analysisBusy) return
    setAnalysisBusy(true)
    setAnalysisMessage('分析任务已提交')
    try {
      const task = await triggerPaperAnalysisTask({ paperId })
      const taskId = task?.task_id
      if (!taskId) throw new Error('分析任务未返回 task_id')
      setPaper((current) => (current ? { ...current, analysis_status: 'queued', analysis_error: '' } : current))
      setActiveTab('analysis')
      let latestTask = task
      while (!TERMINAL_TASK_STATUS.has(latestTask?.status)) {
        await sleep(ANALYSIS_POLL_INTERVAL_MS)
        latestTask = await getTask({ taskId })
        setAnalysisMessage(latestTask?.message || '分析任务执行中')
      }
      if (latestTask.status !== 'completed') {
        throw new Error(latestTask?.error || latestTask?.message || 'AI 分析任务失败')
      }
      const updated = await getPaperDetail({ paperId })
      syncPaper(updated)
      setAnalysisMessage('AI 分析已完成')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'AI 分析失败'
      setAnalysisMessage(message)
      notifyError('AI 分析失败', message)
    } finally {
      setAnalysisBusy(false)
    }
  }

  const content = (
    <div className="sa-workbench">
      {paper?.source === 'upload' ? <PaperProcessingPanel paperId={paperId}
        onPublished={async () => syncPaper(await getPaperDetail({ paperId }))} /> : null}
      <section className="sa-section-card" style={{ gap: 14 }}>
        <div className="sa-section-card__header">
          <p className="sa-section-card__eyebrow">Material Preparation</p>
          <h2 className="sa-section-card__title">先把论文讲清楚，再决定下一步动作</h2>
          <p className="sa-section-card__description">
            论文详情页现在只保留 `总览 / AI 分析 / 原文` 三层内容，先看研究问题、核心方法和主要结论，再决定要不要发起内部研究。
          </p>
        </div>

        <div className="sa-tab-strip" role="tablist" aria-label="论文详情标签">
        {TAB_OPTIONS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className="sa-tab-button"
            data-active={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
            role="tab"
            aria-selected={activeTab === tab.id}
          >
            {tab.label}
          </button>
        ))}
        </div>
      </section>

      {activeTab === 'overview' ? (
        <PaperOverviewTab
          paper={paper}
          paperId={paperId}
          loading={loading}
          error={error}
          onPaperUpdated={syncPaper}
        />
      ) : null}

      {activeTab === 'analysis' ? (
        <PaperAnalysisTab
          paper={paper}
          onAnalyze={triggerAnalysis}
          analyzing={analysisBusy}
          analysisMessage={analysisMessage}
        />
      ) : null}

      {activeTab === 'original' ? <PaperOriginalTab paper={paper} /> : null}

      {activeTab === 'inspector' ? <PaperChunkInspectorTab paperId={paperId} /> : null}
    </div>
  )

  return (
    <WorkspaceLayout
      focusMode
      contentClassName="sa-main-scroll--page"
      header={(
        <PaperDetailHeader
          paper={paper}
          loading={loading}
          projectName={projectStore.activeProject?.name || ''}
          onAddToProject={addToProject}
          onStartChat={startChat}
          onStartResearch={startResearch}
        />
      )}
    >
      {content}
    </WorkspaceLayout>
  )
}
