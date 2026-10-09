'use client'

import { createContext, useContext, useEffect, useMemo, useReducer } from 'react'
import {
  addProjectPaper,
  attachProjectSession,
  createProject,
  deleteProject,
  getProjectActivities,
  getProjectPapers,
  getProjectScope,
  getProjectSessions,
  listProjects,
  removeProjectPaper,
} from '../lib/api/projects'
import { getError } from '../lib/study-assistant-workspace.mjs'

const initialState = {
  projects: [],
  activeProjectId: '',
  activeProject: null,
  scope: null,
  papers: [],
  sessions: [],
  activities: [],
  latestSession: null,
  summary: {
    totalPapers: 0,
    analyzedPapers: 0,
    pendingPapers: 0,
  },
  loading: true,
  error: '',
  draftName: '',
  draftDescription: '',
  busy: false,
}

function reducer(state, action) {
  switch (action.type) {
    case 'setState':
      return { ...state, ...action.patch }
    default:
      return state
  }
}

function getPaperAnalysisStatus(paper) {
  const status = String(paper?.analysis_status || '').trim()
  if (status) return status
  return paper?.analysis?.problem || paper?.analysis?.method || paper?.analysis?.tldr ? 'analysis_ready' : 'not_started'
}

function isAnalysisReady(paper) {
  return ['analysis_ready', 'succeeded', 'ready', 'completed'].includes(getPaperAnalysisStatus(paper))
}

function toTimestamp(value) {
  const time = new Date(value || 0).getTime()
  return Number.isFinite(time) ? time : 0
}

function formatSessionFinding(session) {
  if (Array.isArray(session?.key_findings) && session.key_findings.length) return String(session.key_findings[0]).trim()
  if (typeof session?.preview === 'string' && session.preview.trim()) return session.preview.trim()
  return '研究已发起，等待进一步沉淀。'
}

function buildProjectSummary(papers) {
  const totalPapers = papers.length
  const analyzedPapers = papers.filter((paper) => isAnalysisReady(paper)).length
  return {
    totalPapers,
    analyzedPapers,
    pendingPapers: Math.max(totalPapers - analyzedPapers, 0),
  }
}

function pickLatestSession(sessions) {
  return [...sessions]
    .sort((left, right) => toTimestamp(right?.updated_at || right?.created_at) - toTimestamp(left?.updated_at || left?.created_at))[0] || null
}

function buildActivities({ papers, sessions }) {
  const paperActivities = papers.flatMap((paper) => {
    const paperId = String(paper?.paper_id || '').trim()
    const paperTitle = paper?.title || paperId || '未命名论文'
    const baseTime = paper?.updated_at || paper?.created_at || ''
    const events = []

    if (paperId) {
      events.push({
        id: `paper-added-${paperId}`,
        type: 'paper_added',
        label: '论文加入项目',
        title: paperTitle,
        description: `已纳入当前项目材料区，后续可继续补分析和研究笔记。`,
        at: baseTime,
      })
    }

    if (isAnalysisReady(paper)) {
      events.push({
        id: `paper-analysis-${paperId}`,
        type: 'analysis_completed',
        label: 'AI 分析完成',
        title: paperTitle,
        description: paper?.analysis?.tldr || paper?.analysis?.method || '论文已具备进入正式研究范围的基础材料。',
        at: paper?.analysis_updated_at || baseTime,
      })
    }

    return events
  })

  const sessionActivities = sessions.map((session) => {
    const paperIds = Array.isArray(session?.paper_ids) ? session.paper_ids.filter(Boolean) : []
    const scopeText = paperIds.length ? `基于 ${paperIds.length} 篇论文继续推进。` : '当前会话还没有明确挂到论文范围。'
    return {
      id: `internal-research-${session?.session_id || session?.conversation_id || Math.random()}`,
      type: 'internal_research_started',
      label: '内部研究启动',
      title: session?.topic || '未命名研究',
      description: `${scopeText} ${formatSessionFinding(session)}`.trim(),
      at: session?.updated_at || session?.created_at || '',
    }
  })

  return [...paperActivities, ...sessionActivities]
    .sort((left, right) => toTimestamp(right.at) - toTimestamp(left.at))
    .slice(0, 12)
}

function buildDerivedState({ scope, papers, sessions }) {
  return {
    scope,
    papers,
    sessions,
    summary: buildProjectSummary(papers),
    latestSession: pickLatestSession(sessions),
    activities: buildActivities({ papers, sessions }),
  }
}

const ProjectStoreContext = createContext(null)

async function loadProjectResources(activeProjectId) {
  if (!activeProjectId) {
    return buildDerivedState({ scope: null, papers: [], sessions: [] })
  }

  const [scopeResult, paperResult, sessionResult] = await Promise.allSettled([
    getProjectScope({ projectId: activeProjectId }),
    getProjectPapers({ projectId: activeProjectId }),
    getProjectSessions({ projectId: activeProjectId }),
  ])
  const activityResult = await Promise.allSettled([
    getProjectActivities({ projectId: activeProjectId }),
  ])
  const derived = buildDerivedState({
    scope: scopeResult.status === 'fulfilled' ? scopeResult.value : null,
    papers: paperResult.status === 'fulfilled' ? paperResult.value?.papers ?? [] : [],
    sessions: sessionResult.status === 'fulfilled' ? sessionResult.value?.sessions ?? [] : [],
  })

  return {
    ...derived,
    activities: activityResult[0]?.status === 'fulfilled' && Array.isArray(activityResult[0].value?.activities) && activityResult[0].value.activities.length
      ? activityResult[0].value.activities
      : derived.activities,
  }
}

async function loadProjectActivities(activeProjectId) {
  if (!activeProjectId) return []
  const data = await getProjectActivities({ projectId: activeProjectId })
  return Array.isArray(data?.activities) ? data.activities : []
}

export function ProjectStoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    let active = true
    listProjects()
      .then((data) => {
        if (!active) return
        const items = data?.projects ?? []
        const activeProjectId = items[0]?.project_id || ''
        dispatch({
          type: 'setState',
          patch: {
            projects: items,
            activeProjectId,
            activeProject: items.find((item) => item.project_id === activeProjectId) || null,
            loading: false,
            error: '',
          },
        })
      })
      .catch((reason) => {
        if (!active) return
        dispatch({ type: 'setState', patch: { loading: false, error: getError(reason, '项目列表加载失败') } })
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!state.activeProjectId) {
      dispatch({
        type: 'setState',
        patch: buildDerivedState({ scope: null, papers: [], sessions: [] }),
      })
      return
    }

    let active = true
    loadProjectResources(state.activeProjectId).then((resources) => {
      if (!active) return
      dispatch({ type: 'setState', patch: resources })
    })
    return () => {
      active = false
    }
  }, [state.activeProjectId])

  const actions = useMemo(() => ({
    setActiveProjectId(activeProjectId) {
      dispatch({
        type: 'setState',
        patch: {
          activeProjectId,
          activeProject: state.projects.find((item) => item.project_id === activeProjectId) || null,
        },
      })
    },
    setDraftName(draftName) {
      dispatch({ type: 'setState', patch: { draftName } })
    },
    setDraftDescription(draftDescription) {
      dispatch({ type: 'setState', patch: { draftDescription } })
    },
    async refresh() {
      const data = await listProjects()
      const items = data?.projects ?? []
      const activeProjectId = items.some((item) => item.project_id === state.activeProjectId)
        ? state.activeProjectId
        : items[0]?.project_id || ''
      const resources = await loadProjectResources(activeProjectId)
      dispatch({
        type: 'setState',
        patch: {
          projects: items,
          activeProjectId,
          activeProject: items.find((item) => item.project_id === activeProjectId) || null,
          error: '',
          ...resources,
        },
      })
    },
    async refreshProjectData(projectId = state.activeProjectId) {
      if (!projectId) return
      const resources = await loadProjectResources(projectId)
      dispatch({
        type: 'setState',
        patch: {
          activeProjectId: projectId,
          activeProject: state.projects.find((item) => item.project_id === projectId) || null,
          error: '',
          ...resources,
        },
      })
    },
    async refreshActivities(projectId = state.activeProjectId) {
      if (!projectId) return
      const activities = await loadProjectActivities(projectId)
      dispatch({
        type: 'setState',
        patch: {
          activities: activities.length ? activities : buildActivities({ papers: state.papers, sessions: state.sessions }),
        },
      })
    },
    async create() {
      if (!state.draftName.trim() || state.busy) return
      dispatch({ type: 'setState', patch: { busy: true, error: '' } })
      try {
        const project = await createProject({
          name: state.draftName.trim(),
          description: state.draftDescription.trim(),
        })
        const projects = [project, ...state.projects]
        dispatch({
          type: 'setState',
          patch: {
            busy: false,
            projects,
            activeProjectId: project.project_id,
            activeProject: project,
            ...buildDerivedState({ scope: null, papers: [], sessions: [] }),
            draftName: '',
            draftDescription: '',
          },
        })
      } catch (reason) {
        dispatch({ type: 'setState', patch: { busy: false, error: getError(reason, '项目创建失败') } })
      }
    },
    async delete(projectId) {
      const targetProjectId = String(projectId || '').trim()
      if (!targetProjectId || state.busy) return
      dispatch({ type: 'setState', patch: { busy: true, error: '' } })
      try {
        await deleteProject({ projectId: targetProjectId })
        const data = await listProjects()
        const items = data?.projects ?? []
        const nextActiveProjectId = items[0]?.project_id || ''
        const resources = nextActiveProjectId
          ? await loadProjectResources(nextActiveProjectId)
          : buildDerivedState({ scope: null, papers: [], sessions: [] })
        dispatch({
          type: 'setState',
          patch: {
            busy: false,
            projects: items,
            activeProjectId: nextActiveProjectId,
            activeProject: items.find((item) => item.project_id === nextActiveProjectId) || null,
            ...resources,
          },
        })
      } catch (reason) {
        const message = getError(reason, '项目删除失败')
        dispatch({ type: 'setState', patch: { busy: false, error: message } })
        throw new Error(message)
      }
    },
    async addPaper(paperId) {
      if (!state.activeProjectId || !paperId) return
      await addProjectPaper({ projectId: state.activeProjectId, paperId })
      const papers = await getProjectPapers({ projectId: state.activeProjectId })
      dispatch({
        type: 'setState',
        patch: buildDerivedState({
          scope: state.scope,
          papers: papers?.papers ?? [],
          sessions: state.sessions,
        }),
      })
    },
    async removePaper(paperId) {
      if (!state.activeProjectId || !paperId) return
      await removeProjectPaper({ projectId: state.activeProjectId, paperId })
      const papers = await getProjectPapers({ projectId: state.activeProjectId })
      dispatch({
        type: 'setState',
        patch: buildDerivedState({
          scope: state.scope,
          papers: papers?.papers ?? [],
          sessions: state.sessions,
        }),
      })
    },
    async attachSession(sessionId) {
      if (!state.activeProjectId || !sessionId) return
      if (state.sessions.some((item) => item.session_id === sessionId)) return
      await attachProjectSession({ projectId: state.activeProjectId, sessionId })
      const sessions = await getProjectSessions({ projectId: state.activeProjectId })
      dispatch({
        type: 'setState',
        patch: buildDerivedState({
          scope: state.scope,
          papers: state.papers,
          sessions: sessions?.sessions ?? [],
        }),
      })
    },
  }), [
    state.activeProjectId,
    state.busy,
    state.draftDescription,
    state.draftName,
    state.papers,
    state.projects,
    state.scope,
    state.sessions,
  ])

  const value = useMemo(() => ({ ...state, ...actions }), [actions, state])
  return <ProjectStoreContext.Provider value={value}>{children}</ProjectStoreContext.Provider>
}

export function useProjectStore() {
  const context = useContext(ProjectStoreContext)
  if (!context) throw new Error('useProjectStore must be used within ProjectStoreProvider')
  return context
}
