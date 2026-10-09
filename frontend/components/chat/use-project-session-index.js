'use client'

import { useEffect, useMemo, useState } from 'react'
import { getProjectSessions } from '../../lib/api/projects'
import { allProjectSessionIdSet } from '../../lib/chat-research-scope'

function indexSessions(projects, activeProjectId, activeSessions) {
  const indexed = {}
  for (const project of projects) indexed[project.project_id] = []
  if (activeProjectId) indexed[activeProjectId] = activeSessions
  return indexed
}

export function useProjectSessionIndex({ projects, activeProjectId, activeSessions }) {
  const [state, setState] = useState({ loading: false, error: '', sessionsByProject: {} })

  useEffect(() => {
    let active = true
    const items = Array.isArray(projects) ? projects : []
    if (!items.length) {
      setState({ loading: false, error: '', sessionsByProject: {} })
      return () => { active = false }
    }
    setState((current) => ({ ...current, loading: true, error: '' }))
    Promise.all(items.map((project) => getProjectSessions({ projectId: project.project_id })))
      .then((results) => {
        if (!active) return
        setState({
          loading: false,
          error: '',
          sessionsByProject: Object.fromEntries(items.map((project, index) => [project.project_id, results[index]?.sessions || []])),
        })
      })
      .catch((reason) => {
        if (!active) return
        setState({
          loading: false,
          error: reason instanceof Error ? reason.message : '项目会话归属索引加载失败',
          sessionsByProject: indexSessions(items, activeProjectId, activeSessions),
        })
      })
    return () => { active = false }
  }, [activeProjectId, activeSessions, projects])

  return useMemo(() => {
    const sessionsByProject = { ...state.sessionsByProject }
    if (activeProjectId) sessionsByProject[activeProjectId] = activeSessions || []
    return { ...state, linkedSessionIds: allProjectSessionIdSet(sessionsByProject) }
  }, [activeProjectId, activeSessions, state])
}
