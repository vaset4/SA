import { requestJson } from './core'
import { normalizePaperCollection } from './papers'

function normalizeSessionPaperIds(session) {
  if (!session || typeof session !== 'object') return []
  if (Array.isArray(session.paper_ids)) return session.paper_ids.filter(Boolean)
  if (Array.isArray(session.paperIds)) return session.paperIds.filter(Boolean)
  return []
}

export function normalizeProjectSession(session) {
  if (!session || typeof session !== 'object') return null
  return {
    ...session,
    paper_ids: normalizeSessionPaperIds(session),
    topic: typeof session.topic === 'string' ? session.topic.trim() : '',
    preview: typeof session.preview === 'string' ? session.preview.trim() : '',
  }
}

export function normalizeProjectSessionCollection(sessions) {
  return Array.isArray(sessions)
    ? sessions.map((session) => normalizeProjectSession(session)).filter(Boolean)
    : []
}

export const listProjects = () =>
  requestJson({ path: '/api/v1/projects' })

export const createProject = ({ name, description = '', owner = 'system', defaultLibraryId = 'default' }) =>
  requestJson({
    path: '/api/v1/projects',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        description,
        owner,
        default_library_id: defaultLibraryId,
      }),
    },
  })

export const deleteProject = ({ projectId }) =>
  requestJson({
    path: `/api/v1/projects/${projectId}`,
    init: {
      method: 'DELETE',
    },
  }).catch((error) => {
    const message = error instanceof Error ? error.message : String(error || '')
    if (!/method not allowed|405/i.test(message)) throw error
    return requestJson({
      path: `/api/v1/projects/${projectId}/delete`,
      init: {
        method: 'POST',
      },
    })
  })

export const updateProject = ({ projectId, patch }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}`,
    init: {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  })

export const getProjectScope = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${projectId}/scope` })

export const updateProjectScope = ({ projectId, scope }) =>
  requestJson({
    path: `/api/v1/projects/${projectId}/scope`,
    init: {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scope),
    },
  })

export const getProjectPapers = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${projectId}/papers` })
    .then((data) => ({
      ...data,
      papers: normalizePaperCollection(data?.papers),
    }))

export const getProjectSessions = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${projectId}/sessions` })
    .then((data) => ({
      ...data,
      sessions: normalizeProjectSessionCollection(data?.sessions),
    }))

export const getProjectActivities = ({ projectId, limit = 50 }) =>
  requestJson({ path: `/api/v1/projects/${projectId}/activities?limit=${limit}` })
    .then((data) => ({
      ...data,
      activities: Array.isArray(data?.activities) ? data.activities : [],
    }))

export const addProjectPaper = ({ projectId, paperId, role = 'core', note = '' }) =>
  requestJson({
    path: `/api/v1/projects/${projectId}/papers`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_id: paperId, role, note }),
    },
  })

export const removeProjectPaper = ({ projectId, paperId }) =>
  requestJson({
    path: `/api/v1/projects/${projectId}/papers/${encodeURIComponent(paperId)}`,
    init: {
      method: 'DELETE',
    },
  })

export const attachProjectSession = ({ projectId, sessionId }) =>
  requestJson({
    path: `/api/v1/projects/${projectId}/sessions/${sessionId}`,
    init: {
      method: 'POST',
    },
  })
