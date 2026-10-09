const UNASSIGNED_SCOPE_ID = 'unassigned'
const MAX_PROJECT_SCOPES = 8

function text(value) {
  return String(value || '').trim()
}

function normalizeSessionId(value) {
  return text(value)
}

function sessionUpdatedAt(session) {
  return session?.updated_at || session?.created_at || ''
}

function conversationUpdatedAt(conversation) {
  return conversation?.updatedAt || conversation?.createdAt || ''
}

function toTime(value) {
  const time = new Date(value || 0).getTime()
  return Number.isFinite(time) ? time : 0
}

function projectScopeItem(project, projectStore) {
  const projectId = text(project?.project_id)
  const activeProjectId = text(projectStore?.activeProjectId)
  const isActive = projectId === activeProjectId
  const metrics = projectStore?.projectMetrics?.[projectId] || {}
  const paperCount = isActive ? (projectStore?.summary?.totalPapers || 0) : (metrics.paperCount || 0)
  const sessionCount = isActive ? (projectStore?.sessions || []).length : (metrics.sessionCount || 0)
  return {
    id: projectId,
    type: 'project',
    title: project?.name || projectId || '未命名项目',
    description: `${paperCount} 篇论文 · ${sessionCount} 个会话`,
    active: projectId === activeProjectId,
  }
}

export function buildProjectScopeItems(projectStore) {
  const projects = Array.isArray(projectStore?.projects) ? projectStore.projects : []
  const activeProjectId = text(projectStore?.activeProjectId)
  const items = projects.slice(0, MAX_PROJECT_SCOPES).map((project) => projectScopeItem(project, projectStore))
  return [
    ...items,
    {
      id: UNASSIGNED_SCOPE_ID,
      type: 'unassigned',
      title: '未归属会话',
      description: '临时提问，可稍后绑定项目',
      active: !activeProjectId,
    },
  ]
}

export function projectSessionIdSet(projectSessions) {
  return new Set(
    (Array.isArray(projectSessions) ? projectSessions : [])
      .map((session) => normalizeSessionId(session?.session_id || session?.sessionId))
      .filter(Boolean),
  )
}

export function allProjectSessionIdSet(projectMetrics) {
  const result = new Set()
  Object.values(projectMetrics || {}).forEach((metrics) => {
    const sessions = Array.isArray(metrics) ? metrics : metrics?.sessions
    if (!Array.isArray(sessions)) return
    sessions.forEach((session) => {
      const sessionId = normalizeSessionId(session?.session_id || session?.sessionId)
      if (sessionId) result.add(sessionId)
    })
  })
  return result
}

export function sessionMatchesConversation(session, conversations) {
  const sessionId = normalizeSessionId(session?.session_id || session?.sessionId)
  if (!sessionId) return null
  return (Array.isArray(conversations) ? conversations : []).find((item) => item?.sessionId === sessionId) || null
}

export function buildProjectSessionItems({ projectSessions, conversations, activeConversationId }) {
  return (Array.isArray(projectSessions) ? projectSessions : []).map((session) => {
    const conversation = sessionMatchesConversation(session, conversations)
    const sessionId = normalizeSessionId(session?.session_id || session?.sessionId)
    return {
      id: conversation?.id || `session:${sessionId}`,
      type: 'project-session',
      title: conversation?.title || session?.topic || '未命名研究',
      description: conversation
        ? `${conversation.messageCount || 0} 条消息 · 已绑定当前项目`
        : `${(session?.paper_ids || []).length} 篇背景论文 · 报告可查看`,
      sessionId,
      conversationId: conversation?.id || '',
      active: Boolean(conversation?.id && conversation.id === activeConversationId),
      href: conversation?.id ? `/chat/${encodeURIComponent(conversation.id)}` : `/research/${encodeURIComponent(sessionId)}`,
      status: conversation?.id ? 'dialogue' : 'report',
      updatedAt: conversationUpdatedAt(conversation) || sessionUpdatedAt(session),
    }
  }).sort((left, right) => toTime(right.updatedAt) - toTime(left.updatedAt))
}

export function buildUnassignedConversationItems({ conversations, linkedSessionIds, activeConversationId }) {
  return (Array.isArray(conversations) ? conversations : [])
    .filter((conversation) => !conversation?.sessionId || !linkedSessionIds.has(conversation.sessionId))
    .map((conversation) => ({
      id: conversation.id,
      type: 'unassigned',
      title: conversation.title || '未命名会话',
      description: `${conversation.messageCount || 0} 条消息 · 未归属`,
      sessionId: conversation.sessionId || '',
      conversationId: conversation.id,
      active: conversation.id === activeConversationId,
      href: `/chat/${encodeURIComponent(conversation.id)}`,
      status: conversation.sessionId ? 'unassigned' : 'draft',
      updatedAt: conversationUpdatedAt(conversation),
    }))
}

export function buildActiveOutputItems({ conversation, workspace, projectStore }) {
  const sessionId = workspace?.sessionId || conversation?.sessionId || ''
  const projectId = projectStore?.activeProjectId || ''
  const trace = workspace?.researchTrace || {}
  const evidenceCount = trace.evidenceCount || workspace?.evidenceRecords?.length || 0
  const nextActions = [
    ...(Array.isArray(trace.uncertainties) ? trace.uncertainties : []),
    ...(Array.isArray(trace.ideaCandidates) ? trace.ideaCandidates : []),
  ].filter(Boolean)
  return [
    { id: 'dialogue', label: '对话流', state: conversation?.id ? '当前会话' : '待创建', href: conversation?.id ? `/chat/${encodeURIComponent(conversation.id)}` : '/chat' },
    { id: 'report', label: '会话报告', state: sessionId ? '可查看 / 待生成' : '首轮提问后生成', href: sessionId ? `/research/${encodeURIComponent(sessionId)}` : '' },
    { id: 'evidence', label: '证据图谱', state: evidenceCount ? `${evidenceCount} 条证据线索` : '当前会话暂无图谱预览', href: '/graph' },
    { id: 'progress', label: '推进记录', state: projectId ? `${nextActions.length || 0} 个候选下一步` : '需先进入项目范围', href: projectId ? `/projects/${encodeURIComponent(projectId)}` : '/projects' },
  ]
}

export function projectContextSummary(projectStore, paperCount) {
  const projectName = projectStore?.activeProject?.name || ''
  if (!projectName) return '当前未选择项目，先作为未归属会话继续。'
  return `${projectName} · ${paperCount || 0} 篇材料 · ${(projectStore?.sessions || []).length} 个会话`
}
