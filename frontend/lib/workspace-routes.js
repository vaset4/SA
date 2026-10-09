export const DEFAULT_WORKSPACE_ENTRY = '/projects'

export function normalizeWorkspaceRedirect(target) {
  const value = String(target || '').trim()
  if (!value.startsWith('/') || value.startsWith('//')) return DEFAULT_WORKSPACE_ENTRY
  return value.startsWith('/login') ? DEFAULT_WORKSPACE_ENTRY : value
}

export function buildWorkspaceRedirectTarget(pathname, search = '') {
  const target = `${pathname || DEFAULT_WORKSPACE_ENTRY}${search || ''}`
  return target.startsWith('/login') ? DEFAULT_WORKSPACE_ENTRY : target
}

export function buildResearchWorkspaceHref({
  prompt = '',
  paperId = '',
  projectId = '',
  source = '',
  focus = 'external',
} = {}) {
  const search = new URLSearchParams()
  if (prompt) search.set('prompt', prompt)
  if (paperId) search.set('paperId', paperId)
  if (projectId) search.set('projectId', projectId)
  if (source) search.set('source', source)
  if (focus) search.set('focus', focus)
  const query = search.toString()
  return query ? `/research?${query}` : '/research'
}

export function buildInternalResearchHref({
  prompt = '',
  paperId = '',
} = {}) {
  const search = new URLSearchParams()
  if (prompt) search.set('prompt', prompt)
  if (paperId) search.set('paperId', paperId)
  const query = search.toString()
  return query ? `/chat?${query}` : '/chat'
}

export function resolveResearchSourceLabel(source) {
  switch (String(source || '').trim()) {
    case 'paper':
      return '论文详情'
    case 'graph':
      return '论文图谱'
    case 'paper_search':
      return '论文搜索'
    case 'project':
      return '项目详情'
    case 'chat':
      return '对话'
    default:
      return ''
  }
}
