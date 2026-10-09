export const BRAND = {
  name: '研途',
  shortName: '研',
  tagline: '研究工作台',
  description: '围绕项目、论文与研究过程组织工作。',
  cta: {
    primary: '/projects',
    secondary: '/library',
  },
}

const WORKSPACE_ROUTE_META = [
  { href: '/projects', label: '项目', icon: 'solar:folder-with-files-linear', match: /^\/projects(\/|$)/ },
  {
    id: 'knowledge',
    label: '知识库',
    icon: 'solar:library-linear',
    match: /^\/(papers|library|graph)(\/|$)/,
    children: [
      { href: '/papers', label: '论文库', icon: 'solar:library-linear', match: /^\/(papers|library)(\/|$)/ },
      { href: '/graph', label: '关系图谱', icon: 'solar:graph-new-linear', match: /^\/graph(\/|$)/ },
    ],
  },
  {
    id: 'research',
    label: '研究',
    icon: 'solar:microscope-linear',
    match: /^\/(chat|research|evolution)(\/|$)/,
    children: [
      { href: '/chat', label: '对话', icon: 'solar:chat-round-dots-linear', match: /^\/chat(\/|$)/ },
      { href: '/research', label: '深度研究', icon: 'solar:document-search-linear', match: /^\/research(\/|$)/ },
      { href: '/evolution', label: '研究演化', icon: 'solar:chart-linear', match: /^\/evolution(\/|$)/ },
    ],
  },
  { href: '/paper-search', label: '论文搜索', icon: 'solar:magnifer-linear', match: /^\/paper-search(\/|$)/ },
  { href: '/daily-brief', label: '每日速览', icon: 'solar:calendar-mark-linear', match: /^\/daily-brief(\/|$)/ },
  { href: '/tasks', label: '任务', icon: 'solar:compass-linear', match: /^\/tasks(\/|$)/ },
  { href: '/observability', label: '观测', icon: 'solar:radar-linear', match: /^\/observability(\/|$)/ },
  { href: '/graph-inspector', label: '图谱数据', icon: 'solar:graph-new-linear', match: /^\/graph-inspector(\/|$)/ },
  { href: '/settings', label: '设置', icon: 'solar:settings-linear', match: /^\/settings(\/|$)/, secondary: true },
]

export const WORKSPACE_PRIMARY_NAV = WORKSPACE_ROUTE_META.filter((item) => !item.hidden && !item.secondary)

export const WORKSPACE_SECONDARY_NAV = WORKSPACE_ROUTE_META.filter((item) => item.secondary)

export const WORKSPACE_NAV = WORKSPACE_ROUTE_META

export function findWorkspaceNavItem(pathname = '') {
  for (const item of WORKSPACE_NAV) {
    if (Array.isArray(item.children)) {
      const child = item.children.find((entry) => entry.match.test(pathname))
      if (child) return child
    }
    if (item.match?.test(pathname)) return item
  }
  return null
}
