import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')

function fileExists(relativePath) {
  return fs.existsSync(path.resolve(root, relativePath))
}

function readText(relativePath) {
  return fs.readFileSync(path.resolve(root, relativePath), 'utf8')
}

test('Phase 1：workspace 路由分层文件存在', () => {
  assert.equal(fileExists('app/(workspace)/layout.jsx'), true)
  assert.equal(fileExists('app/(workspace)/chat/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/chat/[conversationId]/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/graph/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/research/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/paper-search/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/daily-brief/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/discover/page.jsx'), false)
  assert.equal(fileExists('app/(workspace)/dashboard/page.jsx'), false)
  assert.equal(fileExists('app/(workspace)/projects/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/projects/[projectId]/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/projects/[projectId]/graph/page.jsx'), false)
  assert.equal(fileExists('app/(workspace)/papers/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/papers/[paperId]/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/settings/page.jsx'), true)
  assert.equal(fileExists('app/(workspace)/research-map/page.jsx'), false)
  assert.equal(fileExists('app/(workspace)/knowledge-map/page.jsx'), false)
  assert.equal(fileExists('app/(workspace)/research/[sessionId]/page.jsx'), true)
})

test('Phase 2：聊天工作台拆分组件存在', () => {
  assert.equal(fileExists('components/layout/AppShell.jsx'), true)
  assert.equal(fileExists('components/layout/SideNav.jsx'), true)
  assert.equal(fileExists('components/layout/WorkspaceLayout.jsx'), true)
  assert.equal(fileExists('components/chat/ConversationSidebar.jsx'), true)
  assert.equal(fileExists('components/chat/ChatHeader.jsx'), true)
  assert.equal(fileExists('components/chat/MessageViewport.jsx'), true)
  assert.equal(fileExists('components/chat/ComposerBar.jsx'), true)
  assert.equal(fileExists('components/chat/InsightDrawer.jsx'), false)
})

test('Phase 3：store 与分域 API 文件存在', () => {
  assert.equal(fileExists('stores/conversation-store.js'), true)
  assert.equal(fileExists('stores/library-store.js'), true)
  assert.equal(fileExists('stores/project-store.js'), true)
  assert.equal(fileExists('stores/workspace-store.js'), true)
  assert.equal(fileExists('stores/ui-store.js'), true)

  assert.equal(fileExists('lib/api/conversations.js'), true)
  assert.equal(fileExists('lib/api/papers.js'), true)
  assert.equal(fileExists('lib/api/projects.js'), true)
  assert.equal(fileExists('lib/api/research.js'), true)
  assert.equal(fileExists('lib/api/notifications.js'), true)
})

test('Phase 4：论文库与论文详情页面组件存在', () => {
  assert.equal(fileExists('components/shared/PageHeader.jsx'), true)
  assert.equal(fileExists('components/library/LibraryPageShell.jsx'), true)
  assert.equal(fileExists('components/paper-graph/PaperGraphShell.jsx'), true)
  assert.equal(fileExists('components/paper-graph/PaperGraphCanvas.jsx'), true)
  assert.equal(fileExists('components/library/PaperUploadPanel.jsx'), true)
  assert.equal(fileExists('components/library/PaperCardList.jsx'), true)
  assert.equal(fileExists('components/project/ProjectWorkspaceShell.jsx'), true)
  assert.equal(fileExists('components/project/ProjectPaperStatusList.jsx'), true)
  assert.equal(fileExists('components/project/ProjectActivityFeed.jsx'), true)
  assert.equal(fileExists('components/project/ProjectResearchDesk.jsx'), true)
  assert.equal(fileExists('components/project/ProjectConversations.jsx'), false)
  assert.equal(fileExists('components/project/ProjectPaperBinder.jsx'), false)
  assert.equal(fileExists('components/project/ProjectPaperBoard.jsx'), false)
  assert.equal(fileExists('components/project/ProjectResearch.jsx'), false)
  assert.equal(fileExists('components/project/ProjectSessionFeed.jsx'), false)
  assert.equal(fileExists('components/project/ProjectSwitcher.jsx'), true)
  assert.equal(fileExists('components/project/ProjectScopePanel.jsx'), true)
  assert.equal(fileExists('components/paper/PaperDetailHeader.jsx'), true)
  assert.equal(fileExists('components/paper/PaperOverviewTab.jsx'), true)
  assert.equal(fileExists('components/paper/PaperAnalysisTab.jsx'), true)
  assert.equal(fileExists('components/paper/PaperOriginalTab.jsx'), true)
  assert.equal(fileExists('components/research/ResearchLauncher.jsx'), true)
  assert.equal(fileExists('components/research/UnifiedResearchWorkbench.jsx'), false)
  assert.equal(fileExists('components/research/ResearchSessionList.jsx'), true)
  assert.equal(fileExists('components/research/ReportExport.jsx'), true)
  assert.equal(fileExists('components/research/TraceReplayPanel.jsx'), true)
  assert.equal(fileExists('components/research-session/ResearchSessionLayout.jsx'), true)
  assert.equal(fileExists('components/research-session/EvidenceDeskSection.jsx'), true)
  assert.equal(fileExists('lib/research-session-view.js'), true)
  assert.equal(fileExists('lib/research-trace-export.js'), true)
  assert.equal(fileExists('components/research-map/ResearchEvolutionShell.jsx'), false)
  assert.equal(fileExists('components/research-map/KnowledgeMapShell.jsx'), false)
})

test('Phase 5：设计系统与品牌配置文件存在', () => {
  assert.equal(fileExists('styles/design-tokens.css'), true)
  assert.equal(fileExists('styles/components.css'), true)
  assert.equal(fileExists('config/brand.js'), true)
  assert.equal(fileExists('components/layout/AppShell.jsx'), true)
  assert.equal(fileExists('components/layout/SideNav.jsx'), true)
  assert.equal(fileExists('components/layout/AppTopBar.jsx'), true)
  assert.equal(fileExists('components/shared/EmptyState.jsx'), true)
  assert.equal(fileExists('components/shared/StatusBadge.jsx'), true)
  assert.equal(fileExists('components/shared/ErrorState.jsx'), true)
  assert.equal(fileExists('components/shared/Skeleton.jsx'), true)
  assert.equal(fileExists('components/auth/DeveloperOnlyRoute.jsx'), true)
  assert.equal(fileExists('components/paper-graph/GraphInterpretationPanel.jsx'), true)
  assert.equal(fileExists('components/project/PaperReadinessTimeline.jsx'), true)
  assert.equal(fileExists('app/(workspace)/error.jsx'), true)
  assert.equal(fileExists('app/(workspace)/loading.jsx'), true)
})

test('Batch 0：toast 与 Tailwind token bridge 基础设施存在', () => {
  const providersSource = readText('app/providers.jsx')
  const tailwindSource = readText('tailwind.config.js')
  const notifySource = readText('lib/notify.js')

  assert.match(providersSource, /ToastProvider/)
  assert.match(notifySource, /addToast/)
  assert.match(notifySource, /NOTIFY_DURATION_MS/)
  assert.match(tailwindSource, /surface: 'var\(--sa-surface\)'/)
  assert.match(tailwindSource, /accent: 'var\(--sa-accent\)'/)
  assert.match(tailwindSource, /'sa-7': 'var\(--sa-space-7\)'/)
})

test('Batch 4：视觉语义 class 与报告页拆分入口存在', () => {
  const rootLayoutSource = readText('app/layout.jsx')
  const visualSource = readText('styles/batch4-visual-system.css')
  const evidenceSource = readText('styles/batch4-evidence-progress.css')

  assert.match(rootLayoutSource, /batch4-visual-system\.css/)
  assert.match(rootLayoutSource, /batch4-chat-workspace\.css/)
  assert.match(rootLayoutSource, /batch4-evidence-progress\.css/)
  assert.match(rootLayoutSource, /batch4-research-session\.css/)
  assert.match(visualSource, /sa-research-room-/)
  assert.match(evidenceSource, /sa-evidence-desk-/)
  assert.match(evidenceSource, /sa-progress-accent-/)
})

test('首页已从工作台直挂改为入口页语义', () => {
  const source = readText('app/page.jsx')
  assert.match(source, /chat|workspace|开始|入口|研途/i)
})

test('papers 页具备页面化语义，library 路由仅作兼容跳转', () => {
  const papersSource = readText('app/(workspace)/papers/page.jsx')
  const papersClientSource = readText('components/library/PapersPageClient.jsx')
  const librarySource = readText('app/library/page.jsx')
  assert.match(papersSource, /PapersPageClient/)
  assert.match(papersClientSource, /LibraryPageShell|论文库|onAddToConversation/)
  assert.match(librarySource, /redirect\('\/papers'\)/)
})

test('全局论文图作为知识库二级入口存在', () => {
  const graphPageSource = readText('app/(workspace)/graph/page.jsx')
  const brandSource = readText('config/brand.js')
  assert.match(graphPageSource, /PaperGraphShell/)
  assert.match(brandSource, /label: '知识库'/)
  assert.match(brandSource, /href: '\/graph'/)
  assert.match(brandSource, /label: '关系图谱'/)
})

test('DESIGN 与前端 UI 指南已切换到项目列表 + 项目详细页语义', () => {
  const designSource = readText('../DESIGN.md')
  const uiGuideSource = readText('docs/FRONTEND_UI_GUIDE.md')
  assert.match(designSource, /项目列表页/)
  assert.match(designSource, /项目详细页/)
  assert.match(designSource, /论文详情页/)
  assert.match(designSource, /AI 分析/)
  assert.doesNotMatch(designSource, /Notion's website/)
  assert.match(uiGuideSource, /项目列表页/)
  assert.match(uiGuideSource, /项目详细页/)
  assert.match(uiGuideSource, /总览\s*\/\s*AI 分析\s*\/\s*原文/)
})
