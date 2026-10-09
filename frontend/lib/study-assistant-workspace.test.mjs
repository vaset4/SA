import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  MESSAGE_EXTRA_KEYS,
  WELCOME,
  hasConversationContent,
  pickMessageExtras,
  resolveMessageRenderMode,
  resolveComposerModeChange,
  restoredMessages,
  selectNextPaperId,
  toMessage,
} from './study-assistant-workspace.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

function readFirst(paths) {
  for (const filePath of paths) {
    if (fs.existsSync(filePath)) return fs.readFileSync(filePath, 'utf8')
  }
  return ''
}

const PANELS_PATH = path.resolve(__dirname, '../components/study-assistant-panels.jsx')
const COMPOSER_BAR_PATH = path.resolve(__dirname, '../components/chat/ComposerBar.jsx')
const SIDEBAR_PATH = path.resolve(__dirname, '../components/chat/ConversationSidebar.jsx')
const LIBRARY_UPLOAD_PANEL_PATH = path.resolve(__dirname, '../components/library/PaperUploadPanel.jsx')
const LIBRARY_PAGE_PATH = path.resolve(__dirname, '../app/library/page.jsx')
const PAPERS_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/papers/page.jsx')
const PAPERS_PAGE_CLIENT_PATH = path.resolve(__dirname, '../components/library/PapersPageClient.jsx')
const HOOKS_PATH = path.resolve(__dirname, '../components/study-assistant-hooks.js')
const WORKSPACE_STORE_PATH = path.resolve(__dirname, '../stores/workspace-store.js')
const SHELL_PATH = path.resolve(__dirname, '../components/study-assistant-shell.jsx')
const CHAT_WORKSPACE_SHELL_PATH = path.resolve(__dirname, '../components/chat/ChatWorkspaceShell.jsx')
const CHAT_WORKSPACE_MODEL_PATH = path.resolve(__dirname, '../components/chat/use-chat-workspace-model.js')
const CHAT_RESEARCH_RAIL_PATH = path.resolve(__dirname, '../components/chat/ChatResearchRail.jsx')
const WORKSPACE_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/chat/page.jsx')
const RESEARCH_HOME_PATH = path.resolve(__dirname, '../app/(workspace)/research/page.jsx')
const RESEARCH_SESSION_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/research/[sessionId]/page.jsx')
const RESEARCH_LAUNCHER_PATH = path.resolve(__dirname, '../components/research/ResearchLauncher.jsx')
const PAPER_SEARCH_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/paper-search/page.jsx')
const DAILY_BRIEF_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/daily-brief/page.jsx')
const RESEARCH_TRACE_PATH = path.resolve(__dirname, './research-trace.js')
const RESEARCH_TRACE_EXPORT_PATH = path.resolve(__dirname, './research-trace-export.js')
const TRACE_REPLAY_PANEL_PATH = path.resolve(__dirname, '../components/research/TraceReplayPanel.jsx')
const CONVERSATIONS_PATH = path.resolve(__dirname, '../components/study-assistant-conversations.js')
const CONVERSATION_STORE_PATH = path.resolve(__dirname, '../stores/conversation-store.js')
const BRAND_CONFIG_PATH = path.resolve(__dirname, '../config/brand.js')
const STORAGE_PATH = path.resolve(__dirname, './storage-service.js')
const API_CORE_PATH = path.resolve(__dirname, './api/core.js')
const RESEARCH_API_PATH = path.resolve(__dirname, './api/research.js')
const AUTH_API_PATH = path.resolve(__dirname, './api/auth.js')
const AUTH_CLIENT_PATH = path.resolve(__dirname, './auth-client.js')
const LOGIN_PAGE_PATH = path.resolve(__dirname, '../app/login/page.jsx')
const LOGIN_PAGE_CLIENT_PATH = path.resolve(__dirname, '../components/auth/LoginPageClient.jsx')
const CONVERSATIONS_API_PATH = path.resolve(__dirname, './api/conversations.js')
const STREAM_CLIENT_PATH = path.resolve(__dirname, './study-assistant-stream.js')
const LEGACY_API_PATH = path.resolve(__dirname, './study-assistant-api.js')
const PROJECT_STORE_PATH = path.resolve(__dirname, '../stores/project-store.js')
const PROJECT_DETAIL_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/projects/[projectId]/page.jsx')
const PAPER_DETAIL_PAGE_PATH = path.resolve(__dirname, '../app/(workspace)/papers/[paperId]/page.jsx')
const PAPER_OVERVIEW_TAB_PATH = path.resolve(__dirname, '../components/paper/PaperOverviewTab.jsx')
const PROJECT_CONVERSATIONS_PATH = path.resolve(__dirname, '../components/project/ProjectConversations.jsx')
const PAPER_GRAPH_SHELL_PATH = path.resolve(__dirname, '../components/paper-graph/PaperGraphShell.jsx')
const MESSAGE_VIEWPORT_PATH = path.resolve(__dirname, '../components/chat/MessageViewport.jsx')
const MESSAGE_CONTENT_PATH = path.resolve(__dirname, '../components/chat/MessageContent.jsx')
const MESSAGE_RESEARCH_PLANNER_PATH = path.resolve(__dirname, '../components/chat/MessageResearchPlannerProcess.jsx')
const MESSAGE_RESEARCH_PLANNER_SUMMARY_PATH = path.resolve(__dirname, '../components/chat/MessageResearchPlannerSummary.jsx')
const MESSAGE_RESEARCH_SYNTHESIS_PATH = path.resolve(__dirname, '../components/chat/MessageResearchSynthesisSummary.jsx')
const SIDENAV_LAYOUT_PATH = path.resolve(__dirname, '../components/layout/SideNav.jsx')
const WORKSPACE_ROUTE_LAYOUT_PATH = path.resolve(__dirname, '../components/layout/WorkspaceRouteLayoutClient.jsx')
const CONVERSATION_REPOSITORY_PATH = path.resolve(__dirname, '../../backend/app/repositories/conversation_repository.py')
const HOME_PAGE_PATH = path.resolve(__dirname, '../app/page.jsx')

const composerSource = readFirst([COMPOSER_BAR_PATH, PANELS_PATH])
const sidebarSource = readFirst([SIDEBAR_PATH, PANELS_PATH])
const libraryPanelSource = readFirst([LIBRARY_UPLOAD_PANEL_PATH, PANELS_PATH])
const libraryPageSource = readFirst([PAPERS_PAGE_CLIENT_PATH, PAPERS_PAGE_PATH, LIBRARY_PAGE_PATH])
const libraryRedirectSource = readFirst([LIBRARY_PAGE_PATH])
const hooksSource = readFirst([HOOKS_PATH, WORKSPACE_STORE_PATH])
const workspaceStoreSource = readFirst([WORKSPACE_STORE_PATH])
const panelsSource = readFirst([PANELS_PATH])
const shellLikeSource = readFirst([SHELL_PATH, WORKSPACE_PAGE_PATH])
const chatWorkspaceShellSource = readFirst([CHAT_WORKSPACE_SHELL_PATH]) + readFirst([CHAT_WORKSPACE_MODEL_PATH])
const chatResearchRailSource = readFirst([CHAT_RESEARCH_RAIL_PATH])
const conversationSource = readFirst([CONVERSATION_STORE_PATH, CONVERSATIONS_PATH])
const brandSource = readFirst([BRAND_CONFIG_PATH])
const storageSource = readFirst([STORAGE_PATH])
const apiCoreSource = readFirst([API_CORE_PATH])
const researchApiSource = readFirst([RESEARCH_API_PATH])
const authApiSource = readFirst([AUTH_API_PATH])
const authClientSource = readFirst([AUTH_CLIENT_PATH])
const loginPageSource = readFirst([LOGIN_PAGE_PATH])
const loginPageClientSource = readFirst([LOGIN_PAGE_CLIENT_PATH])
const conversationsApiSource = readFirst([CONVERSATIONS_API_PATH])
const streamClientSource = readFirst([STREAM_CLIENT_PATH])
const legacyApiSource = readFirst([LEGACY_API_PATH])
const projectStoreSource = readFirst([PROJECT_STORE_PATH])
const projectDetailPageSource = readFirst([PROJECT_DETAIL_PAGE_PATH])
const paperDetailPageSource = readFirst([PAPER_DETAIL_PAGE_PATH])
const paperOverviewTabSource = readFirst([PAPER_OVERVIEW_TAB_PATH])
const projectConversationsSource = readFirst([PROJECT_CONVERSATIONS_PATH])
const paperGraphShellSource = readFirst([PAPER_GRAPH_SHELL_PATH])
const messageViewportSource = readFirst([MESSAGE_VIEWPORT_PATH])
const messageContentSource = readFirst([MESSAGE_CONTENT_PATH])
const plannerProcessSource = readFirst([MESSAGE_RESEARCH_PLANNER_PATH])
const plannerSummarySource = readFirst([MESSAGE_RESEARCH_PLANNER_SUMMARY_PATH])
const synthesisSummarySource = readFirst([MESSAGE_RESEARCH_SYNTHESIS_PATH])
const sideNavSource = readFirst([SIDENAV_LAYOUT_PATH])
const workspaceRouteLayoutSource = readFirst([WORKSPACE_ROUTE_LAYOUT_PATH])
const conversationRepositorySource = readFirst([CONVERSATION_REPOSITORY_PATH])
const homePageSource = readFirst([HOME_PAGE_PATH])
const researchHomeSource = readFirst([RESEARCH_HOME_PATH])
const researchSessionPageSource = readFirst([RESEARCH_SESSION_PAGE_PATH])
const researchLauncherSource = readFirst([RESEARCH_LAUNCHER_PATH])
const paperSearchPageSource = readFirst([PAPER_SEARCH_PAGE_PATH])
const dailyBriefPageSource = readFirst([DAILY_BRIEF_PAGE_PATH])
const researchTraceSource = readFirst([RESEARCH_TRACE_PATH])
const researchTraceExportSource = readFirst([RESEARCH_TRACE_EXPORT_PATH])
const traceReplayPanelSource = readFirst([TRACE_REPLAY_PANEL_PATH])
const researchSessionActionsSource = readFirst([path.resolve(__dirname, '../components/research-session/ResearchSessionActions.jsx')])
const researchSessionLayoutSource = readFirst([path.resolve(__dirname, '../components/research-session/ResearchSessionLayout.jsx')])
const researchProcessSectionSource = readFirst([path.resolve(__dirname, '../components/research-session/ResearchProcessSection.jsx')])
const researchSessionSideSource = readFirst([path.resolve(__dirname, '../components/research-session/ResearchSessionSidePanel.jsx')])
const researchSessionViewSource = readFirst([path.resolve(__dirname, './research-session-view.js')])

test('selectNextPaperId 保留已选论文，否则回退到第一篇或空字符串', () => {
  const items = [{ paper_id: 'p1' }, { paper_id: 'p2' }]
  assert.equal(selectNextPaperId(items, 'p2'), 'p2')
  assert.equal(selectNextPaperId(items, 'missing'), 'p1')
  assert.equal(selectNextPaperId([], 'missing'), '')
})

test('resolveComposerModeChange 让 Deep Research 固定进入 plan 模式', () => {
  assert.deepEqual(resolveComposerModeChange('chat'), { mode: 'chat' })
  assert.deepEqual(resolveComposerModeChange('research'), { mode: 'research', researchMode: 'plan' })
})

test('聊天输入区保留项目与论文范围，并收敛解释文案', () => {
  assert.match(composerSource, /论文范围 \{paperCount\}/)
  assert.match(composerSource, /项目 \{projectName\}/)
  assert.doesNotMatch(composerSource, /当前问题会先结合/)
  assert.doesNotMatch(composerSource, /Deep Research|外部补证/)
})

test('Deep Research 不再显示计划模式或流式模式切换按钮', () => {
  assert.doesNotMatch(composerSource, /计划模式/)
  assert.doesNotMatch(composerSource, /流式模式/)
})

test('restoredMessages 在恢复会话时拼接欢迎消息与历史消息', () => {
  const messages = restoredMessages({
    session: { session_id: 'session-1', chat_history: [{ role: 'user', content: 'hello' }] },
    pending: null,
  })

  assert.equal(messages[0].title, 'Session restored')
  assert.equal(messages[1].role, 'user')
  assert.equal(messages[1].content, 'hello')
  assert.equal(WELCOME.role, 'assistant')
})

test('restoredMessages 兼容 LangGraph snapshot.values.chat_history 结构', () => {
  const messages = restoredMessages({
    session: {
      session_id: 'session-2',
      values: {
        chat_history: [{ role: 'assistant', content: 'from checkpoint' }],
      },
    },
    pending: null,
  })

  assert.equal(messages[0].title, 'Session restored')
  assert.equal(messages[1].role, 'assistant')
  assert.equal(messages[1].content, 'from checkpoint')
})

test('toMessage 支持显式 renderMode，free chat 默认保持 plain', () => {
  const research = toMessage('assistant', '# 标题\n\n- 要点', 'Research result', undefined, 'markdown')
  const chat = toMessage('assistant', '普通回答', 'Chat result')
  const user = toMessage('user', '你好')

  assert.equal(research.renderMode, 'markdown')
  assert.equal(chat.renderMode, 'plain')
  assert.equal(user.renderMode, 'plain')
})

test('resolveMessageRenderMode 能从 Markdown 特征和标题推断类型', () => {
  assert.equal(resolveMessageRenderMode({ role: 'assistant', content: '## 结论\n\n- A' }), 'markdown')
  assert.equal(resolveMessageRenderMode({ role: 'assistant', title: 'Literature Review', content: '纯文本也应按 Markdown 处理' }), 'markdown')
  assert.equal(resolveMessageRenderMode({ role: 'assistant', title: 'Chat result', content: '这是一段普通回答。' }), 'plain')
})

test('restoredMessages 会用 session report 回填被截断的 assistant preview', () => {
  const messages = restoredMessages({
    session: {
      session_id: 'session-3',
      values: {
        chat_history: [{ role: 'assistant', content: '# 深度研究报告...' }],
        report: '# 深度研究报告\n\n## 结论\n\n- 完整内容',
      },
    },
    pending: null,
  })

  assert.equal(messages[1].content, '# 深度研究报告\n\n## 结论\n\n- 完整内容')
  assert.equal(messages[1].renderMode, 'markdown')
})

test('restoredMessages 会保留研究整理附加字段', () => {
  const messages = restoredMessages({
    session: {
      session_id: 'session-4',
      values: {
        chat_history: [{
          role: 'assistant',
          content: '已完成研究整理。',
          dialogue_mode: 'research',
          synthesis_state: {
            status: 'ready',
            dialogue_mode: 'research',
            supported_dimensions: ['methods'],
          },
        }],
      },
    },
    pending: null,
  })

  assert.equal(messages[1].dialogue_mode, 'research')
  assert.equal(messages[1].synthesis_state.status, 'ready')
})

test('restoredMessages 会从 metadata 恢复研究附加字段', () => {
  const messages = restoredMessages({
    session: {
      session_id: 'session-5',
      values: {
        chat_history: [{
          role: 'assistant',
          content: '已完成证据整理。',
          metadata: {
            evidence_records: [{ layer: 'milvus', excerpt: 'direct quote' }],
            graph_hint_section: '图谱线索：preview',
            graph_hint_full: '图谱线索（展开）：full graph context',
            answer_brief: { summary: '可回答' },
            missing_evidence: [{ paper_id: 'p1', reason: 'no_hit' }],
            research_plan: { units: [{ unit_id: 'u1' }] },
            research_unit_results: [{ unit_id: 'u1', status: 'evidence_found' }],
            coverage_review: { covered_facet_ids: ['f1'] },
            planner_trace: { analyzer_plan_type: 'single_unit' },
            trace_summary: { source_status: { milvus: 'ok' } },
            run_id: 'run-1',
          },
        }],
      },
    },
    pending: null,
  })

  assert.equal(messages[1].evidence_records[0].layer, 'milvus')
  assert.equal(messages[1].graph_hint_section, '图谱线索：preview')
  assert.equal(messages[1].graph_hint_full, '图谱线索（展开）：full graph context')
  assert.equal(messages[1].answer_brief.summary, '可回答')
  assert.equal(messages[1].missing_evidence[0].reason, 'no_hit')
  assert.equal(messages[1].research_plan.units[0].unit_id, 'u1')
  assert.equal(messages[1].research_unit_results[0].status, 'evidence_found')
  assert.equal(messages[1].coverage_review.covered_facet_ids[0], 'f1')
  assert.equal(messages[1].planner_trace.analyzer_plan_type, 'single_unit')
  assert.equal(messages[1].trace_summary.source_status.milvus, 'ok')
  assert.equal(messages[1].run_id, 'run-1')
})

test('pickMessageExtras 只从 metadata 恢复白名单研究字段', () => {
  const extras = pickMessageExtras({
    metadata: {
      evidence_records: [{ evidence_id: 'ev-1' }],
      research_plan: { units: [{ unit_id: 'u1' }] },
      planner_trace: { status: 'ok' },
      planner_final_retry_attempts: [{ status: 'failed' }],
      ignored_raw_payload: { text: 'too large' },
    },
  })

  assert.deepEqual(extras.evidence_records, [{ evidence_id: 'ev-1' }])
  assert.deepEqual(extras.research_plan, { units: [{ unit_id: 'u1' }] })
  assert.deepEqual(extras.planner_trace, { status: 'ok' })
  assert.deepEqual(extras.planner_final_retry_attempts, [{ status: 'failed' }])
  assert.equal(extras.ignored_raw_payload, undefined)
})

test('hasConversationContent 只把真实会话消息算作已加载内容', () => {
  assert.equal(hasConversationContent([]), false)
  assert.equal(hasConversationContent([{ id: 'welcome' }]), false)
  assert.equal(hasConversationContent([{ id: 'restored' }]), false)
  assert.equal(hasConversationContent([{ id: 'user-1', role: 'user', content: 'hello' }]), true)
})

test('会话侧栏包含创建会话动作', () => {
  assert.match(sidebarSource, /onCreateConversation|handleCreateConversation|createNew/)
})

test('Composer 具备输入与提交行为标记', () => {
  assert.match(composerSource, /sa-input|composer-input|textarea/)
  assert.match(composerSource, /sa-submit|composer-submit|onSubmit/)
})

test('Composer 已移除 Chat / Deep Research / arXiv 模式切换按钮', () => {
  assert.doesNotMatch(composerSource, /item === 'chat'/)
  assert.doesNotMatch(composerSource, /arxiv/i)
  assert.doesNotMatch(composerSource, /Deep Research/)
})

test('上传面板保留后端处理中提示', () => {
  assert.match(libraryPanelSource, /后端处理中|upload|processing/i)
})

test('论文库页面展示删除任务状态和失败信息', () => {
  assert.match(libraryPanelSource, /论文删除任务/)
  assert.match(libraryPanelSource, /排队中|清理中|已完成|失败/)
  assert.match(libraryPanelSource, /deleteResult\.error/)
})

test('有关联论文时聊天仍会走 crossPaperQA', () => {
  assert.match(hooksSource, /crossPaperQA/)
  assert.match(hooksSource, /papers\.length\s*>\s*0/)
  assert.match(hooksSource, /chatHistory:\s*buildChatHistory\(baseMessages,\s*prompt\)/)
  assert.match(hooksSource, /events:\s*\[\]/)
  assert.match(researchApiSource, /chat_history:\s*chatHistory/)
})

test('Deep Research API 在新旧封装中都显式使用 600s 超时', () => {
  assert.match(researchApiSource, /const RESEARCH_REQUEST_TIMEOUT_MS = 600_000/)
  assert.match(researchApiSource, /timeoutMs:\s*RESEARCH_REQUEST_TIMEOUT_MS/)
  assert.match(researchApiSource, /crossPaperQA[\s\S]*timeoutMs:\s*RESEARCH_REQUEST_TIMEOUT_MS/)
  assert.equal(legacyApiSource, '')
})

test('认证相关前端调用统一切到 /api/v1/auth 前缀', () => {
  assert.match(authApiSource, /\/api\/v1\/auth\/login/)
  assert.match(authApiSource, /\/api\/v1\/auth\/register/)
  assert.match(authApiSource, /\/api\/v1\/auth\/me/)
  assert.match(authClientSource, /\/api\/v1\/auth\/refresh/)
})

test('登录默认回到项目页，并保留受保护页 redirect', () => {
  assert.match(authClientSource, /DEFAULT_WORKSPACE_ENTRY|buildWorkspaceRedirectTarget/)
  assert.match(loginPageSource, /normalizeWorkspaceRedirect/)
  assert.match(loginPageClientSource, /redirectTarget = DEFAULT_WORKSPACE_ENTRY/)
  assert.match(loginPageClientSource, /登录后进入项目驱动的研究工作台/)
  assert.match(workspaceRouteLayoutSource, /buildWorkspaceRedirectTarget/)
  assert.doesNotMatch(loginPageSource, /return '\/chat'/)
  assert.doesNotMatch(loginPageClientSource, /redirectTarget = '\/chat'/)
})

test('workspace 仍以会话级 sessionId 作为调用上下文', () => {
  assert.match(
    chatWorkspaceShellSource,
    /sessionId:\s*activeConversation\?\.sessionId \|\| ''/,
  )
  assert.match(
    chatWorkspaceShellSource,
    /workspace\.sessionId/,
  )
})

test('聊天工作区会把当前 active projectId 绑定到 workspace，并维护会话论文范围', () => {
  assert.match(chatWorkspaceShellSource, /const activeProjectId = projectStore\.activeProjectId \|\| ''/)
  assert.match(chatWorkspaceShellSource, /const scopedPapers = useMemo\(\(\) => \{/)
  assert.match(chatWorkspaceShellSource, /paperIdParam/)
  assert.match(chatWorkspaceShellSource, /bindConversationRef\.current\(\{[\s\S]*projectId:\s*activeProjectId/)
  assert.match(chatWorkspaceShellSource, /paperActions\.addPaper/)
  assert.match(chatWorkspaceShellSource, /paperActions\.removePaper/)
  assert.match(chatWorkspaceShellSource, /if \(paperIdParam\)[\s\S]*library\.setSelectedPaperId\(paperIdParam\)/)
})

test('聊天工作区固定回到内部研究模式，不再承接 URL 模式切换', () => {
  assert.doesNotMatch(chatWorkspaceShellSource, /modeParam/)
  assert.match(chatWorkspaceShellSource, /workspace\.setMode\('chat'\)/)
})

test('项目空间已经具备 store 与页面入口', () => {
  assert.match(projectStoreSource, /activeProjectId|createProject|listProjects|deleteProject/)
  const projectShellSource = readFirst([path.resolve(__dirname, '../components/project/ProjectWorkspaceShell.jsx')])
  assert.match(projectShellSource, /项目列表/)
  assert.match(projectShellSource, /进入项目详情/)
  assert.match(projectShellSource, /删除项目/)
})

test('项目空间会拉取项目会话并自动绑定当前聊天 session', () => {
  const projectsApiSource = readFirst([path.resolve(__dirname, './api/projects.js')])
  assert.match(projectStoreSource, /getProjectSessions|attachSession|sessions/)
  assert.match(projectsApiSource, /getProjectSessions|attachProjectSession/)
  assert.match(sidebarSource, /createNew\('新对话', \{ returnConversation: true \}\)/)
  assert.match(sidebarSource, /projectStore\.attachSession\(sessionId\)/)
  assert.match(workspaceStoreSource, /attachProjectSessionIfNeeded\(projectId,\s*requestSessionId\)/)
  assert.doesNotMatch(chatWorkspaceShellSource, /attachSession\(workspace\.sessionId\)/)
})

test('研究与问答请求会透传 project_id', () => {
  assert.match(researchApiSource, /project_id|projectId/)
  assert.match(workspaceStoreSource, /const projectId = options\.projectId \?\? currentState\.projectId|projectId,/)
})

test('图谱请求支持外部 AbortSignal 透传，requestJson 能识别 signal', () => {
  assert.match(researchApiSource, /signal/)
  assert.match(apiCoreSource, /externalSignal:\s*init\.signal/)
  assert.match(apiCoreSource, /externalSignal\?\.aborted/)
})

test('研究模式已切到 SSE 流式客户端，而不是同步 researchWithAssistant', () => {
  assert.match(streamClientSource, /\/api\/v1\/research\/stream/)
  assert.match(workspaceStoreSource, /runResearchStream/)
  assert.doesNotMatch(workspaceStoreSource, /researchWithAssistant\(\{/)
})

test('durable stream 复用幂等键并按 sequence 重连，停止时取消后端 run', () => {
  const ragRunsApiSource = readFirst([path.resolve(__dirname, './api/rag-runs.js')])
  assert.match(streamClientSource, /Idempotency-Key/)
  assert.match(streamClientSource, /X-RAG-Run-ID/)
  assert.match(streamClientSource, /events\/stream\?after_sequence=/)
  assert.match(streamClientSource, /method:\s*runId\s*\?\s*'GET'\s*:\s*'POST'/)
  assert.match(workspaceStoreSource, /activeStreamControllerRef\s*=\s*useRef/)
  assert.match(workspaceStoreSource, /activeRunIdRef\s*=\s*useRef/)
  assert.match(workspaceStoreSource, /cancelRagRun\(\{ runId \}\)/)
  assert.match(ragRunsApiSource, /rag-runs\/\$\{encodeURIComponent\(runId\)\}\/cancel/)
  assert.match(composerSource, /停止当前研究|停止/)
})

test('聊天头部优先展示实时 sessionId，避免本地会话元数据滞后', () => {
  assert.match(chatWorkspaceShellSource, /currentSessionId=\{workspace\.sessionId\}/)
})

test('聊天页右栏只展示知识库论文选择', () => {
  assert.match(chatWorkspaceShellSource, /library=\{library\}/)
  assert.match(chatWorkspaceShellSource, /activePaperIds=\{model\.activeConversation\?\.associatedPapers \|\| \[\]\}/)
  assert.match(chatResearchRailSource, /知识库论文/)
  assert.match(chatResearchRailSource, /onAddPaper/)
  assert.match(chatResearchRailSource, /筛选论文/)
  assert.doesNotMatch(chatResearchRailSource, /当前上下文/)
  assert.doesNotMatch(chatResearchRailSource, /ActiveSessionOutputs/)
  assert.doesNotMatch(chatResearchRailSource, /项目目标/)
  assert.doesNotMatch(chatResearchRailSource, /下一步/)
  assert.doesNotMatch(chatWorkspaceShellSource, /const researchHref = buildResearchWorkspaceHref\(/)
  assert.doesNotMatch(composerSource, /researchHref/)
})

test('workspace 会维护 researchTrace 并合并流式事件状态', () => {
  assert.match(workspaceStoreSource, /researchTrace/)
  assert.match(workspaceStoreSource, /mergeResearchTracePatch/)
  assert.match(workspaceStoreSource, /attachProjectSessionIfNeeded/)
})

test('研究页使用统一 trace 格式化工具，展示更细粒度执行轨迹', () => {
  assert.match(researchTraceSource, /formatResearchTraceTitle/)
  assert.match(researchTraceSource, /formatResearchTraceDetail/)
  assert.match(researchHomeSource, /formatResearchTraceTitle/)
  assert.match(researchSessionViewSource, /buildResearchTraceBundle/)
})

test('侧栏导航使用本地 SVG 图标回退，避免依赖远程 Iconify 拉取', () => {
  assert.match(sideNavSource, /function WorkspaceIcon/)
  assert.doesNotMatch(sideNavSource, /@iconify\/react/)
})

test('新对话会主动创建后端 session，并保留本地规范化能力', () => {
  assert.match(conversationSource, /await createSession\(/)
  assert.match(storageSource, /static normalizeSessionId\(sessionId\)/)
})

test('workspace 恢复阶段会阻止旧消息串写到当前会话', () => {
  assert.match(chatWorkspaceShellSource, /if \(workspace\.restoring\) return/)
  assert.match(workspaceStoreSource, /restoring:\s*shouldRestore/)
})

test('workspace 恢复空会话时会覆盖当前消息，避免沿用上个会话内容', () => {
  assert.match(workspaceStoreSource, /messages:\s*restored/)
  assert.match(workspaceStoreSource, /hasConversationContent\(currentMessages\)[\s\S]*\?\s*currentMessages[\s\S]*:\s*restoredSnapshot/)
})

test('消息视图统一通过 MessageContent 渲染，支持 Markdown 与纯文本共存', () => {
  assert.match(messageViewportSource, /MessageContent/)
  assert.equal(panelsSource, '')
  assert.match(messageContentSource, /sa-richtext/)
  assert.match(messageContentSource, /resolveMessageRenderMode/)
  assert.match(messageContentSource, /remarkMath/)
  assert.match(messageContentSource, /rehypeKatex/)
  assert.match(messageContentSource, /normalizeMathDelimiters/)
})

test('会话消息同步会持久化 render_mode 到前后端', () => {
  assert.match(conversationsApiSource, /render_mode/)
  assert.match(conversationRepositorySource, /render_mode/)
})

test('会话消息同步会把研究字段写入 metadata', () => {
  assert.match(conversationsApiSource, /MESSAGE_EXTRA_KEYS/)
  assert.match(conversationsApiSource, /metadata:\s*messageMetadata\(m\)/)
  assert.match(conversationRepositorySource, /metadata JSONB/)
  assert.match(conversationRepositorySource, /_message_data/)
})

test('会话后端 reload 会从 metadata 恢复研究字段', () => {
  assert.match(conversationSource, /pickMessageExtras/)
  assert.match(conversationSource, /\.\.\.extras/)
})

test('workspace 遇到会话不存在时会清理当前 conversation 的 stale sessionId', () => {
  assert.match(workspaceStoreSource, /conversations\.setSessionId\(requestConversationId,\s*''\)/)
  assert.match(workspaceStoreSource, /sessionId:\s*''/)
})

test('知识库页面提供会话连接或研究动作入口', () => {
  assert.match(libraryPageSource, /onAddToConversation|添加到会话|开始提问|开始研究/)
})

test('论文库页面支持原地加入项目，旧图谱入口待下线', () => {
  const paperCardSource = readFirst([path.resolve(__dirname, '../components/library/PaperCardList.jsx')])
  assert.match(libraryPageSource, /onAddToProject/)
  assert.match(paperCardSource, /添加到项目/)
  assert.match(paperCardSource, /查看图谱/)
  assert.match(paperCardSource, /卡片视图|列表视图/)
})

test('论文全库删除必须确认，取消确认不会调用删除请求', () => {
  const libraryShellSource = readFirst([path.resolve(__dirname, '../components/library/LibraryPageShell.jsx')])
  const taskCenterSource = readFirst([path.resolve(__dirname, '../app/(workspace)/tasks/page.jsx')])
  assert.match(libraryShellSource, /ConfirmDialog/)
  assert.match(libraryShellSource, /onCancel=\{closeDeleteConfirmation\}/)
  assert.match(libraryShellSource, /onConfirm=\{confirmDeletePaper\}/)
  assert.match(libraryShellSource, /tools\.remove\(paperId, library\.refresh\)/)
  assert.doesNotMatch(libraryShellSource, /onDelete=\{\(paperId\) => tools\.remove/)
  assert.match(taskCenterSource, /task\.task_type !== 'paper_delete'/)
})

test('论文详情页支持原地加入当前项目', () => {
  const paperHeaderSource = readFirst([path.resolve(__dirname, '../components/paper/PaperDetailHeader.jsx')])
  const paperOverviewSource = readFirst([path.resolve(__dirname, '../components/paper/PaperOverviewTab.jsx')])
  assert.match(paperHeaderSource, /添加到项目|加入项目/)
  assert.match(paperOverviewSource, /待补元数据|元数据完整/)
  assert.match(paperOverviewSource, /编辑元数据|保存元数据/)
  const processingSource = readFirst([path.resolve(__dirname, '../components/paper/PaperProcessingPanel.jsx')])
  assert.match(processingSource, /继续补全论文信息/)
  assert.doesNotMatch(paperOverviewSource, /handleAutofill|autofillPaperMetadata/)
})

test('论文详情页开始提问通过 URL 传递 request-scoped paperId', () => {
  assert.match(
    paperDetailPageSource,
    /router\.push\(`\/chat\/\$\{targetConversationId\}\?paperId=\$\{encodeURIComponent\(paperId\)\}`\)/,
  )
})

test('论文详情页把研究请求收口到研究工作区', () => {
  assert.match(paperDetailPageSource, /buildResearchWorkspaceHref\(\{/)
  assert.match(paperDetailPageSource, /paperId,/)
  assert.match(paperDetailPageSource, /source:\s*'paper'/)
  assert.doesNotMatch(paperDetailPageSource, /mode=research/)
  assert.doesNotMatch(paperDetailPageSource, /const startResearch[\s\S]*ensureConversation\(/)
})

test('项目页以研究工作面组织，而不是分发跳转', () => {
  assert.match(projectDetailPageSource, /ProjectResearchDesk/)
  assert.match(projectDetailPageSource, /ProjectActivityFeed/)
  assert.match(projectDetailPageSource, /ProjectPaperStatusList/)
  assert.doesNotMatch(projectDetailPageSource, /进入研究工作区|打开内部执行台|独立图页/)
})

test('项目页提供卡片式项目列表与详情入口', () => {
  const projectShellSource = readFirst([path.resolve(__dirname, '../components/project/ProjectWorkspaceShell.jsx')])
  assert.match(projectShellSource, /项目列表/)
  assert.match(projectShellSource, /进入项目详情/)
  assert.match(projectShellSource, /论文 \{metrics\.paperCount \?\? 0\}/)
  assert.match(projectShellSource, /对话 \{metrics\.conversationCount \?\? 0\}/)
  assert.match(projectShellSource, /研究 \{metrics\.researchCount \?\? 0\}/)
})

test('项目详情页围绕研究材料、研究入口和活动回看组织', () => {
  assert.match(projectDetailPageSource, /ProjectHeroPanel/)
  assert.match(projectDetailPageSource, /ProjectPaperStatusList/)
  assert.match(projectDetailPageSource, /ProjectResearchDesk/)
  assert.match(projectDetailPageSource, /ProjectActivityFeed/)
  assert.doesNotMatch(projectDetailPageSource, /ProjectWorkbenchPanels/)
  assert.doesNotMatch(projectDetailPageSource, /ProjectPaperBoard/)
  assert.doesNotMatch(projectDetailPageSource, /ProjectPaperBinder/)
})

test('项目列表区分 loading、error、empty 和列表态', () => {
  const projectShellSource = readFirst([path.resolve(__dirname, '../components/project/ProjectWorkspaceShell.jsx')])
  assert.match(projectShellSource, /projectStore\.loading/)
  assert.match(projectShellSource, /<Skeleton/)
  assert.match(projectShellSource, /<ErrorState/)
  assert.match(projectShellSource, /onRetry={projectStore\.refresh}/)
  assert.match(projectShellSource, /还没有项目/)
})

test('项目保存和删除失败会触发 toast 反馈', () => {
  const projectShellSource = readFirst([path.resolve(__dirname, '../components/project/ProjectWorkspaceShell.jsx')])
  assert.match(projectShellSource, /notifyError\('项目删除失败'/)
  assert.match(projectDetailPageSource, /notifyError\('项目保存失败'/)
  assert.doesNotMatch(projectDetailPageSource, /console\.warn\('project update failed'/)
})

test('旧项目子面板死代码已删除，项目详情使用当前研究桌面', () => {
  assert.equal(projectConversationsSource, '')
  assert.equal(fs.existsSync(path.resolve(__dirname, '../components/project/ProjectPaperBinder.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../components/project/ProjectPaperBoard.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../components/project/ProjectResearch.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../components/project/ProjectSessionFeed.jsx')), false)
  assert.match(projectDetailPageSource, /ProjectResearchDesk/)
})

test('项目图入口已删除，只保留全局图谱', () => {
  assert.equal(fs.existsSync(path.resolve(__dirname, '../components/project/ProjectGraph.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../app/(workspace)/projects/[projectId]/graph/page.jsx')), false)
})

test('图谱侧栏支持基于当前论文节点直接提问，并把研究请求送入研究工作区', () => {
  assert.match(paperGraphShellSource, /useConversationStore/)
  assert.match(paperGraphShellSource, /const ensureConversation = async/)
  assert.match(paperGraphShellSource, /router\.push\(`\/chat\/\$\{encodeURIComponent\(conversationId\)\}\?paperId=\$\{encodeURIComponent\(activeNode\.paper_id\)\}`\)/)
  assert.match(paperGraphShellSource, /buildResearchWorkspaceHref\(\{/)
  assert.match(paperGraphShellSource, /source:\s*'graph'/)
})

test('旧 library 路由会重定向到统一的 papers 入口', () => {
  assert.match(libraryRedirectSource, /redirect\('\/papers'\)/)
})

test('论文详情页已移除研究演化入口', () => {
  const paperHeaderSource = readFirst([path.resolve(__dirname, '../components/paper/PaperDetailHeader.jsx')])
  assert.doesNotMatch(paperHeaderSource, /查看研究演化/)
})

test('论文详情页使用 总览 / AI 分析 / 原文 三个主标签', () => {
  const paperPageSource = readFirst([path.resolve(__dirname, '../app/(workspace)/papers/[paperId]/page.jsx')])
  assert.match(paperPageSource, /overview/)
  assert.match(paperPageSource, /analysis/)
  assert.match(paperPageSource, /original/)
  assert.doesNotMatch(paperPageSource, /citations/)
  assert.doesNotMatch(paperPageSource, /graph/)
})

test('外部研究页保留启动器、列表和导出能力', () => {
  assert.match(researchHomeSource, /ResearchLauncher/)
  assert.match(researchHomeSource, /ResearchSessionList/)
  assert.match(researchSessionActionsSource, /ReportExport/)
  assert.match(researchProcessSectionSource, /研究过程/)
})

test('研究详情页拆成报告、证据、过程和操作区', () => {
  assert.match(researchSessionPageSource, /buildResearchSessionModel/)
  assert.match(researchSessionLayoutSource, /ResearchReportBody/)
  assert.match(researchSessionLayoutSource, /EvidenceDeskSection/)
  assert.match(researchSessionLayoutSource, /ResearchProcessSection/)
  assert.match(researchSessionActionsSource, /导出证据附录/)
  assert.match(researchSessionActionsSource, /导出过程 JSON/)
  assert.match(researchSessionSideSource, /保存研究结论/)
  assert.match(researchProcessSectionSource, /TraceReplayPanel/)
  assert.match(researchProcessSectionSource, /details/)
  assert.match(researchSessionViewSource, /renderResearchTraceAppendix/)
  assert.match(traceReplayPanelSource, /开始回放|重新开始|退出回放/)
  assert.match(researchTraceExportSource, /buildResearchTraceBundle/)
  assert.match(researchTraceExportSource, /renderResearchTraceAppendix/)
  assert.match(readFirst([path.resolve(__dirname, '../components/research/ReportExport.jsx')]), /mimeType/)
})

test('论文搜索页和每日速览页已作为主侧边栏入口建立', () => {
  assert.match(brandSource, /href: '\/paper-search'/)
  assert.match(brandSource, /label: '论文搜索'/)
  assert.match(brandSource, /href: '\/daily-brief'/)
  assert.match(brandSource, /label: '每日速览'/)
  assert.match(paperSearchPageSource, /论文搜索/)
  assert.match(paperSearchPageSource, /searchArxiv|arXiv/)
  assert.match(paperSearchPageSource, /送入 Deep Research/)
  assert.match(dailyBriefPageSource, /每日速览/)
  assert.doesNotMatch(dailyBriefPageSource, /占位页|还未展开实现/)
})

test('批 1 去工程化：观测页守卫、图谱解读和论文就绪时间线已接入', () => {
  const observabilitySource = readFirst([path.resolve(__dirname, '../app/(workspace)/observability/page.jsx')])
  const traceDetailSource = readFirst([path.resolve(__dirname, '../app/(workspace)/observability/[requestId]/page.jsx')])
  const tasksSource = readFirst([path.resolve(__dirname, '../app/(workspace)/tasks/page.jsx')])
  const developerAccessGateSource = readFirst([path.resolve(__dirname, '../components/auth/DeveloperAccessGate.jsx')])
  const readinessSource = readFirst([path.resolve(__dirname, '../components/project/ProjectPaperStatusList.jsx')])
  const notesSource = readFirst([path.resolve(__dirname, '../components/project/ProjectNotesPanel.jsx')])

  assert.match(observabilitySource, /DeveloperAccessGate/)
  assert.match(traceDetailSource, /DeveloperAccessGate/)
  assert.doesNotMatch(tasksSource, /DeveloperAccessGate/)
  assert.match(tasksSource, /resumeUploadTask/)
  assert.match(developerAccessGateSource, /当前账号没有访问权限/)
  assert.doesNotMatch(sideNavSource, /item\.roles|useAuthStore/)
  assert.match(paperGraphShellSource, /GraphInterpretationPanel/)
  assert.match(readinessSource, /PaperReadinessTimeline/)
  assert.match(notesSource, /formatNoteResourceLabels/)
  assert.doesNotMatch(notesSource, /paper \{note\.paper_id\}|session \{note\.session_id\}/)
})

test('discover 和 dashboard 页面已移除', () => {
  assert.equal(fs.existsSync(path.resolve(__dirname, '../app/(workspace)/discover/page.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../app/(workspace)/dashboard/page.jsx')), false)
})

test('顶部导航已移除旧知识地图与演化图入口', () => {
  assert.doesNotMatch(brandSource, /knowledge-map/)
  assert.doesNotMatch(brandSource, /知识地图/)
  assert.doesNotMatch(brandSource, /research-map|演化图/)
  assert.match(brandSource, /href: '\/graph'/)
  assert.match(brandSource, /label: '关系图谱'/)
})

test('一级导航已收敛为项目、知识库、研究、论文搜索、每日速览、任务、观测与设置', () => {
  assert.match(brandSource, /label: '项目'/)
  assert.match(brandSource, /label: '知识库'/)
  assert.match(brandSource, /label: '论文库'/)
  assert.match(brandSource, /label: '关系图谱'/)
  assert.match(brandSource, /label: '研究'/)
  assert.match(brandSource, /label: '对话'/)
  assert.match(brandSource, /label: '深度研究'/)
  assert.match(brandSource, /label: '研究演化'/)
  assert.match(brandSource, /label: '论文搜索'/)
  assert.match(brandSource, /label: '每日速览'/)
  assert.match(brandSource, /label: '任务'/)
  assert.match(brandSource, /label: '观测'/)
  assert.match(brandSource, /label: '设置'/)
  assert.doesNotMatch(brandSource, /roles: \['developer', 'admin'\]/)
  assert.doesNotMatch(brandSource, /label: 'Deep Research'/)
  assert.doesNotMatch(brandSource, /项目推荐|项目信号/)
})

test('旧 research-map 与 knowledge-map 路由已删除', () => {
  assert.equal(fs.existsSync(path.resolve(__dirname, '../app/(workspace)/research-map/page.jsx')), false)
  assert.equal(fs.existsSync(path.resolve(__dirname, '../app/(workspace)/knowledge-map/page.jsx')), false)
})

test('项目页不再把单库 default 当成首页核心指标', () => {
  const heroSource = readFirst([path.resolve(__dirname, '../components/project/ProjectHeroPanel.jsx')])
  const scopePanelSource = readFirst([path.resolve(__dirname, '../components/project/ProjectScopePanel.jsx')])
  assert.match(heroSource, /研究范围/)
  assert.doesNotMatch(heroSource, /作用库/)
  assert.match(scopePanelSource, /默认知识库范围/)
})

test('项目列表页已移除项目图入口', () => {
  const projectShellSource = readFirst([path.resolve(__dirname, '../components/project/ProjectWorkspaceShell.jsx')])
  assert.doesNotMatch(projectShellSource, /项目图|打开项目图|\/graph/)
})

test('项目详情页已新增研究笔记视图', () => {
  const projectDetailSource = readFirst([path.resolve(__dirname, '../app/(workspace)/projects/[projectId]/page.jsx')])
  const projectNotesSource = readFirst([path.resolve(__dirname, '../components/project/ProjectNotesPanel.jsx')])
  assert.match(projectDetailSource, /ProjectNotesPanel/)
  assert.match(projectNotesSource, /includeProjectPapers: true/)
  assert.match(projectNotesSource, /includeProjectSessions: true/)
})

test('聊天研究面板保存笔记时会附带工作台上下文', () => {
  const researchPanelSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPanel.jsx')])
  assert.match(researchPanelSource, /projectId: workspace\.projectId/)
  assert.match(researchPanelSource, /conversationId: workspace\.conversationId/)
  assert.match(researchPanelSource, /sessionId: workspace\.sessionId/)
  assert.match(researchPanelSource, /sourceType: action\.source_type \|\| 'local_rag'/)
})

test('内部研究消息会保留 dialogue_mode 和 synthesis_state，并在面板里展示整理状态', () => {
  const researchPanelSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPanel.jsx')])
  const evidencePanelSource = readFirst([path.resolve(__dirname, '../components/chat/MessageEvidencePanel.jsx')])
  assert.match(workspaceStoreSource, /synthesis_state:\s*data\?\.synthesis_state \|\| \{\}/)
  assert.match(workspaceStoreSource, /dialogue_mode:\s*data\?\.dialogue_mode \|\| ''/)
  assert.ok(MESSAGE_EXTRA_KEYS.includes('graph_hint_section'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('graph_hint_full'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('research_plan'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('research_unit_results'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('coverage_review'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('planner_trace'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('planner_final_blocks'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('planner_final_validation_errors'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('planner_final_retry_attempts'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('source_status'))
  assert.ok(MESSAGE_EXTRA_KEYS.includes('source_status_reasons'))
  // R2: research_steps 进白名单，历史消息可离线渲染分步卡；typewriter 故意不进白名单，避免回看重播
  assert.ok(MESSAGE_EXTRA_KEYS.includes('research_steps'))
  assert.ok(!MESSAGE_EXTRA_KEYS.includes('typewriter'))
  assert.match(workspaceStoreSource, /graph_hint_section:\s*data\?\.graph_hint_section \|\| ''/)
  assert.match(workspaceStoreSource, /graph_hint_full:\s*data\?\.graph_hint_full \|\| ''/)
  assert.match(workspaceStoreSource, /research_plan:\s*data\?\.research_plan \|\| null/)
  assert.match(workspaceStoreSource, /research_unit_results:\s*data\?\.research_unit_results \|\| \[\]/)
  assert.match(workspaceStoreSource, /planner_final_blocks:\s*data\?\.planner_final_blocks \|\| \[\]/)
  assert.match(workspaceStoreSource, /planner_final_validation_errors:\s*data\?\.planner_final_validation_errors \|\| \[\]/)
  assert.match(workspaceStoreSource, /source_status:\s*data\?\.source_status \|\| \{\}/)
  assert.match(workspaceStoreSource, /source_status_reasons:\s*data\?\.source_status_reasons \|\| \{\}/)
  // R2: 运行中 pending 占位 + done 时 research_steps 快照
  assert.match(workspaceStoreSource, /pending:\s*true/)
  assert.match(workspaceStoreSource, /research_steps:\s*researchStepsSnapshot/)
  assert.match(workspaceStoreSource, /typewriter:\s*plannerFinalOk/)
  // 答案主卡：Viewport 挂 MessageAnswerShell；步骤/正文/记忆在主卡内
  assert.match(messageViewportSource, /MessageAnswerShell/)
  const answerShellSource = readFirst([path.resolve(__dirname, '../components/chat/MessageAnswerShell.jsx')])
  assert.match(answerShellSource, /MessageResearchSteps/)
  assert.match(answerShellSource, /MessageContent/)
  assert.match(answerShellSource, /MessageResearchSteps[\s\S]*MessageContent/)
  assert.match(answerShellSource, /MessageMemoryCapture/)
  assert.match(answerShellSource, /compact/)
  assert.match(answerShellSource, /activePanel/)
  // 默认底栏：留下判断 / AI 整理 / 依据 / 过程；步骤在正文上方（4/4 可展开）
  assert.match(answerShellSource, /依据/)
  assert.match(answerShellSource, /过程/)
  assert.match(answerShellSource, /MessageResearchSteps/)
  assert.match(answerShellSource, /MessageContent/)
  assert.match(answerShellSource, /MessageResearchSteps[\s\S]*MessageContent/)
  assert.doesNotMatch(answerShellSource, /forceExpand/)
  assert.doesNotMatch(answerShellSource, />轨迹</)
  const memoryCaptureSource = readFirst([path.resolve(__dirname, '../components/chat/MessageMemoryCapture.jsx')])
  assert.match(memoryCaptureSource, /留下判断|快速保存判断/)
  assert.match(memoryCaptureSource, /AI 整理/)
  assert.match(memoryCaptureSource, /curateMemory/)
  assert.match(memoryCaptureSource, /createMemory/)
  assert.match(memoryCaptureSource, /conversation_id/)
  assert.match(memoryCaptureSource, /compact/)
  assert.match(researchPanelSource, /MessageResearchSynthesisSummary/)
  assert.match(synthesisSummarySource, /研究整理/)
  assert.match(synthesisSummarySource, /整理完成|证据不足/)
  assert.match(synthesisSummarySource, /已覆盖维度|待补维度/)
  assert.match(evidencePanelSource, /直接证据/)
  assert.match(evidencePanelSource, /GraphHintSection/)
  assert.match(evidencePanelSource, /图谱线索/)
  assert.match(evidencePanelSource, /线索，不可作为事实引用/)
  assert.match(evidencePanelSource, /evidence-\$\{safeAnchorSegment\(messageId \|\| 'message'\)\}-\$\{evidenceId\}/)
  // 答案优先：独立挂载默认折叠；主卡内 embedded 直接展开
  assert.match(evidencePanelSource, /useState\(Boolean\(embedded\)\)/)
  assert.match(evidencePanelSource, /LayerSection title="原文证据" records=\{directText\} defaultOpen=\{false\}/)
  assert.match(evidencePanelSource, /const citations = hasFinalDelivery\(message\)/)
  assert.match(evidencePanelSource, /finalCitationRecords\(message\)/)
  assert.match(evidencePanelSource, /UnifiedEvidenceView[\s\S]*citations=\{citations\}/)
  assert.doesNotMatch(evidencePanelSource, /citations\.slice\(0,\s*3\)/)
  assert.match(messageContentSource, /normalizeEvidenceLinks/)
  assert.match(messageContentSource, /#\$\{anchorPrefix\}-\$\{marker\}/)
  // R4: 定稿成功才打字机；失败态整块显示
  assert.match(messageContentSource, /useTypewriterText/)
  assert.match(messageContentSource, /source_status\?\.planner_final === 'ok'/)
  assert.match(messageContentSource, /message\?\.typewriter/)
  // 分步：进行中转圈 + 结束后可折叠
  const stepsSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchSteps.jsx')])
  assert.match(stepsSource, /sa-step-spin/)
  assert.match(stepsSource, /展开步骤|查看步骤/)
  assert.match(stepsSource, /useState\(false\)/)
  // 诊断默认折叠
  const plannerSummaryUiSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPlannerSummary.jsx')])
  assert.match(plannerSummaryUiSource, /查看诊断/)
  assert.match(plannerSummaryUiSource, /useState\(false\)/)
})

test('planner 研究过程视图复用研究面板并显示中文状态', () => {
  const researchPanelSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPanel.jsx')])
  assert.match(researchPanelSource, /MessageResearchPlannerProcess/)
  assert.match(researchPanelSource, /更多细节|查看研究细节/)
  assert.match(researchPanelSource, /MessageResearchPlannerSummary/)
  assert.match(researchPanelSource, /hasPlannerSummary/)
  assert.match(researchPanelSource, /evidenceRecords=\{message\?\.evidence_records \|\| \[\]\}/)
  assert.match(researchPanelSource, /messageId=\{message\?\.id\}/)
  assert.match(researchPanelSource, /!\s*model\.plannerSummaryAvailable \?/)
  assert.match(researchPanelSource, /<MessageResearchSynthesisSummary/)
  assert.match(plannerSummarySource, /function buildPlannerSummary/)
  assert.match(plannerSummarySource, /runtimeChain === 'planner'/)
  assert.match(plannerSummarySource, /取证 \{formatPlannerStatus\(plannerSummary\.plannerStatus\)\}/)
  assert.match(plannerSummarySource, /最终回答 \{plannerSummary\.finalStatusLabel \|\| formatPlannerStatus\(plannerSummary\.finalStatus\)\}/)
  assert.doesNotMatch(plannerSummarySource, /planner_final_blocks[\s\S]*\.text/)
  assert.match(plannerProcessSource, /研究过程/)
  assert.match(plannerProcessSource, /已找到证据/)
  assert.match(plannerProcessSource, /基于证据推断/)
  assert.match(plannerProcessSource, /证据不足/)
  assert.match(plannerProcessSource, /需要外部背景/)
  assert.match(plannerProcessSource, /无法执行/)
  assert.match(plannerProcessSource, /function EvidenceLinks/)
  assert.match(plannerProcessSource, /supporting_evidence_ids/)
  assert.match(plannerProcessSource, /related_evidence_ids/)
  assert.match(plannerProcessSource, /#evidence-\$\{safeAnchorSegment\(messageId \|\| 'message'\)\}-\$\{evidenceId\}/)
  assert.doesNotMatch(plannerProcessSource, /JSON\.stringify/)
  // R3: 研究过程细节默认折叠
  assert.match(plannerProcessSource, /useState\(false\)/)
  assert.match(plannerProcessSource, /查看细节/)
})

test('Planner failed-final messages expose final-only retry action through narrow API', () => {
  const ragRunsApiSource = readFirst([path.resolve(__dirname, 'api/rag-runs.js')])
  const plannerSummarySource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPlannerSummary.jsx')])
  const researchPanelSource = readFirst([path.resolve(__dirname, '../components/chat/MessageResearchPanel.jsx')])
  assert.match(ragRunsApiSource, /retry-final-answer/)
  assert.match(plannerSummarySource, /canRetryPlannerFinalAnswer/)
  // 重试资格放宽：取证 ok|partial，最终 failed|skipped|partial
  assert.match(plannerSummarySource, /\['ok', 'partial', 'insufficient', 'no_evidence'\]\.includes\(planner\)/)
  assert.match(plannerSummarySource, /plannerFinal === 'failed'/)
  assert.match(plannerSummarySource, /message\?\.run_id/)
  assert.match(researchPanelSource, /retryRagRunFinalAnswer/)
  assert.match(researchPanelSource, /deliveryResponsePatch\(result\)/)
})

test('论文图谱侧栏支持保存图谱观察为研究笔记', () => {
  const paperGraphTabSource = readFirst([path.resolve(__dirname, '../components/paper/PaperGraphTab.jsx')])
  assert.match(paperGraphTabSource, /保存图谱观察/)
  assert.match(paperGraphTabSource, /defaultNoteType=.*comparison_note|reading_note/)
})

test('项目总览条优先展示论文状态和最近研究，而不是旧分发动作', () => {
  const heroSource = readFirst([path.resolve(__dirname, '../components/project/ProjectHeroPanel.jsx')])
  assert.match(heroSource, /项目研究桌面/)
  assert.match(heroSource, /总论文数/)
  assert.match(heroSource, /已完成分析/)
  assert.match(heroSource, /待补分析/)
  assert.match(heroSource, /最近一次内部研究/)
  assert.doesNotMatch(heroSource, /工作台动作|打开内部执行台|项目推荐|项目信号/)
})

test('项目详情页以顶部标签组织总览、论文、笔记和活动', () => {
  const researchDeskSource = readFirst([path.resolve(__dirname, '../components/project/ProjectResearchDesk.jsx')])
  const activityFeedSource = readFirst([path.resolve(__dirname, '../components/project/ProjectActivityFeed.jsx')])
  const paperStatusSource = readFirst([path.resolve(__dirname, '../components/project/ProjectPaperStatusList.jsx')])
  assert.match(projectDetailPageSource, /PROJECT_TABS/)
  assert.match(projectDetailPageSource, /总览|论文|研究笔记|活动/)
  assert.match(researchDeskSource, /开始新的内部研究/)
  assert.match(researchDeskSource, /补一轮外部研究/)
  assert.doesNotMatch(researchDeskSource, /查看统一研究页/)
  assert.match(activityFeedSource, /项目动态/)
  assert.match(activityFeedSource, /论文加入项目|AI 分析完成|内部研究启动/)
  assert.match(activityFeedSource, /刷新动态/)
  assert.match(paperStatusSource, /待分析论文|已完成分析/)
  assert.doesNotMatch(paperStatusSource, /加入当前项目/)
})

test('研究首页已改成外部研究页', () => {
  assert.match(researchHomeSource, /title="外部研究"/)
  assert.match(researchHomeSource, /useSearchParams/)
  assert.match(researchHomeSource, /prefill=\{prefill\}/)
  assert.match(researchHomeSource, /ResearchLauncher/)
  assert.doesNotMatch(researchHomeSource, /UnifiedResearchWorkbench/)
  assert.match(researchLauncherSource, /输入研究主题/)
  assert.match(researchLauncherSource, /独立研究流程|独立会话/)
})

test('项目 store 提供项目工作面和动态的显式刷新动作', () => {
  assert.match(projectStoreSource, /async refreshProjectData\(projectId = state\.activeProjectId\)/)
  assert.match(projectStoreSource, /async refreshActivities\(projectId = state\.activeProjectId\)/)
  assert.match(projectDetailPageSource, /刷新项目工作面/)
})

test('论文详情页已拆成总览、AI 分析和原文三层内容', () => {
  const analysisTabSource = readFirst([path.resolve(__dirname, '../components/paper/PaperAnalysisTab.jsx')])
  const originalTabSource = readFirst([path.resolve(__dirname, '../components/paper/PaperOriginalTab.jsx')])
  assert.match(paperDetailPageSource, /label: '总览'/)
  assert.match(paperDetailPageSource, /label: 'AI 分析'/)
  assert.match(paperDetailPageSource, /label: '原文'/)
  assert.match(paperOverviewTabSource, /基础信息|摘要/)
  assert.match(paperOverviewTabSource, /总览页只保留基础信息和原始摘要/)
  assert.match(analysisTabSource, /AI 分析|方法重点|参考文献/)
  assert.match(originalTabSource, /原始 PDF|Markdown 解析文件|回源说明/)
})

test('图谱页会对低信号或空图给出解释和下一步动作', () => {
  assert.match(paperGraphShellSource, /const hasLowSignalGraph = !busy && visibleNodes\.length > 0 && !visibleEdges\.length/)
  assert.match(paperGraphShellSource, /当前图里只有论文节点，还没有高置信关系边/)
  assert.match(paperGraphShellSource, /不是坏数据，也不是完整图/)
  assert.match(paperGraphShellSource, /重置筛选/)
  assert.match(paperGraphShellSource, /回项目工作台|去绑定论文|去每日速览|进入外部研究/)
})

test('聊天页已删除项目选择与项目 URL 承接，但保留当前项目上下文', () => {
  assert.doesNotMatch(chatWorkspaceShellSource, /ProjectSwitcher/)
  assert.doesNotMatch(chatWorkspaceShellSource, /searchParams\.get\('projectId'\)/)
  assert.match(chatWorkspaceShellSource, /projectId:\s*activeProjectId/)
})

test('聊天页头部显示当前项目 chip，并提供项目页跳转入口', () => {
  const headerSource = readFirst([path.resolve(__dirname, '../components/chat/ChatHeader.jsx')])
  assert.match(chatWorkspaceShellSource, /projectStore\.activeProject\?\.name/)
  assert.match(headerSource, /当前项目：\s*\{projectName\}/)
  assert.match(headerSource, /`\/projects\/\$\{encodeURIComponent\(projectId\)\}`|\/projects/)
})

test('对话侧栏按项目作用域组织研究会话和未归属会话', () => {
  assert.match(sidebarSource, /项目范围/)
  assert.match(sidebarSource, /未归属/)
  assert.match(sidebarSource, /buildProjectSessionItems/)
  assert.match(sidebarSource, /buildUnassignedConversationItems/)
  assert.doesNotMatch(sidebarSource, /ActiveSessionOutputs/)
  assert.doesNotMatch(sidebarSource, /统一对话池/)
})

test('品牌与顶栏文案已改为新名称，并去掉主工作流提示', () => {
  const topbarSource = readFirst([path.resolve(__dirname, '../components/layout/AppTopBar.jsx')])
  assert.match(brandSource, /name: '研途'/)
  assert.match(brandSource, /shortName: '研'/)
  assert.doesNotMatch(sideNavSource, /主工作流/)
  assert.match(topbarSource, /currentNav\?\.label/)
  assert.doesNotMatch(topbarSource, /主工作流/)
})

test('论文库页仍承接 URL 中的检索词和项目上下文', () => {
  assert.match(libraryPageSource, /searchParams\.get\('projectId'\)/)
  assert.match(libraryPageSource, /searchParams\.get\('q'\)/)
  assert.match(libraryPageSource, /initialQuery=\{queryParam\}/)
})

test('论文库和引用链入口已改为新论文图页面', () => {
  const citationsSource = readFirst([path.resolve(__dirname, '../components/paper/PaperCitationsTab.jsx')])
  const libraryShellSource = readFirst([path.resolve(__dirname, '../components/library/LibraryPageShell.jsx')])
  assert.match(libraryShellSource, /router\.push\('\/graph'\)/)
  assert.match(citationsSource, /查看论文图/)
  assert.doesNotMatch(citationsSource, /research-map|查看研究演化/)
})

import './upload-recovery.test.mjs'
