export const LIMITS = { abstract: 320, list: 50 }
export const DEFAULT_DRAFT = '请结合当前论文库，先给我一个值得继续探索的研究方向。'
export const RENDER_MODE_PLAIN = 'plain'
export const RENDER_MODE_MARKDOWN = 'markdown'
export const MESSAGE_EXTRA_KEYS = [
  'citations',
  'paper_sources',
  'chunk_citations',
  'graph_evidence',
  'graph_hint_section',
  'graph_hint_full',
  'evidence_summary',
  'answer_confidence',
  'research_scope',
  'research_objects',
  'note_actions',
  'synthesis_state',
  'dialogue_mode',
  'evidence_records',
  'answer_brief',
  'missing_evidence',
  'research_plan',
  'research_unit_results',
  'coverage_review',
  'planner_trace',
  'planner_final_blocks',
  'planner_final_validation_errors',
  'final_delivery_report',
  'final_citation_bindings',
  'final_generation_attempt',
  'final_delivery_applicability',
  'turn_outcome',
  'planner_final_retry_attempts',
  'trace_summary',
  'runtime_chain',
  'source_status',
  'source_status_reasons',
  'request_id',
  'run_id',
  'run_status',
  'delivery_revision',
  'target_run_id',
  'producer_run_id',
  'persistence_status',
  'persistence_error',
  'can_retry_save',
  // 跨论文流式分步卡：done 时把 researchTrace.events 快照写入，历史消息可离线渲染
  'research_steps',
  // 项目科研回合级字段：回答卡按本轮真实深度差异化展示，并显示记忆读取与候选写回的真实终态。
  // 不在这份清单里的键即使后端下发也会被 pickMessageExtras 丢掉，历史消息就重现不出本轮结果
  'turn_depth',
  'depth_transitions',
  'memory_read_result',
  'memory_writeback',
]

const MARKDOWN_TITLE_HINTS = new Set([
  'Research result',
  'Research clarification',
  'Research stream',
  'Literature Review',
])

export const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  title: 'Workspace ready',
  content: '左侧继续当前项目会话，中间围绕问题做内部研究，右侧持续查看研究轨迹和论文上下文。',
  tone: 'accent',
  renderMode: RENDER_MODE_PLAIN,
}

export function createId(prefix) {
  const fallback = `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? fallback}`
}

export function joinText(values, fallback) {
  return values?.length ? values.join(' · ') : fallback
}

export function clip(value, size = 180) {
  if (!value) return '暂无内容。'
  return value.length > size ? `${value.slice(0, size).trim()}…` : value
}

export function normalizeRenderMode(renderMode) {
  if (renderMode === RENDER_MODE_MARKDOWN) return RENDER_MODE_MARKDOWN
  if (renderMode === RENDER_MODE_PLAIN) return RENDER_MODE_PLAIN
  return undefined
}

export function sanitizeMessageContent(content) {
  if (!content) return content
  return content.replace(/\[paper_id=[^\]]+\]/g, '')
}

export function hasMarkdownSyntax(content = '') {
  if (!content || typeof content !== 'string') return false
  return [
    /^#{1,6}\s/m,
    /^>\s/m,
    /^(\s*[-*+]\s|\s*\d+\.\s)/m,
    /```/,
    /\\\(.+?\\\)/s,
    /\\\[.+?\\\]/s,
    /\$\$.+?\$\$/s,
    /\[[^\]]+\]\((https?:\/\/|mailto:)[^)]+\)/,
    /^\|.+\|\s*$/m,
  ].some((pattern) => pattern.test(content))
}

function inferRenderMode(role, title = '', content = '') {
  if (role !== 'assistant') return RENDER_MODE_PLAIN
  if (MARKDOWN_TITLE_HINTS.has(title)) return RENDER_MODE_MARKDOWN
  if (hasMarkdownSyntax(content)) return RENDER_MODE_MARKDOWN
  return RENDER_MODE_PLAIN
}

export function pickMessageExtras(item = {}) {
  const extras = {}
  const metadata = item?.metadata && typeof item.metadata === 'object' ? item.metadata : {}
  for (const key of MESSAGE_EXTRA_KEYS) {
    const value = item?.[key] !== undefined ? item[key] : metadata[key]
    if (value !== undefined) {
      extras[key] = value
    }
  }
  return extras
}

export function resolveMessageRenderMode(message = {}) {
  const explicit = normalizeRenderMode(message.renderMode ?? message.render_mode)
  if (explicit) return explicit
  return inferRenderMode(message.role, message.title, sanitizeMessageContent(message.content || ''))
}

export function toMessage(role, content, title = '', tone, renderMode, extras = {}) {
  const resolvedRenderMode = normalizeRenderMode(renderMode)
    ?? inferRenderMode(role, title, sanitizeMessageContent(content || ''))
  return {
    id: createId(role),
    role,
    content,
    ...(title ? { title } : {}),
    ...(tone ? { tone } : {}),
    renderMode: resolvedRenderMode,
    ...extras,
  }
}

export function patchMessage(items, id, updater) {
  return items.map((item) => item.id === id ? updater(item) : item)
}

export function getError(reason, fallback = '请求失败') {
  return reason instanceof Error ? reason.message : fallback
}

export function buildPaperContext(paper) {
  if (!paper) return ''
  return [
    `当前关注论文：${paper.title}`,
    `作者：${joinText(paper.authors, '未知作者')}`,
    `摘要：${clip(paper.abstract || '暂无摘要', LIMITS.abstract)}`,
  ].join('\n')
}

export function buildChatHistory(messages, prompt) {
  return [
    ...messages
      .filter((item) => item.role === 'user' || item.role === 'assistant')
      .map((item) => ({ role: item.role, content: item.content })),
    { role: 'user', content: prompt },
  ]
}

export function hasConversationContent(messages = []) {
  if (!Array.isArray(messages)) return false
  return messages.some((message) => {
    const id = String(message?.id || '')
    return id !== 'welcome' && id !== 'restored'
  })
}

export function resolveComposerModeChange(nextMode) {
  if (nextMode === 'research') return { mode: 'research', researchMode: 'plan' }
  return { mode: nextMode }
}

export function selectNextPaperId(items, preferred) {
  if (!Array.isArray(items) || !items.length) return ''
  return items.some((item) => item.paper_id === preferred) ? preferred : items[0]?.paper_id ?? ''
}

function upgradeRestoredAssistantMessage(history, session, snapshotValues) {
  const fullAssistantContent = (
    session?.report
    || session?.answer
    || snapshotValues?.report
    || snapshotValues?.answer
    || session?.clarification_question
    || snapshotValues?.clarification_question
    || ''
  ).trim()
  const snapshotExtras = pickMessageExtras(snapshotValues)

  if (!fullAssistantContent || !Array.isArray(history) || history.length === 0) {
    return history
  }

  const upgraded = [...history]
  for (let index = upgraded.length - 1; index >= 0; index -= 1) {
    const item = upgraded[index]
    if (item?.role !== 'assistant') continue
    const preview = `${item.content || ''}`.trim()
    const previewPrefix = preview.endsWith('...') ? preview.slice(0, -3) : preview
    if (!preview || fullAssistantContent === preview || fullAssistantContent.startsWith(previewPrefix)) {
      upgraded[index] = { ...item, ...snapshotExtras, content: fullAssistantContent }
      return upgraded
    }
  }

  return history
}

export function restoredMessages({ session, pending, welcome = WELCOME }) {
  const snapshotValues = session?.values && typeof session.values === 'object' ? session.values : {}
  const history = Array.isArray(session?.chat_history)
    ? session.chat_history
    : Array.isArray(snapshotValues?.chat_history)
      ? snapshotValues.chat_history
      : []
  const hydratedHistory = upgradeRestoredAssistantMessage(history, session, snapshotValues)
  const resolvedSessionId = session?.session_id || snapshotValues?.session_id || ''
  const restored = hydratedHistory.map((item, index) => {
    const role = item.role === 'assistant' ? 'assistant' : 'user'
    const title = typeof item.title === 'string' ? item.title : ''
    const tone = typeof item.tone === 'string' ? item.tone : undefined
    const content = item.content
    const extras = role === 'assistant' ? pickMessageExtras(item) : {}
    return {
      id: `restored-${index}`,
      role,
      content,
      ...(title ? { title } : {}),
      ...(tone ? { tone } : {}),
      ...extras,
      renderMode: resolveMessageRenderMode({
        role,
        title,
        tone,
        content,
        renderMode: item.renderMode ?? item.render_mode,
      }),
    }
  })
  return restored.length
    ? [{
      ...welcome,
      id: 'restored',
      title: 'Session restored',
      content: `已恢复会话 ${resolvedSessionId}。`,
      tone: 'accent',
      renderMode: RENDER_MODE_PLAIN,
    }, ...restored]
    : [welcome]
}
