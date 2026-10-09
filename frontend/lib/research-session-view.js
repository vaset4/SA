import { externalResearchModeLabel, normalizeExternalResearchMode } from './external-research-mode'
import { buildResearchTraceBundle, renderResearchTraceAppendix } from './research-trace-export'
import { describeDelivery, describePersistence, sessionDeliveryValues } from './result-delivery.mjs'

export const DEVELOPER_ROLES = new Set(['developer', 'admin'])

export function asList(value) {
  return Array.isArray(value) ? value : []
}

/** @param {object} session 会话响应 @returns {object} 正式提交结果，执行checkpoint不代替已保存正文 */
export function readSessionValues(session) { return sessionDeliveryValues(session) }

export function formatSessionTime(value) {
  if (!value) return '时间未知'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '时间未知'
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

export function researchModeLabel(mode) {
  return externalResearchModeLabel(normalizeExternalResearchMode(mode))
}

function clipText(value, limit = 900) {
  const text = `${value || ''}`.trim()
  if (!text) return ''
  return text.length > limit ? `${text.slice(0, limit)}...` : text
}

function appendListSection(sections, title, values, limit) {
  if (!values.length) return
  sections.push([`${title}：`, ...values.slice(0, limit).map((item) => `- ${item}`)].join('\n'))
}

export function buildResearchNoteDraft({ report, keyFindings, externalGaps, followUpQuestions, candidatePapersToImport }) {
  const sections = []
  if (report) sections.push(`研究结论：\n${clipText(report)}`)
  appendListSection(sections, '关键发现', keyFindings, 4)
  appendListSection(sections, '外部空白', externalGaps, 3)
  appendListSection(sections, '后续追问', followUpQuestions, 3)
  if (candidatePapersToImport.length) {
    sections.push(['候选补库：', ...candidatePapersToImport.slice(0, 3).map((item) => `- ${item.title || item.url || '候选来源'}`)].join('\n'))
  }
  return sections.join('\n\n').trim()
}

export function deriveSnapshotEvents(session, values) {
  const events = []
  if (values.intent) events.push({ stage: 'route', detail: `已路由到 ${values.intent}` })
  if (values.research_mode) events.push({ stage: 'run_header', research_mode: values.research_mode, paper_count: asList(values.selected_papers).length })
  if (asList(values.selected_papers).length) events.push({ stage: 'paper_set_resolved', result_count: asList(values.selected_papers).length })
  if (asList(values.evidence_cards).length) events.push({ stage: 'evidence_accumulated', evidence_count: asList(values.evidence_cards).length })
  asList(values.reasoning_summaries).slice(0, 2).forEach((item) => {
    events.push({ stage: 'reasoning_summary', summary: item.summary, next_action: item.next_action })
  })
  if (session?.interrupts?.length) events.push({ stage: 'clarify', detail: '存在待处理澄清/中断' })
  if (values.awaiting_clarification) events.push({ stage: 'done', detail: values.clarification_question || values.report || '等待补充澄清' })
  else if (values.report) events.push({ stage: 'done', detail: '研究报告已生成' })
  return events
}

export function buildResearchStages(values, liveTrace, status = '') {
  const awaitingClarification = status === 'awaiting_clarification'
  const reportReady = Boolean(values.answer || values.report || values.clarification_question)
  const delivery = describeDelivery(values.final_delivery_report)
  const selectedPapers = asList(values.selected_papers)
  const evidenceCards = asList(values.evidence_cards)
  const reasoningSummaries = asList(values.reasoning_summaries)
  const traceStatus = liveTrace?.status || ''
  return [
    { id: 'config', label: '研究配置', done: Boolean(values.research_mode || liveTrace?.researchMode), active: traceStatus === 'queued', detail: `${researchModeLabel(values.research_mode || liveTrace?.researchMode)} · 背景论文 ${selectedPapers.length || liveTrace?.paperIds?.length || 0} 篇` },
    { id: 'brief_context', label: 'Brief 上下文', done: Boolean(selectedPapers.length), active: traceStatus === 'routing', detail: selectedPapers.length ? `已装载 ${selectedPapers.length} 篇项目背景论文` : '当前未绑定项目论文背景' },
    { id: 'evidence', label: '证据汇聚', done: Boolean(evidenceCards.length || values.research_notes?.length), active: traceStatus === 'streaming', detail: evidenceCards.length ? `已装配 ${evidenceCards.length} 张外部证据卡` : '搜索与外部证据整合中' },
    { id: 'reasoning', label: '推理收束', done: Boolean(reasoningSummaries.length || asList(values.key_findings).length), active: traceStatus === 'streaming' && !reportReady, detail: reasoningSummaries.length ? `推理摘要 ${reasoningSummaries.length} 条` : '等待关键发现与不确定性收束' },
    { id: 'report', label: awaitingClarification ? '等待澄清' : delivery?.label || '研究结果', done: reportReady, active: awaitingClarification, detail: awaitingClarification ? '需要用户补充信息' : reportReady ? '可查看实际正文与交付说明' : '尚无已提交结果' },
  ]
}

export function buildResearchSessionModel({ sessionId, session, liveTrace }) {
  const values = readSessionValues(session)
  const report = values.answer ?? values.report ?? values.research_brief ?? values.clarification_question ?? ''
  const researchMode = normalizeExternalResearchMode(session?.research_mode || values.research_mode || liveTrace?.researchMode || 'external_only')
  const model = baseSessionModel({ sessionId, session, values, report, researchMode, liveTrace })
  const traceBundle = buildResearchTraceBundle(model.traceBundleInput)
  return {
    ...model,
    traceBundleJson: JSON.stringify(traceBundle, null, 2),
    traceAppendixMarkdown: renderResearchTraceAppendix(traceBundle),
  }
}

function baseSessionModel({ sessionId, session, values, report, researchMode, liveTrace }) {
  const traceEvents = liveTrace?.events?.length ? liveTrace.events : deriveSnapshotEvents(session, values)
  const paperIds = asList(session?.paper_ids).map((item) => String(item || '').trim()).filter(Boolean)
  const lists = sessionLists(values, session, liveTrace, paperIds)
  const status = resolveSessionStatus(session, values)
  const stages = buildResearchStages(values, liveTrace, status)
  return {
    sessionId,
    values,
    report,
    researchMode,
    traceEvents,
    stages,
    ...lists,
    projectId: `${session?.project_id || values.project_id || values?.structured_query?.project_id || ''}`.trim(),
    paperIds,
    threadId: `${session?.thread_id || values.thread_id || ''}`.trim(),
    conversationId: `${session?.conversation_id || values.conversation_id || ''}`.trim(),
    status,
    delivery: describeDelivery(values.final_delivery_report),
    persistence: describePersistence(session?.persistence_status ? session : values),
    resultRunId: session?.result_run_id || values.target_run_id || values.run_id || '',
    cancelRequested: session?.cancel_requested === true,
    resultMessage: { ...values, id: values.message_id || session?.result_run_id || `research-${sessionId}`,
      role: 'assistant', content: report, renderMode: 'markdown' },
    allowedActions: asList(session?.allowed_actions),
    lastError: `${session?.last_error || ''}`.trim(),
    attemptCount: Number(session?.attempt_count || 0),
    updatedAt: formatSessionTime(session?.updated_at || session?.created_at),
    title: session?.topic || values.user_query || `研究会话 ${sessionId}`,
    traceBundleInput: { sessionId, session, values, researchMode, stages, traceEvents, report, ...lists },
  }
}

function resolveSessionStatus(session, values) {
  const explicitStatus = `${session?.status || values.status || ''}`.trim()
  if (explicitStatus) return explicitStatus
  if (values.awaiting_clarification) return 'awaiting_clarification'
  if (values.report || values.research_brief) return 'completed'
  return 'created'
}

function sessionLists(values, session, liveTrace, paperIds) {
  const selectedPapers = asList(values.selected_papers)
  const selectedById = new Map(selectedPapers.map((paper) => [String(paper?.paper_id || '').trim(), paper]))
  const scopedPapers = paperIds.length
    ? paperIds.map((paperId) => selectedById.get(paperId) || { paper_id: paperId })
    : selectedPapers
  return {
    chatHistory: asList(values.chat_history || session?.chat_history),
    selectedPapers: scopedPapers,
    evidenceCards: asList(values.evidence_cards),
    reasoningSummaries: asList(values.reasoning_summaries),
    sourceRecords: asList(values.source_records),
    keyFindings: asList(values.key_findings || liveTrace?.keyFindings),
    uncertainties: asList(values.uncertainties || liveTrace?.uncertainties),
    ideaCandidates: asList(values.idea_candidates || liveTrace?.ideaCandidates),
    externalGaps: asList(values.external_gaps),
    followUpQuestions: asList(values.follow_up_questions),
    candidatePapersToImport: asList(values.candidate_papers_to_import),
  }
}

export function headerStatus(state, values) {
  if (state.error) return { tone: 'danger', label: '加载失败' }
  if (state.loading) return { tone: 'warning', label: '加载中' }
  const current = values || {}
  const status = typeof current === 'string' ? current : current.status || (current.awaiting_clarification ? 'awaiting_clarification' : '')
  if (current.cancelRequested && status === 'running') return { tone: 'warning', label: '正在停止' }
  if (status === 'awaiting_clarification') return { tone: 'warning', label: '等待澄清' }
  if (status === 'created' || status === 'running') return { tone: 'warning', label: status === 'created' ? '已创建' : '研究进行中' }
  if (status === 'failed' || status === 'timed_out') return { tone: 'danger', label: status === 'timed_out' ? '运行超时' : '研究失败' }
  if (status === 'cancelled') return { tone: 'neutral', label: '已取消' }
  if (current.persistence && current.persistence.label !== '已保存') return { tone: 'danger', label: current.persistence.label }
  return current.delivery || describeDelivery(current.final_delivery_report) || { tone: 'neutral', label: '研究已结束' }
}

export function buildNotePayload(model) {
  return {
    projectId: model.projectId,
    sessionId: model.sessionId,
    evidenceReference: {
      key_findings: model.keyFindings.slice(0, 4),
      candidate_papers: model.candidatePapersToImport.slice(0, 3).map((item) => ({
        title: item.title || '',
        url: item.url || '',
      })),
    },
  }
}
