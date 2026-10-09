'use client'

import { useRef, useState } from 'react'

import { addNote } from '../../lib/api/papers'
import { retryRagRunFinalAnswer } from '../../lib/api/rag-runs'
import { createId, getError } from '../../lib/study-assistant-workspace.mjs'
import { deliveryResponsePatch } from '../../lib/result-delivery.mjs'
import { useWorkspaceStore } from '../../stores/workspace-store'
import { MessageAnswerBriefSection } from './MessageAnswerBriefSection'
import { MessageResearchPlannerProcess } from './MessageResearchPlannerProcess'
import { MessageResearchPlannerSummary, hasPlannerSummary, canRetryPlannerFinalAnswer } from './MessageResearchPlannerSummary'
import { MessageResearchSynthesisSummary, normalizeSynthesisState } from './MessageResearchSynthesisSummary'

const PANEL_STYLE = {
  marginTop: 10,
  border: '1px solid var(--sa-border)',
  borderRadius: 10,
  padding: 12,
  background: 'var(--sa-bg)',
}

function normalizeSectionItems(raw) {
  if (!Array.isArray(raw)) return []
  return raw.filter((item) => item && typeof item === 'object' && `${item.label || ''}`.trim())
}

function normalizeSections(raw) {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((item) => item && typeof item === 'object' && `${item.label || ''}`.trim())
    .map((item) => ({
      ...item,
      items: normalizeSectionItems(item.items),
    }))
    .filter((item) => item.items.length > 0)
}

function normalizeActions(raw) {
  if (!Array.isArray(raw)) return []
  return raw.filter((item) => item && typeof item === 'object' && `${item.label || ''}`.trim() && `${item.note || ''}`.trim())
}

function hasObjectFields(raw) {
  return raw && typeof raw === 'object' && Object.keys(raw).length > 0
}

function buildActionKey(action) {
  return `${action.key || action.label}-${action.type || 'action'}`
}

async function copyText(value) {
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  throw new Error('当前浏览器不支持剪贴板写入')
}

async function saveResearchNote(action, workspace) {
  const paperId = `${action.target_paper_id || ''}`.trim()
  if (!paperId) {
    throw new Error('当前动作缺少目标论文，无法直接保存。')
  }
  await addNote({
    paperId,
    projectId: workspace.projectId,
    conversationId: workspace.conversationId,
    sessionId: workspace.sessionId,
    content: `${action.note}`.trim(),
    noteType: action.note_type || 'reading_note',
    sourceType: action.source_type || 'local_rag',
    citation: action.citation || '',
    excerpt: action.excerpt || '',
    evidenceReference: action.evidence_reference || {},
  })
}

async function executeResearchAction(action, workspace) {
  if (action.type === 'save_note') {
    await saveResearchNote(action, workspace)
    return '已保存到研究笔记。'
  }
  await copyText(`${action.note}`.trim())
  return '已复制到剪贴板。'
}

async function runResearchAction({ action, workspace, setBusyKey, setStatus }) {
  const actionKey = buildActionKey(action)
  setBusyKey(actionKey)
  setStatus('')
  try {
    setStatus(await executeResearchAction(action, workspace))
  } catch (reason) {
    setStatus(getError(reason, '笔记动作执行失败'))
  } finally {
    setBusyKey('')
  }
}

/** @param {object} options 消息、工作区与状态更新端口 @returns {Promise<void>} 保存同次重试投影并显示真实结果 */
async function retryFinalAnswer({ message, workspace, setBusyKey, setStatus, request }) {
  const actionKey = `retry-final-${message?.run_id || ''}`
  const identity = `${message.run_id}:${message.delivery_revision || 0}`
  if (request.current?.identity !== identity) request.current = { identity, key: createId('final') }
  setBusyKey(actionKey)
  setStatus('')
  try {
    const result = await retryRagRunFinalAnswer({
      conversationId: workspace.conversationId,
      runId: message.run_id, idempotencyKey: request.current.key,
    })
    if (['failed', 'cancelled', 'timed_out'].includes(result.operation?.status)) request.current = null
    workspace.applyResultDelivery({ conversationId: workspace.conversationId,
      messageId: message.id, runId: message.run_id, patch: deliveryResponsePatch(result) })
    request.current = null
    if (result.persistence_status && result.persistence_status !== 'saved') {
      setStatus(result.response?.persistence_error || '回答已生成但尚未保存，请使用重试保存')
      return
    }
    // 1. 按 retry_status 给用户可读反馈；failed 但正文已更新也要说清楚
    const retryStatus = `${result?.retry_status || ''}`.trim()
    if (retryStatus === 'ok') {
      setStatus('最终回答已恢复。')
    } else if (retryStatus === 'partial') {
      setStatus('最终回答已部分恢复，请结合证据核对。')
    } else if (retryStatus === 'skipped') {
      setStatus('本次仍缺少可交付的内容，请查看具体缺口。')
    } else if (retryStatus === 'failed') {
      setStatus('重试完成，但校验仍未通过；若正文有更新请结合证据核对。')
    } else {
      setStatus('最终回答已更新。')
    }
  } catch (reason) {
    // 2. 后端 422 detail 已在 Error.message 中，直接展示给人
    setStatus(getError(reason, '最终回答重试失败'))
  } finally {
    setBusyKey('')
  }
}

function buildPanelModel(message) {
  const scope = hasObjectFields(message?.research_scope) ? message.research_scope : null
  const sections = normalizeSections(message?.research_objects)
  const actions = normalizeActions(message?.note_actions)
  const synthesis = normalizeSynthesisState(message?.synthesis_state)
  const answerBrief = hasObjectFields(message?.answer_brief) ? message.answer_brief : null
  const researchPlan = hasObjectFields(message?.research_plan) ? message.research_plan : null
  const unitResults = Array.isArray(message?.research_unit_results) ? message.research_unit_results : []
  const coverageReview = hasObjectFields(message?.coverage_review) ? message.coverage_review : null
  const plannerSummaryAvailable = hasPlannerSummary(message, researchPlan)
  return { scope, sections, actions, synthesis, answerBrief, researchPlan, unitResults, coverageReview, plannerSummaryAvailable }
}

function shouldRenderPanel(model) {
  return Boolean(
    model.scope
    || model.sections.length
    || model.actions.length
    || model.synthesis
    || model.answerBrief
    || model.researchPlan
    || model.plannerSummaryAvailable
  )
}

function ResearchStructure({ scope, hasTrailingContent }) {
  if (!scope) return null
  return (
    <div className="sa-stack" style={{ marginBottom: hasTrailingContent ? 12 : 0 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>研究结构</p>
        {scope.label ? <span className="sa-chip" style={{ fontSize: 12 }}>{scope.label}</span> : null}
        {scope.confidence ? <span className="sa-chip" style={{ fontSize: 12 }}>置信度 {scope.confidence}</span> : null}
      </div>
      {scope.summary ? <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>{scope.summary}</p> : null}
      {Array.isArray(scope.paper_ids) && scope.paper_ids.length ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
          涉及论文: {scope.paper_ids.join(', ')}
        </p>
      ) : null}
    </div>
  )
}

function ResearchObjectSections({ sections }) {
  if (!sections.length) return null
  return (
    <div className="sa-stack">
      {sections.map((section) => (
        <div key={section.key || section.label} className="sa-stack">
          <p className="sa-section-title" style={{ marginBottom: 0 }}>{section.label}</p>
          <div style={{ display: 'grid', gap: 8 }}>
            {section.items.slice(0, 4).map((item, index) => (
              <ResearchObjectCard key={`${section.key || section.label}-${item.label}-${index}`} item={item} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function ResearchObjectCard({ item }) {
  return (
    <div className="sa-tool-card" style={{ padding: 10 }}>
      <div style={{ fontWeight: 600 }}>{item.label}</div>
      {item.detail ? <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)' }}>{item.detail}</p> : null}
      {Array.isArray(item.paper_ids) && item.paper_ids.length ? (
        <p style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--sa-text-faint)' }}>
          证据论文: {item.paper_ids.join(', ')}
        </p>
      ) : null}
    </div>
  )
}

function ResearchNoteActions(props) {
  if (!props.actions.length) return null
  return (
    <div className="sa-stack" style={{ marginTop: props.hasLeadingContent ? 12 : 0 }}>
      <p className="sa-section-title" style={{ marginBottom: 0 }}>笔记动作</p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {props.actions.map((action) => (
          <ResearchNoteActionButton key={buildActionKey(action)} action={action} {...props} />
        ))}
      </div>
    </div>
  )
}

function ResearchNoteActionButton({ action, busyKey, workspace, setBusyKey, setStatus }) {
  const actionKey = buildActionKey(action)
  const busy = busyKey === actionKey
  return (
    <button
      type="button"
      className="sa-button sa-button--ghost"
      onClick={() => runResearchAction({ action, workspace, setBusyKey, setStatus })}
      disabled={Boolean(busyKey)}
    >
      {busy ? '处理中' : action.label}
    </button>
  )
}

function ResearchActionStatus({ status }) {
  if (!status) return null
  return <p style={{ margin: '10px 0 0', fontSize: 12, color: 'var(--sa-text-faint)' }}>{status}</p>
}

/** @param {object} props 当前研究与笔记操作 @returns {JSX.Element} 两种布局共用的研究详情 */
function ResearchPanelDetails(props) {
  const { message, model, busyKey, workspace, setBusyKey, setStatus } = props
  return (
              <>
                <MessageResearchPlannerProcess
                  plan={model.researchPlan}
                  unitResults={model.unitResults}
                  coverageReview={model.coverageReview}
                  evidenceRecords={message?.evidence_records || []}
                  messageId={message?.id}
                />
                <MessageAnswerBriefSection answerBrief={model.answerBrief} />
                {!model.plannerSummaryAvailable ? (
                  <MessageResearchSynthesisSummary
                    synthesis={model.synthesis}
                    message={message}
                    hasTrailingContent={Boolean(model.scope || model.sections.length || model.actions.length)}
                  />
                ) : null}
                <ResearchStructure scope={model.scope} hasTrailingContent={Boolean(model.sections.length || model.actions.length)} />
                <ResearchObjectSections sections={model.sections} />
                <ResearchNoteActions
                  actions={model.actions}
                  busyKey={busyKey}
                  workspace={workspace}
                  setBusyKey={setBusyKey}
                  setStatus={setStatus}
                  hasLeadingContent={Boolean(model.sections.length)}
                />
              </>
  )
}

/** @param {object} props 研究说明显示状态 @returns {JSX.Element} 回答主卡内嵌布局 */
function EmbeddedResearchPanel(props) {
  const { message, model, workspace, setBusyKey, status, setStatus, summaryOpen, setSummaryOpen, detailsOpen, setDetailsOpen, retryBusy, showRetry, request, sectionStyle } = props
    return (
      <section style={sectionStyle}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            type="button"
            className="sa-button sa-button--ghost"
            style={{ fontSize: 12, minHeight: 28 }}
            onClick={() => setSummaryOpen((value) => !value)}
          >
            {summaryOpen ? '收起说明' : '研究说明'}
          </button>
          {showRetry ? (
            <button
              type="button"
              className="sa-button sa-button--ghost"
              style={{ fontSize: 12, minHeight: 28 }}
              disabled={retryBusy}
              onClick={() => retryFinalAnswer({ message, workspace, setBusyKey, setStatus, request })}
            >
              {retryBusy ? '重试中' : '重试最终回答'}
            </button>
          ) : null}
        </div>
        {summaryOpen ? (
          <div style={{ marginTop: 8 }}>
            <MessageResearchPlannerSummary
              message={message}
              plan={model.researchPlan}
              unitResults={model.unitResults}
              hasTrailingContent={false}
              onRetryFinalAnswer={(targetMessage) => retryFinalAnswer({ message: targetMessage, workspace, setBusyKey, setStatus, request })}
              retryingFinalAnswer={retryBusy}
            />
            <button
              type="button"
              className="sa-button sa-button--ghost"
              style={{ fontSize: 12, minHeight: 28, marginTop: 8 }}
              onClick={() => setDetailsOpen((value) => !value)}
            >
              {detailsOpen ? '收起细节' : '更多细节'}
            </button>
            {detailsOpen ? (
              <ResearchPanelDetails {...props} />
            ) : null}
          </div>
        ) : null}
        <ResearchActionStatus status={status} />
      </section>
    )
}

/** @param {object} props 研究说明显示状态 @returns {JSX.Element} 独立诊断布局 */
function StandaloneResearchPanel(props) {
  const { message, model, workspace, setBusyKey, status, setStatus, detailsOpen, setDetailsOpen, retryBusy, request, sectionStyle } = props

  return (
    <section style={sectionStyle}>
      <MessageResearchPlannerSummary
        message={message}
        plan={model.researchPlan}
        unitResults={model.unitResults}
        hasTrailingContent={false}
        onRetryFinalAnswer={(targetMessage) => retryFinalAnswer({ message: targetMessage, workspace, setBusyKey, setStatus, request })}
        retryingFinalAnswer={retryBusy}
      />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
        <button
          type="button"
          className="sa-button sa-button--ghost"
          style={{ fontSize: 12, minHeight: 28 }}
          onClick={() => setDetailsOpen((value) => !value)}
        >
          {detailsOpen ? '收起细节' : '更多细节'}
        </button>
      </div>
      {detailsOpen ? (
        <ResearchPanelDetails {...props} />
      ) : null}
      <ResearchActionStatus status={status} />
    </section>
  )
}

/**
 * 研究过程/摘要面板。embedded 时由主卡「过程」打开：默认仅步骤区外的轻量入口，不铺芯片墙。
 * @param {{ message: object, embedded?: boolean }} props
 */
export function MessageResearchPanel({ message, embedded = false }) {
  const workspace = useWorkspaceStore()
  const [busyKey, setBusyKey] = useState('')
  const request = useRef(null)
  const [status, setStatus] = useState('')
  // 嵌入主卡：摘要与细节均默认关，避免打开「过程」就铺芯片
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [summaryOpen, setSummaryOpen] = useState(false)
  const model = buildPanelModel(message)

  if (!shouldRenderPanel(model)) {
    return null
  }

  const sectionStyle = embedded
    ? { marginTop: 0, border: 0, borderRadius: 0, padding: 0, background: 'transparent' }
    : PANEL_STYLE

  const retryBusy = busyKey === `retry-final-${message?.run_id || ''}`
  const showRetry = canRetryPlannerFinalAnswer(message)

  const props = { message, model, workspace, busyKey, setBusyKey, status, setStatus, summaryOpen, setSummaryOpen, detailsOpen, setDetailsOpen, retryBusy, showRetry, request, sectionStyle }
  return embedded ? <EmbeddedResearchPanel {...props} /> : <StandaloneResearchPanel {...props} />
}
