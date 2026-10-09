'use client'

import { createContext, useContext, useEffect, useMemo, useReducer, useRef } from 'react'
import { attachProjectSession } from '../lib/api/projects'
import { chatWithAssistant, crossPaperQA, getSessionDetail } from '../lib/api/research'
import { setConversationMessages } from '../lib/api/conversations'
import { getRagRun, cancelRagRun, getLatestConversationRagRun } from '../lib/api/rag-runs'
import { useConversationStore } from './conversation-store'
import { sameMessages } from '../utils/message-utils'
import {
  DEFAULT_DRAFT,
  WELCOME,
  buildChatHistory,
  buildPaperContext,
  createId,
  getError,
  hasConversationContent,
  patchMessage,
  pickMessageExtras,
  restoredMessages,
  toMessage,
} from '../lib/study-assistant-workspace.mjs'
import { EXTERNAL_RESEARCH_MODE, normalizeExternalResearchMode } from '../lib/external-research-mode'
import { runResearchStream, runCrossPaperStream } from '../lib/study-assistant-stream'
import { applyCurrentResult, hasPendingCandidates, candidateRecoveryError, applyDeliveryPatch, assertDeliveryPatch } from '../lib/result-delivery.mjs'

import { usePathname } from 'next/navigation'
import { useAuthStore } from './auth-store'
import { useProjectStore } from './project-store'
import { captureAuthContext, isCurrentAuthContext } from '../lib/auth-client'

const RESULT_POLL_INTERVAL_MS = 2000
const EMPTY_RESEARCH_CONFIRMATION = { open: false, prompt: '' }
const EMPTY_RESEARCH_TRACE = {
  sessionId: '',
  prompt: '',
  status: 'idle',
  events: [],
  projectId: '',
  paperIds: [],
  researchMode: '',
  allowedActions: [],
  lastError: '',
  attemptCount: 0,
  evidenceCount: 0,
  keyFindings: [],
  uncertainties: [],
  ideaCandidates: [],
}

const initialState = {
  mode: 'chat',
  researchMode: EXTERNAL_RESEARCH_MODE,
  draft: DEFAULT_DRAFT,
  messages: [WELCOME],
  researchConfirmation: EMPTY_RESEARCH_CONFIRMATION,
  busy: false,
  error: '',
  restoring: false,
  sessionId: '',
  conversationId: '',
  projectId: '',
  papers: [],
  researchTrace: EMPTY_RESEARCH_TRACE,
  evidenceRecords: [],
  answerBrief: {},
  missingEvidence: [],
  activeRun: null,
}

const RUN_STATUS_LABELS = {
  queued: '内部研究请求已排队，刷新后仍可恢复。',
  running: '内部研究仍在后端运行，稍后刷新可继续查看状态。',
  timed_out: '本次等待已超时，但后端运行状态仍可查询，稍后刷新可能看到最终结果。',
  failed: '内部研究失败，错误已记录。',
}

function reducer(state, action) {
  switch (action.type) {
    case 'currentResult':
      if (state.conversationId !== action.conversationId) return state
      return { ...state, messages: applyCurrentResult(state.messages, action.result), candidateQueryError: action.error, candidateQueryRunId: action.result.run_id }
    case 'deliveryPatch':
      if (state.conversationId !== action.conversationId) return state
      return { ...state, messages: applyDeliveryPatch(state.messages, action) }
    case 'memoryWriteback':
      if (state.conversationId !== action.conversationId) return state
      return { ...state, messages: state.messages.map((message) => (
        message.run_id === action.result.run_id && message.delivery_revision === action.result.delivery_revision
          ? { ...message, memory_writeback: action.result.memory_writeback } : message
      )) }
    case 'setState':
      return { ...state, ...action.patch }
    default:
      return state
  }
}

const WorkspaceStoreContext = createContext(null)

function samePapers(left = [], right = []) {
  if (left === right) return true
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
  return left.every((item, index) => item?.paper_id === right[index]?.paper_id)
}

function normalizeResearchMode(mode, { projectId = '', papers = [] } = {}) {
  void projectId
  void papers
  return normalizeExternalResearchMode(mode)
}

function createTraceEvent(payload) {
  return {
    ...payload,
    at: new Date().toISOString(),
  }
}

function mergePaperIds(...groups) {
  const seen = new Set()
  const merged = []
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      const paperId = String(item || '').trim()
      if (!paperId || seen.has(paperId)) continue
      seen.add(paperId)
      merged.push(paperId)
    }
  }
  return merged
}

function extractRunError(reason) {
  const detail = reason?.body?.detail
  return {
    runId: reason?.run_id || reason?.body?.run_id || detail?.run_id || '',
    runStatus: reason?.run_status || reason?.body?.run_status || detail?.run_status || '',
  }
}

function runStatusMessage(run) {
  if (!run?.status) return ''
  if (run.status === 'completed') return run.answer || ''
  return run.error || RUN_STATUS_LABELS[run.status] || ''
}

function messageHasRun(messages = [], runId = '') {
  if (!runId) return false
  return messages.some((message) => message?.run_id === runId)
}

function messageHasRunAnswer(messages = [], run = null) {
  const answer = String(run?.answer || '').trim()
  if (!answer) return false
  return messages.some((message) => message?.role === 'assistant' && String(message?.content || '').trim() === answer)
}

function withoutEmptyStreamPlaceholder(messages = []) {
  return messages.filter((item, index, items) => (
    index !== items.length - 1
    || item.role !== 'assistant'
    || item.content
    || (!item.pending && item.title !== 'Research stream')
  ))
}

function messageHasRunQuery(messages = [], run = null) {
  const query = String(run?.query || '').trim()
  if (!query) return false
  return messages.some((message) => message?.role === 'user' && String(message?.content || '').trim() === query)
}

function appendRunRecoveryMessages(messages = [], run = null) {
  if (!run?.run_id || messageHasRun(messages, run.run_id) || messageHasRunAnswer(messages, run)) return messages
  const runMessage = runStatusMessage(run)
  if (!runMessage) return messages
  const metadata = run?.metadata && typeof run.metadata === 'object' ? run.metadata : {}
  if (run.status !== 'completed') {
    return [
      ...messages,
      toMessage('assistant', runMessage, '内部研究状态', run.status === 'failed' ? 'error' : 'accent', undefined, {
        run_id: run.run_id,
        run_status: run.status,
      }),
    ]
  }
  const userMessage = run.query && !messageHasRunQuery(messages, run) ? [toMessage('user', run.query)] : []
  return [
    ...messages,
    ...userMessage,
    toMessage('assistant', runMessage, '', undefined, 'markdown', {
      ...metadata,
      run_id: run.run_id,
      run_status: run.status,
    }),
  ]
}

export function WorkspaceStoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const conversations = useConversationStore()
  const projects = useProjectStore()
  const auth = useAuthStore()
  const pathname = usePathname()
  const [queryRequest, requestCandidateQuery] = useReducer((current, request) => ({ ...request, token: current.token + 1 }), { token: 0 })
  const queryToken = queryRequest.token
  const latestStateRef = useRef(state)
  const activeStreamControllerRef = useRef(null)
  const activeRunIdRef = useRef('')
  const activeStreamConversationRef = useRef('')

  const latestCandidate = state.messages.findLast((message) => message.run_id && message.persistence_status === 'saved')
  const requestedCandidate = queryRequest.conversationId === state.conversationId && queryRequest.anchorRunId === latestCandidate?.run_id
    ? state.messages.find((message) => message.run_id === queryRequest.runId) : null
  const candidateTarget = requestedCandidate || latestCandidate
  /** @param {string} runId 用户显式查询的答案身份 @returns {void} 替换共享监听目标，不新建卡片计时器 */
  function reloadCandidates(runId) {
    requestCandidateQuery({ runId, conversationId: state.conversationId, anchorRunId: latestCandidate?.run_id })
  }
  const candidateRunId = candidateTarget?.run_id || ''
  const candidateRevision = candidateTarget?.delivery_revision

  // 1. 【研究交付】【当前结果】每个工作区只有一个监听；同版本候选变化不重新启动计时器
  useEffect(() => {
    if (!candidateRunId || !pathname?.startsWith('/chat') || state.busy) return
    let active = true
    let timer
    const controller = new AbortController()
    const context = captureAuthContext()
    const conversationId = state.conversationId
    let associationRefreshed = false
    async function load() {
      try {
        const run = await getRagRun({ runId: candidateRunId, authContext: context, signal: controller.signal })
        if (!active || !isCurrentAuthContext(context)) return
        const result = run.current_result
        if (result?.run_id !== candidateRunId) throw new Error('当前结果与目标研究不匹配')
        const error = candidateRecoveryError(result)
        dispatch({ type: 'currentResult', conversationId, result, error })
        // 2. 【项目会话】【权威关联】只在服务端结果已确认项目归属后刷新既有清单
        if (!associationRefreshed && result.project_id === state.projectId && result.project_id) {
          associationRefreshed = true
          await Promise.all([projects.refresh(), conversations.refreshActiveConversation()])
        }
        if (!active || !isCurrentAuthContext(context)) return
        if (hasPendingCandidates(result) && !error) timer = setTimeout(load, RESULT_POLL_INTERVAL_MS)
      } catch (reason) {
        if (active && isCurrentAuthContext(context)) dispatch({ type: 'setState', patch: {
          candidateQueryError: getError(reason, '候选状态查询失败'), candidateQueryRunId: candidateRunId,
        } })
      }
    }
    load()
    return () => { active = false; clearTimeout(timer); controller.abort() }
  }, [candidateRunId, candidateRevision, state.conversationId, state.projectId, state.busy, auth.sessionContext, pathname, queryToken])

  useEffect(() => {
    latestStateRef.current = state
  }, [state])

  useEffect(() => {
    const boundConversationId = state.conversationId
    return () => {
      if (activeStreamConversationRef.current !== boundConversationId) return
      activeStreamControllerRef.current?.abort()
      activeStreamControllerRef.current = null
      activeRunIdRef.current = ''
      activeStreamConversationRef.current = ''
    }
  }, [state.conversationId])

  useEffect(() => {
    if (!state.sessionId) return
    let active = true
    const requestSessionId = state.sessionId
    const requestConversationId = state.conversationId
    dispatch({ type: 'setState', patch: { restoring: true } })

    const sessionRequest = getSessionDetail({ sessionId: requestSessionId, conversationId: requestConversationId })
    const runRequest = requestConversationId
      ? getLatestConversationRagRun({ conversationId: requestConversationId }).catch((error) => {
        console.warn('Failed to load latest RAG run:', error?.message || error)
        return null
      })
      : Promise.resolve(null)

    Promise.all([sessionRequest, runRequest])
      .then(([session, run]) => {
        if (!active) return
        const latestState = latestStateRef.current
        if (
          latestState.sessionId !== requestSessionId
          || latestState.conversationId !== requestConversationId
        ) {
          return
        }
        const restoredSnapshot = restoredMessages({ session, pending: null })
        const currentMessages = latestState.messages
        const baseMessages = hasConversationContent(currentMessages)
          ? currentMessages
          : restoredSnapshot
        const restored = appendRunRecoveryMessages(baseMessages, run)
        const shouldAppendRun = restored !== baseMessages
        if (shouldAppendRun && requestConversationId && run.status === 'completed') {
          conversations.setMessages(requestConversationId, restored)
          setConversationMessages({
            conversationId: requestConversationId,
            messages: restored,
            sessionId: run.session_id || requestSessionId,
          }).catch((error) => {
            console.warn('Failed to sync recovered RAG run messages:', error?.message || error)
          })
        }
        dispatch({
          type: 'setState',
          patch: {
            error: '',
            restoring: false,
            activeRun: run,
            messages: restored,
            ...(session?.topic && !latestState.draft ? { draft: session.topic } : {}),
            ...(run ? {
              researchTrace: {
                ...(latestState.researchTrace || EMPTY_RESEARCH_TRACE),
                status: run.status,
                sessionId: run.session_id || latestState.researchTrace?.sessionId || '',
                prompt: run.query || latestState.researchTrace?.prompt || '',
              },
            } : {}),
          },
        })
      })
      .catch((reason) => {
        if (!active) return
        const latestState = latestStateRef.current
        if (
          latestState.sessionId !== requestSessionId
          || latestState.conversationId !== requestConversationId
        ) {
          return
        }
        const message = getError(reason, '会话恢复失败')
        const shouldResetSession = message === '会话不存在'
        if (shouldResetSession && requestConversationId) {
          conversations.setSessionId(requestConversationId, '')
        }
        const latestMessages = latestStateRef.current.messages
        const hasLoadedContent = latestMessages.length > 1 || (latestMessages.length === 1 && latestMessages[0]?.id !== 'welcome')
        dispatch({
          type: 'setState',
          patch: {
            restoring: false,
            error: shouldResetSession ? '' : message,
            ...(shouldResetSession ? { sessionId: '' } : {}),
            ...(!hasLoadedContent && shouldResetSession ? { messages: [WELCOME], draft: DEFAULT_DRAFT } : {}),
          },
        })
      })

    return () => {
      active = false
    }
  }, [state.sessionId, state.conversationId])

  const actions = useMemo(() => {
    const mergeResearchTracePatch = (patch = {}, eventPayload = null) => {
      const trace = latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE
      const nextTrace = patch.researchTrace || {}
      return {
        ...patch,
        researchTrace: {
          ...trace,
          ...nextTrace,
          sessionId: nextTrace.sessionId || trace.sessionId || '',
          prompt: nextTrace.prompt ?? trace.prompt ?? '',
          status: nextTrace.status ?? trace.status ?? 'idle',
          projectId: nextTrace.projectId ?? trace.projectId ?? '',
          paperIds: Array.isArray(nextTrace.paperIds) ? nextTrace.paperIds : (trace.paperIds || []),
          researchMode: nextTrace.researchMode ?? trace.researchMode ?? '',
          allowedActions: Array.isArray(nextTrace.allowedActions) ? nextTrace.allowedActions : (trace.allowedActions || []),
          lastError: nextTrace.lastError ?? trace.lastError ?? '',
          attemptCount: Number.isFinite(nextTrace.attemptCount) ? nextTrace.attemptCount : (trace.attemptCount || 0),
          evidenceCount: Number.isFinite(nextTrace.evidenceCount) ? nextTrace.evidenceCount : (trace.evidenceCount || 0),
          keyFindings: Array.isArray(nextTrace.keyFindings) ? nextTrace.keyFindings : (trace.keyFindings || []),
          uncertainties: Array.isArray(nextTrace.uncertainties) ? nextTrace.uncertainties : (trace.uncertainties || []),
          ideaCandidates: Array.isArray(nextTrace.ideaCandidates) ? nextTrace.ideaCandidates : (trace.ideaCandidates || []),
          events: eventPayload
            ? [...(Array.isArray(nextTrace.events) ? nextTrace.events : (trace.events || [])), createTraceEvent(eventPayload)]
            : (Array.isArray(nextTrace.events) ? nextTrace.events : (trace.events || [])),
        },
      }
    }

    const appendResearchTraceEvent = (payload, patch = {}) => {
      dispatch({
        type: 'setState',
        patch: mergeResearchTracePatch(patch, payload),
      })
    }

    const attachProjectSessionIfNeeded = (projectId, sessionId) => {
      if (!projectId || !sessionId) return
      attachProjectSession({ projectId, sessionId }).catch((error) => {
        dispatch({ type: 'setState', patch: { error: getError(error, '项目会话同步失败') } })
      })
    }

    const syncSessionId = (conversationId, resolvedSessionId) => {
      if (!resolvedSessionId || !conversationId) return
      conversations.setSessionId(conversationId, resolvedSessionId)
      if (
        latestStateRef.current.conversationId === conversationId
        && latestStateRef.current.sessionId !== resolvedSessionId
      ) {
        dispatch({ type: 'setState', patch: { sessionId: resolvedSessionId } })
      }
    }

    const persistMessages = (conversationId, messages, extraPatch = {}) => {
      if (conversationId) {
        conversations.setMessages(conversationId, messages)
        // 1. 新交付由服务端原子保存，浏览器不得用消息列表覆盖提交版本或补造保存成功
        if (!messages.some((message) => message.persistence_status)) setConversationMessages({
          conversationId,
          messages,
          sessionId: latestStateRef.current.sessionId,
        }).catch((err) => {
          console.warn('Failed to sync messages to backend:', err.message)
        })
      }
      if (!conversationId || latestStateRef.current.conversationId === conversationId) {
        dispatch({ type: 'setState', patch: { ...extraPatch, messages } })
      }
    }

    const previewMessages = (conversationId, messages, extraPatch = {}) => {
      if (conversationId) {
        conversations.setMessages(conversationId, messages)
      }
      if (!conversationId || latestStateRef.current.conversationId === conversationId) {
        dispatch({ type: 'setState', patch: { ...extraPatch, messages } })
      }
    }

    const pushFeed = (message) => {
      const currentState = latestStateRef.current
      const nextMessages = [...currentState.messages, message]
      persistMessages(currentState.conversationId, nextMessages)
    }

    const runSubmit = async (prompt, options = {}) => {
      const currentState = latestStateRef.current
      if (!prompt || currentState.busy || currentState.restoring) return
      const mode = options.mode || currentState.mode
      const projectId = options.projectId ?? currentState.projectId
      const papers = Array.isArray(options.papers) ? options.papers : currentState.papers
      const researchMode = normalizeResearchMode(options.researchMode ?? currentState.researchMode, { projectId, papers })
      const onResearchSession = typeof options.onResearchSession === 'function' ? options.onResearchSession : () => {}
      const baseMessages = Array.isArray(options.baseMessages) ? options.baseMessages : currentState.messages
      let requestConversationId = options.conversationId ?? currentState.conversationId
      let requestSessionId = options.sessionId ?? currentState.sessionId
      if (!requestConversationId) {
        const fallbackTitle = mode === 'research'
          ? `深度研究 · ${prompt.slice(0, 20) || '未命名课题'}`
          : '新对话'
        const createdConversation = await conversations.createNew(fallbackTitle, { returnConversation: true })
        requestConversationId = typeof createdConversation === 'string' ? createdConversation : createdConversation?.id || ''
        if (typeof createdConversation !== 'string' && createdConversation?.sessionId) {
          requestSessionId = createdConversation.sessionId
        }
      }
      if (!requestSessionId && requestConversationId) {
        const existingConversation = (conversations.conversations || []).find((item) => item.id === requestConversationId)
        requestSessionId = existingConversation?.sessionId || ''
      }
      attachProjectSessionIfNeeded(projectId, requestSessionId)
      const linkedPaperIds = (conversations.conversations || [])
        .find((item) => item.id === requestConversationId)
        ?.associatedPapers || []
      const paperIds = mergePaperIds(
        papers.map((paper) => paper?.paper_id),
        linkedPaperIds,
      )
      if (mode === 'research' && paperIds.length) {
        for (const paperId of paperIds) {
          await conversations.addPaper(requestConversationId, paperId)
        }
      }
      const userMessage = toMessage('user', prompt)
      const nextPendingMessages = [...baseMessages, userMessage]
      dispatch({
        type: 'setState',
        patch: {
          busy: true,
          error: '',
          draft: '',
          mode,
          sessionId: requestSessionId,
          conversationId: requestConversationId,
          projectId,
          papers,
          messages: nextPendingMessages,
          evidenceRecords: [],
          answerBrief: {},
          missingEvidence: [],
        },
      })
      if (requestConversationId) {
        conversations.setMessages(requestConversationId, nextPendingMessages)
      }

      const streamController = new AbortController()
      const idempotencyKey = createId('rag-run')
      activeStreamControllerRef.current = streamController
      activeRunIdRef.current = ''
      activeStreamConversationRef.current = requestConversationId
      const registerStreamRun = (runId) => {
        activeRunIdRef.current = runId
        dispatch({
          type: 'setState',
          patch: {
            activeRun: {
              run_id: runId,
              status: 'running',
              session_id: requestSessionId,
              query: prompt,
            },
          },
        })
      }
      let streamedMessages = nextPendingMessages
      try {
        if (mode === 'chat') {
          if (paperIds.length > 0 || projectId) {
            const scopeLabel = paperIds.length > 0 ? '对话论文优先' : (projectId ? '当前项目论文池' : '全局知识库')
            // 1. 先放一个 pending 占位 assistant 消息，MessageResearchSteps 读 live events 实时点亮
            const pendingAssistant = toMessage(
              'assistant',
              '',
              '',
              undefined,
              undefined,
              { pending: true },
            )
            const streamingMessages = [...nextPendingMessages, pendingAssistant]
            previewMessages(
              requestConversationId,
              streamingMessages,
              mergeResearchTracePatch({
                researchTrace: {
                  sessionId: requestSessionId,
                  prompt,
                  status: 'streaming',
                  projectId,
                  paperIds,
                  researchMode: 'internal_research',
                  evidenceCount: 0,
                  keyFindings: [],
                  uncertainties: [],
                  ideaCandidates: [],
                  events: [],
                },
              }, {
                type: 'event',
                stage: 'route',
                intent: 'internal_research',
                detail: `已按${scopeLabel}进入内部研究`,
              }),
            )
            streamedMessages = streamingMessages
            // 2. 跨论文流式：阶段事件进 researchTrace；done 时用定稿载荷替换 pending 占位
            const data = await runCrossPaperStream({
              query: prompt,
              paperIds,
              sessionId: requestSessionId,
              conversationId: requestConversationId,
              projectId,
              chatHistory: buildChatHistory(baseMessages, prompt),
              idempotencyKey,
              signal: streamController.signal,
              onRun: registerStreamRun,
              onEvent(payload) {
                if (payload?.session_id) {
                  syncSessionId(requestConversationId, payload.session_id)
                }
                appendResearchTraceEvent(payload, {
                  researchTrace: {
                    ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                    status: 'streaming',
                    sessionId: payload?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                    researchMode: 'internal_research',
                  },
                })
              },
            })
            syncSessionId(requestConversationId, data?.session_id)
            attachProjectSessionIfNeeded(projectId, data?.session_id || requestSessionId)
            // 3. 落地前快照 researchTrace.events，供历史消息离线渲染分步卡
            const researchStepsSnapshot = Array.isArray(latestStateRef.current.researchTrace?.events)
              ? [...latestStateRef.current.researchTrace.events]
              : []
            const plannerFinalOk = data?.source_status?.planner_final === 'ok'
            previewMessages(
              requestConversationId,
              [
                ...nextPendingMessages,
                toMessage('assistant', data?.answer || '后端没有返回回答内容。', '', undefined, undefined, {
                  citations: data?.citations || [],
                  paper_sources: data?.paper_sources || [],
                  chunk_citations: data?.chunk_citations || [],
                  graph_evidence: data?.graph_evidence || [],
                  graph_hint_section: data?.graph_hint_section || '',
                  graph_hint_full: data?.graph_hint_full || '',
                  evidence_summary: data?.evidence_summary || '',
                  answer_confidence: data?.confidence || '',
                  research_scope: data?.research_scope || {},
                  research_objects: data?.research_objects || [],
                  note_actions: data?.note_actions || [],
                  synthesis_state: data?.synthesis_state || {},
                  dialogue_mode: data?.dialogue_mode || '',
                  evidence_records: data?.evidence_records || [],
                  answer_brief: data?.answer_brief || {},
                  missing_evidence: data?.missing_evidence || [],
                  research_plan: data?.research_plan || null,
                  research_unit_results: data?.research_unit_results || [],
                  coverage_review: data?.coverage_review || null,
                  planner_trace: data?.planner_trace || null,
                  planner_final_blocks: data?.planner_final_blocks || [],
                  planner_final_validation_errors: data?.planner_final_validation_errors || [],
                  final_delivery_report: data?.final_delivery_report ?? null,
                  final_citation_bindings: data?.final_citation_bindings ?? null,
                  final_generation_attempt: data?.final_generation_attempt ?? null,
                  final_delivery_applicability: data?.final_delivery_applicability ?? null,
                  delivery_revision: data?.delivery_revision ?? null,
                  target_run_id: data?.target_run_id ?? null,
                  producer_run_id: data?.producer_run_id ?? null,
                  persistence_status: data?.persistence_status ?? null,
                  persistence_error: data?.persistence_error ?? null,
                  can_retry_save: data?.can_retry_save ?? null,
                  turn_outcome: data?.turn_outcome ?? null,
                  trace_summary: data?.trace_summary || null,
                  request_id: data?.request_id || null,
                  source_status: data?.source_status || {},
                  source_status_reasons: data?.source_status_reasons || {},
                  run_id: data?.run_id || null,
                  run_status: data?.run_status || null,
                  research_steps: researchStepsSnapshot,
                  // 回合级字段按后端原值落地：缺失保持 null，不补默认值 ——
                  // 编一个空深度或空写回终态会让回答卡把「本轮没跑项目科研」显示成「跑过且什么都没做」
                  turn_depth: data?.turn_depth || null,
                  depth_transitions: data?.depth_transitions || null,
                  memory_read_result: data?.memory_read_result || null,
                  memory_writeback: data?.memory_writeback || null,
                  // 仅成功定稿启用前端打字机；失败/跳过态整块显示，不做无意义动画
                  typewriter: plannerFinalOk,
                }),
              ],
              {
                evidenceRecords: data?.evidence_records || [],
                answerBrief: data?.answer_brief || {},
                missingEvidence: data?.missing_evidence || [],
                activeRun: data?.run_id ? {
                  run_id: data.run_id,
                  status: data?.run_status || 'completed',
                  session_id: data?.session_id || requestSessionId,
                  query: prompt,
                  answer: data?.answer || '',
                } : null,
                researchTrace: {
                  ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                  status: data?.status || 'completed',
                  sessionId: data?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                  projectId,
                  paperIds,
                  researchMode: 'internal_research',
                  allowedActions: Array.isArray(data?.allowed_actions) ? data.allowed_actions : [],
                  lastError: data?.last_error || '',
                  attemptCount: Number(data?.attempt_count || 0),
                  evidenceCount: Array.isArray(data?.citations) ? data.citations.length : 0,
                  keyFindings: Array.isArray(data?.research_objects)
                    ? data.research_objects.map((item) => item?.label).filter(Boolean).slice(0, 4)
                    : [],
                  events: researchStepsSnapshot,
                },
              },
            )
            appendResearchTraceEvent({
              type: 'event',
              stage: 'done',
              detail: Array.isArray(data?.citations) && data.citations.length
                ? `已命中 ${data.citations.length} 条引用证据`
                : '当前没有命中明确引用，可继续缩小范围或补材料',
              session_id: data?.session_id || requestSessionId,
              evidence_count: Array.isArray(data?.citations) ? data.citations.length : 0,
            })
          } else {
            const contextSummary = papers.length > 0
              ? papers.map((paper) => buildPaperContext(paper)).filter(Boolean).join('\n\n---\n\n')
              : ''
            dispatch({
              type: 'setState',
              patch: { researchTrace: EMPTY_RESEARCH_TRACE },
            })
            const data = await chatWithAssistant({
              query: prompt,
              sessionId: requestSessionId,
              conversationId: requestConversationId,
              projectId,
              chatHistory: buildChatHistory(baseMessages, prompt),
              contextSummary,
              signal: streamController.signal,
            })
            syncSessionId(requestConversationId, data?.session_id)
            attachProjectSessionIfNeeded(projectId, data?.session_id || requestSessionId)
            previewMessages(
              requestConversationId,
              [...nextPendingMessages, toMessage('assistant', data?.answer || '后端没有返回回答内容。', '对话回答', undefined, undefined, pickMessageExtras(data))],
            )
          }
        } else {
          const tracePatch = {
            researchTrace: {
              sessionId: requestSessionId,
              prompt,
              status: 'queued',
              events: [createTraceEvent({ type: 'event', stage: 'queued', prompt, research_mode: researchMode })],
              projectId,
              paperIds,
              researchMode,
              evidenceCount: 0,
              keyFindings: [],
              uncertainties: [],
              ideaCandidates: [],
            },
          }
          const assistantMessage = toMessage('assistant', '', 'Research stream', undefined, 'markdown')
          let streamedContent = ''
          streamedMessages = [...nextPendingMessages, assistantMessage]
          previewMessages(requestConversationId, streamedMessages, tracePatch)

          let resolvedResearchSession = ''
          const resolveResearchSession = (resolvedSessionId) => {
            if (!resolvedSessionId || resolvedResearchSession === resolvedSessionId) return
            resolvedResearchSession = resolvedSessionId
            syncSessionId(requestConversationId, resolvedSessionId)
            attachProjectSessionIfNeeded(projectId, resolvedSessionId)
            dispatch({
              type: 'setState',
              patch: mergeResearchTracePatch({
                sessionId: resolvedSessionId,
                researchTrace: {
                  sessionId: resolvedSessionId,
                },
              }),
            })
            onResearchSession(resolvedSessionId)
          }

          const data = await runResearchStream({
            query: prompt,
            sessionId: requestSessionId,
            conversationId: requestConversationId,
            projectId,
            paperIds,
            researchMode,
            intent: 'deep_research',
            idempotencyKey,
            signal: streamController.signal,
            onRun: registerStreamRun,
            onToken(token) {
              if (!token) return
              streamedContent += token
              streamedMessages = patchMessage(streamedMessages, assistantMessage.id, (current) => ({
                ...current,
                content: streamedContent,
              }))
              const trace = latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE
              if (trace.status !== 'streaming') {
                dispatch({
                  type: 'setState',
                  patch: {
                    researchTrace: {
                      ...trace,
                      status: 'streaming',
                      researchMode: trace.researchMode || researchMode,
                    },
                  },
                })
              }
              previewMessages(requestConversationId, streamedMessages)
            },
            onEvent(payload) {
              if (payload?.session_id) {
                resolveResearchSession(payload.session_id)
              }
              if (payload?.stage === 'evidence_snapshot') {
                dispatch({
                  type: 'setState',
                  patch: {
                    evidenceRecords: payload.evidence_records || [],
                    missingEvidence: payload.missing_evidence || [],
                  },
                })
              }
              if (payload?.stage === 'answer_brief_snapshot') {
                dispatch({
                  type: 'setState',
                  patch: { answerBrief: payload.answer_brief || {} },
                })
              }
              appendResearchTraceEvent(payload, {
                researchTrace: {
                  ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                  status: payload?.stage === 'route' ? 'routing' : payload?.stage === 'done' ? 'completed' : 'streaming',
                  sessionId: payload?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                  researchMode: payload?.research_mode || latestStateRef.current.researchTrace?.researchMode || researchMode,
                  evidenceCount: Number.isFinite(payload?.evidence_count)
                    ? payload.evidence_count
                    : (latestStateRef.current.researchTrace?.evidenceCount || 0),
                  keyFindings: Array.isArray(payload?.key_findings)
                    ? payload.key_findings
                    : (latestStateRef.current.researchTrace?.keyFindings || []),
                  uncertainties: Array.isArray(payload?.uncertainties)
                    ? payload.uncertainties
                    : (latestStateRef.current.researchTrace?.uncertainties || []),
                  ideaCandidates: Array.isArray(payload?.idea_candidates)
                    ? payload.idea_candidates
                    : (latestStateRef.current.researchTrace?.ideaCandidates || []),
                },
              })
            },
            onDone(payload) {
              if (payload?.session_id) {
                resolveResearchSession(payload.session_id)
              }
              appendResearchTraceEvent(payload, {
                researchTrace: {
                  ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                  status: payload?.status || (payload?.awaiting_clarification ? 'awaiting_clarification' : 'completed'),
                  sessionId: payload?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                  researchMode: payload?.research_mode || latestStateRef.current.researchTrace?.researchMode || researchMode,
                  allowedActions: Array.isArray(payload?.allowed_actions) ? payload.allowed_actions : (latestStateRef.current.researchTrace?.allowedActions || []),
                  lastError: payload?.last_error || latestStateRef.current.researchTrace?.lastError || '',
                  attemptCount: Number(payload?.attempt_count || latestStateRef.current.researchTrace?.attemptCount || 0),
                  evidenceCount: Number.isFinite(payload?.evidence_count)
                    ? payload.evidence_count
                    : (latestStateRef.current.researchTrace?.evidenceCount || 0),
                  keyFindings: Array.isArray(payload?.key_findings)
                    ? payload.key_findings
                    : (latestStateRef.current.researchTrace?.keyFindings || []),
                  uncertainties: Array.isArray(payload?.uncertainties)
                    ? payload.uncertainties
                    : (latestStateRef.current.researchTrace?.uncertainties || []),
                  ideaCandidates: Array.isArray(payload?.idea_candidates)
                    ? payload.idea_candidates
                    : (latestStateRef.current.researchTrace?.ideaCandidates || []),
                },
              })
            },
          })
          resolveResearchSession(data?.session_id)
          const content = data?.answer ?? data?.report ?? (streamedContent || data?.clarification_question || '后端没有返回研究内容。')
          const title = data?.awaiting_clarification ? 'Research clarification' : 'Research result'
          streamedMessages = patchMessage(streamedMessages, assistantMessage.id, (current) => ({
            ...current,
            ...pickMessageExtras(data),
            content,
            title,
            tone: data?.awaiting_clarification ? 'accent' : undefined,
            renderMode: 'markdown',
          }))
          previewMessages(
            requestConversationId,
            streamedMessages,
            {
              researchTrace: {
                ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                status: data?.status || (data?.awaiting_clarification ? 'awaiting_clarification' : 'completed'),
                sessionId: data?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                researchMode: data?.research_mode || latestStateRef.current.researchTrace?.researchMode || researchMode,
                allowedActions: Array.isArray(data?.allowed_actions) ? data.allowed_actions : (latestStateRef.current.researchTrace?.allowedActions || []),
                lastError: data?.last_error || latestStateRef.current.researchTrace?.lastError || '',
                attemptCount: Number(data?.attempt_count || latestStateRef.current.researchTrace?.attemptCount || 0),
                evidenceCount: Number.isFinite(data?.evidence_count)
                  ? data.evidence_count
                  : (latestStateRef.current.researchTrace?.evidenceCount || 0),
                keyFindings: Array.isArray(data?.key_findings)
                  ? data.key_findings
                  : (latestStateRef.current.researchTrace?.keyFindings || []),
                uncertainties: Array.isArray(data?.uncertainties)
                  ? data.uncertainties
                  : (latestStateRef.current.researchTrace?.uncertainties || []),
                ideaCandidates: Array.isArray(data?.idea_candidates)
                  ? data.idea_candidates
                  : (latestStateRef.current.researchTrace?.ideaCandidates || []),
              },
            },
          )
        }
      } catch (reason) {
        if (streamController.signal.aborted || reason?.cancelled) {
          const cancelledMessages = withoutEmptyStreamPlaceholder(streamedMessages)
          previewMessages(requestConversationId, cancelledMessages)
          dispatch({
            type: 'setState',
            patch: {
              activeRun: activeRunIdRef.current ? {
                run_id: activeRunIdRef.current,
                status: 'cancelled',
                session_id: requestSessionId,
                query: prompt,
              } : null,
              researchTrace: mode === 'research' || activeRunIdRef.current
                ? {
                  ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                  status: 'cancelled',
                  lastError: '',
                }
                : latestStateRef.current.researchTrace,
              messages: !requestConversationId || latestStateRef.current.conversationId === requestConversationId
                ? cancelledMessages
                : latestStateRef.current.messages,
            },
          })
          return
        }
        const message = getError(reason)
        const runError = extractRunError(reason)
        const latestRun = runError.runId || !requestConversationId
          ? null
          : await getLatestConversationRagRun({ conversationId: requestConversationId }).catch((error) => {
            console.warn('Failed to recover latest RAG run after request error:', error?.message || error)
            return null
          })
        const recoveredRun = latestRun || (runError.runId ? {
          run_id: runError.runId,
          status: runError.runStatus || 'failed',
          error: message,
          query: prompt,
          session_id: requestSessionId,
        } : null)
        const baseMessages = withoutEmptyStreamPlaceholder(streamedMessages)
        const canRecoverCompleted = recoveredRun?.status === 'completed' && recoveredRun?.answer
        const failureTitle = recoveredRun?.status === 'timed_out' ? '内部研究等待超时' : '内部研究失败'
        const failureTone = recoveredRun?.status === 'timed_out' ? 'accent' : 'error'
        const visibleFailureMessages = canRecoverCompleted
          ? baseMessages
          : [
            ...baseMessages,
            toMessage('assistant', message, failureTitle, failureTone, undefined, {
              ...(recoveredRun?.run_id ? { run_id: recoveredRun.run_id, run_status: recoveredRun.status } : {}),
            }),
          ]
        const failureMessages = appendRunRecoveryMessages(visibleFailureMessages, recoveredRun)
        previewMessages(requestConversationId, failureMessages)
        dispatch({
          type: 'setState',
          patch: {
            error: canRecoverCompleted ? '' : message,
            activeRun: recoveredRun,
            researchTrace: mode === 'research' || recoveredRun
              ? {
                ...(latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE),
                status: recoveredRun?.status || 'failed',
                sessionId: recoveredRun?.session_id || latestStateRef.current.researchTrace?.sessionId || '',
                prompt,
                lastError: canRecoverCompleted ? '' : message,
                events: [
                  ...((latestStateRef.current.researchTrace || EMPTY_RESEARCH_TRACE).events || []),
                  createTraceEvent({
                    type: 'error',
                    detail: message,
                    run_id: recoveredRun?.run_id || '',
                    run_status: recoveredRun?.status || '',
                  }),
                ],
              }
              : latestStateRef.current.researchTrace,
            messages: !requestConversationId || latestStateRef.current.conversationId === requestConversationId
              ? failureMessages
              : latestStateRef.current.messages,
          },
        })
      } finally {
        if (activeStreamControllerRef.current === streamController) {
          activeStreamControllerRef.current = null
          activeRunIdRef.current = ''
          activeStreamConversationRef.current = ''
        }
        dispatch({ type: 'setState', patch: { busy: false } })
      }
    }

    return {
      setMode(mode) {
        dispatch({ type: 'setState', patch: { mode } })
      },
      setResearchMode(researchMode) {
        dispatch({ type: 'setState', patch: { researchMode } })
      },
      setDraft(draft) {
        dispatch({ type: 'setState', patch: { draft } })
      },
      setMessages(messages) {
        dispatch({ type: 'setState', patch: { messages } })
      },
      /** @param {object} result 明确会话、目标消息与提交版本 @returns {void} 原子应用当前结果 */
      applyResultDelivery(result) {
        assertDeliveryPatch({ runId: result.runId, patch: result.patch })
        dispatch({ ...result, type: 'deliveryPatch' })
      },
      /** @param {object} update 会话及同版本候选结果 @returns {void} 不覆盖更新答案的候选状态 */
      applyMemoryWriteback(update) {
        dispatch({ ...update, type: 'memoryWriteback' })
      },
      pushFeed,
      bindConversation({ conversationId = '', sessionId = '', projectId = '', papers = [], messages }) {
        const hasMessages = Array.isArray(messages)
        const nextMessages = hasMessages ? messages : state.messages
        const shouldRestore = Boolean(sessionId) && (
          state.conversationId !== conversationId
          || state.sessionId !== sessionId
        )
        if (
          state.conversationId === conversationId
          && state.sessionId === sessionId
          && state.projectId === projectId
          && samePapers(state.papers, papers)
          && (!hasMessages || sameMessages(state.messages, nextMessages))
        ) {
          return
        }
        const patch = {
          conversationId,
          sessionId,
          projectId,
          papers,
          restoring: shouldRestore,
          ...(state.conversationId !== conversationId || !conversationId
            ? { researchTrace: EMPTY_RESEARCH_TRACE }
            : {}),
        }
        if (hasMessages) {
          patch.messages = nextMessages
        }
        dispatch({ type: 'setState', patch })
      },
      setResolvedSessionId(sessionId) {
        syncSessionId(latestStateRef.current.conversationId, sessionId)
      },
      async submit() {
        const currentState = latestStateRef.current
        const prompt = currentState.draft.trim()
        if (!prompt || currentState.busy || currentState.restoring) return
        if (currentState.mode === 'research') {
          dispatch({
            type: 'setState',
            patch: {
              researchConfirmation: { open: true, prompt },
            },
          })
          return
        }
        await runSubmit(prompt)
      },
      async stop() {
        const controller = activeStreamControllerRef.current
        if (!controller || controller.signal.aborted) return
        const runId = activeRunIdRef.current
        try {
          if (runId) await cancelRagRun({ runId })
        } catch (reason) {
          const message = getError(reason, '停止研究失败')
          console.warn('Failed to cancel durable RAG run:', message)
          dispatch({ type: 'setState', patch: { error: message } })
        } finally {
          controller.abort(new DOMException('用户已停止研究', 'AbortError'))
        }
      },
      async confirmResearch(options = {}) {
        const prompt = latestStateRef.current.researchConfirmation.prompt.trim()
        dispatch({
          type: 'setState',
          patch: { researchConfirmation: EMPTY_RESEARCH_CONFIRMATION },
        })
        await runSubmit(prompt, { ...options, mode: 'research' })
      },
      cancelResearchConfirmation() {
        dispatch({
          type: 'setState',
          patch: { researchConfirmation: EMPTY_RESEARCH_CONFIRMATION },
        })
      },
      async launchResearch({ prompt, projectId = '', papers = [], researchMode, onResearchSession = () => {} }) {
        const cleanPrompt = String(prompt || '').trim()
        if (!cleanPrompt) return
        const normalizedResearchMode = normalizeResearchMode(researchMode ?? latestStateRef.current.researchMode, {
          projectId,
          papers,
        })
        dispatch({
          type: 'setState',
          patch: {
            mode: 'research',
            researchMode: normalizedResearchMode,
            projectId,
            papers,
            draft: '',
            messages: [WELCOME],
            researchConfirmation: EMPTY_RESEARCH_CONFIRMATION,
            researchTrace: EMPTY_RESEARCH_TRACE,
          },
        })
        await runSubmit(cleanPrompt, {
          conversationId: '',
          mode: 'research',
          projectId,
          papers,
          researchMode: normalizedResearchMode,
          onResearchSession,
          baseMessages: [WELCOME],
        })
      },
      seedDraft(draft) {
        dispatch({ type: 'setState', patch: { draft } })
      },
    }
  }, [conversations, projects, state])

  const value = useMemo(() => ({
    ...state,
    ...actions,
    reloadCandidates,
  }), [actions, state, queryRequest])

  return (
    <WorkspaceStoreContext.Provider value={value}>
      {children}
    </WorkspaceStoreContext.Provider>
  )
}

export function useWorkspaceStore() {
  const context = useContext(WorkspaceStoreContext)
  if (!context) {
    throw new Error('useWorkspaceStore must be used within WorkspaceStoreProvider')
  }
  return context
}
