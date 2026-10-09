import { requestJson } from './core'
import { createId } from '../study-assistant-workspace.mjs'

const RESEARCH_REQUEST_TIMEOUT_MS = 600_000

export const getSessionDetail = ({ sessionId, conversationId = '' }) =>
  requestJson({
    path: `/api/v1/sessions/${sessionId}${conversationId ? `?conversation_id=${encodeURIComponent(conversationId)}` : ''}`,
  })

export const listResearchSessions = ({ limit = 20, offset = 0 } = {}) =>
  requestJson({ path: `/api/v1/research/sessions?limit=${limit}&offset=${offset}` })

/** @param {object} request 研究会话、澄清与同次请求标识 @returns {Promise<object>} 操作受理或结果 */
export const resumeResearchSession = ({ sessionId, answer, idempotencyKey = createId('resume') }) =>
  requestJson({
    path: `/api/v1/research/sessions/${encodeURIComponent(sessionId)}/resume`,
    timeoutMs: RESEARCH_REQUEST_TIMEOUT_MS,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({ answer }),
    },
  })

/** @param {object} request 会话和同次重试标识 @returns {Promise<object>} 操作受理或结果 */
export const retryResearchSession = ({ sessionId, idempotencyKey = createId('retry') }) =>
  requestJson({
    path: `/api/v1/research/sessions/${encodeURIComponent(sessionId)}/retry`,
    timeoutMs: RESEARCH_REQUEST_TIMEOUT_MS,
    init: { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey } },
  })

export const cancelResearchSession = ({ sessionId }) =>
  requestJson({
    path: `/api/v1/research/sessions/${encodeURIComponent(sessionId)}/cancel`,
    init: { method: 'POST' },
  })

export const createSession = ({ topic }) =>
  requestJson({
    path: '/api/v1/sessions',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic }),
    },
  })

export const chatWithAssistant = ({ query, sessionId, conversationId, chatHistory, contextSummary, projectId, signal, idempotencyKey = createId('chat') }) =>
  requestJson({
    path: '/api/v1/chat',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey },
      signal,
      body: JSON.stringify({
        query,
        session_id: sessionId,
        conversation_id: conversationId,
        project_id: projectId,
        chat_history: chatHistory,
        context_summary: contextSummary,
      }),
    },
  })

/** @param {object} request 用户问题及研究范围 @returns {Promise<object>} 同次研究响应 */
export const researchWithAssistant = ({ query, sessionId, conversationId, projectId, paperIds = [], idempotencyKey = createId('research') }) =>
  requestJson({
    path: '/api/v1/research',
    timeoutMs: RESEARCH_REQUEST_TIMEOUT_MS,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        conversation_id: conversationId,
        project_id: projectId,
        paper_ids: paperIds,
      }),
    },
  })

export const searchArxiv = ({ query, sessionId, conversationId, projectId }) =>
  requestJson({
    path: '/api/v1/search/arxiv',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId, conversation_id: conversationId, project_id: projectId }),
    },
  })

/** @param {object} request 问题、论文及同次请求标识 @returns {Promise<object>} 正文和实际保存结果 */
export const crossPaperQA = ({ query, paperIds, sessionId, conversationId, projectId, chatHistory = [], idempotencyKey = createId('qa') }) =>
  requestJson({
    path: '/api/v1/qa/cross-paper',
    timeoutMs: RESEARCH_REQUEST_TIMEOUT_MS,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        query,
        paper_ids: paperIds,
        chat_history: chatHistory,
        session_id: sessionId,
        conversation_id: conversationId,
        project_id: projectId,
      }),
    },
  })

export const getGraphEvidencePaths = ({ paperIds, query, pathType = 'support', maxHops = 3 }) =>
  requestJson({
    path: '/api/v1/graph/evidence-paths',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paper_ids: paperIds,
        query,
        path_type: pathType,
        max_hops: maxHops,
      }),
    },
  })

export const formatCitation = ({ paperId = '', doi = '', arxivId = '', style = 'apa' }) =>
  requestJson({
    path: '/api/v1/citation/format',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paper_id: paperId,
        doi,
        arxiv_id: arxivId,
        style,
      }),
    },
  })

export const batchFormatCitations = ({ paperIds, style = 'apa' }) =>
  requestJson({
    path: '/api/v1/citation/batch',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paper_ids: paperIds,
        style,
      }),
    },
  })

export const getGraphStats = ({ projectId } = {}) =>
  requestJson({ path: `/api/v1/graph/stats${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''}` })

export const getPaperSubgraph = ({ paperId }) =>
  requestJson({ path: `/api/v1/graph/papers/${encodeURIComponent(paperId)}/subgraph` })

export const getPaperGraph = ({
  projectId = '',
  libraryId = '',
  limit = 200,
  signal,
} = {}) => {
  const params = new URLSearchParams()
  if (projectId) params.set('project_id', projectId)
  if (libraryId) params.set('library_id', libraryId)
  if (limit) params.set('limit', String(limit))
  const query = params.toString()
  return requestJson({
    path: `/api/v1/graph/papers/network${query ? `?${query}` : ''}`,
    init: signal ? { signal } : {},
  })
}

export const getEvolutionGraph = ({
  viewMode = 'paper',
  yearStart,
  yearEnd,
} = {}) => {
  const params = new URLSearchParams()
  if (viewMode) params.set('view_mode', viewMode)
  if (Number.isFinite(yearStart)) params.set('from_year', String(yearStart))
  if (Number.isFinite(yearEnd)) params.set('to_year', String(yearEnd))
  const query = params.toString()
  return requestJson({ path: `/api/v1/graph/evolution${query ? `?${query}` : ''}` })
}

export const getControversyMap = ({ paperIds = [], topic = '', includeGaps = true, includeConflicts = true, projectId = '' }) =>
  requestJson({
    path: '/api/v1/graph/controversy-map',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paper_ids: paperIds,
        topic,
        include_gaps: includeGaps,
        include_conflicts: includeConflicts,
        project_id: projectId,
      }),
    },
  })

export const queryGraph = ({ query, mode = 'hybrid', topK = 5, projectId }) =>
  requestJson({
    path: '/api/v1/graph/query',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        mode,
        top_k: topK,
        project_id: projectId,
      }),
    },
  })

export const generateLiteratureReview = ({ query, sessionId }) =>
  requestJson({
    path: '/api/v1/chat',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        intent: 'literature_review',
      }),
    },
  })
