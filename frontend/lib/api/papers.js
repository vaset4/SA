import { requestJson, requestBlob, uploadWithProgress } from './core'
import { normalizeNoteCollection } from '../notes'

/** @param {object} options 论文身份 @returns {Promise<Blob>} 经认证核验的原文件 */
export const downloadPaperSource = ({ paperId }) => requestBlob({ path: `/api/v1/papers/${encodeURIComponent(paperId)}/source` })

function hasAnalysisContent(analysis) {
  if (!analysis || typeof analysis !== 'object') return false
  return Boolean(
    analysis.problem
    || analysis.method
    || analysis.tldr
    || analysis.conclusion
    || analysis.conclusion_zh
    || (Array.isArray(analysis.contribution) && analysis.contribution.length)
    || (Array.isArray(analysis.section_summaries) && analysis.section_summaries.length)
  )
}

export function getPaperAnalysisStatus(paper) {
  const explicitStatus = typeof paper?.analysis_status === 'string' ? paper.analysis_status.trim() : ''
  if (explicitStatus) return explicitStatus
  return hasAnalysisContent(paper?.analysis) ? 'analysis_ready' : 'not_started'
}

export function normalizePaper(paper) {
  if (!paper) return null
  const publishedDate = typeof paper.published_date === 'string' ? paper.published_date.trim() : ''
  const categories = Array.isArray(paper.categories)
    ? paper.categories.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim())
    : []
  const authors = Array.isArray(paper.authors)
    ? paper.authors.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim())
    : []
  const keywords = Array.isArray(paper.keywords)
    ? paper.keywords.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim())
    : []
  const inferredYear = /^\d{4}/.test(publishedDate) ? publishedDate.slice(0, 4) : ''
  return {
    ...paper,
    authors,
    categories,
    keywords,
    year: paper.year || inferredYear,
    category: paper.category || categories.join(', '),
    analysis_status: getPaperAnalysisStatus(paper),
  }
}

export function normalizePaperCollection(papers) {
  return Array.isArray(papers) ? papers.map((paper) => normalizePaper(paper)).filter(Boolean) : []
}

export function normalizeDeleteTask(result) {
  if (!result || typeof result !== 'object') return null
  return {
    ...result,
    task_id: typeof result.task_id === 'string' ? result.task_id.trim() : '',
    task_type: typeof result.task_type === 'string' ? result.task_type.trim() : 'paper_delete',
    status: typeof result.status === 'string' ? result.status.trim() : 'queued',
    message: typeof result.message === 'string' ? result.message.trim() : '',
    error: typeof result.error === 'string' ? result.error.trim() : '',
    payload: result.payload && typeof result.payload === 'object' ? result.payload : {},
    result: result.result && typeof result.result === 'object' ? result.result : null,
  }
}

export const listPapers = ({ limit = 50, offset = 0, libraryId } = {}) =>
  requestJson({ path: `/api/v1/papers?limit=${limit}&offset=${offset}${libraryId ? `&library_id=${encodeURIComponent(libraryId)}` : ''}` })
    .then((data) => ({
      ...data,
      papers: normalizePaperCollection(data?.papers),
    }))

export const getPaperDetail = ({ paperId }) =>
  requestJson({ path: `/api/v1/papers/${encodeURIComponent(paperId)}` })
    .then((data) => normalizePaper(data))

export const deletePaper = ({ paperId }) =>
  requestJson({
    path: `/api/v1/papers/${encodeURIComponent(paperId)}`,
    init: { method: 'DELETE' },
  }).then((data) => normalizeDeleteTask(data))

export const updatePaperMetadata = ({ paperId, payload }) =>
  requestJson({
    path: `/api/v1/papers/${encodeURIComponent(paperId)}`,
    init: {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  }).then((data) => normalizePaper(data))

export const triggerPaperAnalysisTask = ({ paperId }) =>
  requestJson({
    path: `/api/v1/papers/${encodeURIComponent(paperId)}/analysis`,
    init: {
      method: 'POST',
    },
  })

export const getPaperChunkMap = ({ paperId }) =>
  requestJson({ path: `/api/v1/knowledge/papers/${encodeURIComponent(paperId)}/chunk-map` })

export const getPaperChunkAudit = ({ paperId }) =>
  requestJson({ path: `/api/v1/knowledge/papers/${encodeURIComponent(paperId)}/chunk-audit` })

export const getChunkDetail = ({ chunkId, includeText = 'preview', maxTextChars = 4000 }) => {
  const params = new URLSearchParams()
  params.set('include_text', includeText)
  params.set('max_text_chars', String(maxTextChars))
  return requestJson({
    path: `/api/v1/knowledge/chunks/${encodeURIComponent(chunkId)}?${params.toString()}`,
  })
}

export const uploadPaperWithProgress = ({ file, onProgress, projectId = '', libraryId = '' }) =>
  uploadWithProgress({
    path: '/api/v1/papers/upload',
    file,
    onProgress,
    fields: {
      library_id: projectId ? libraryId : (libraryId || 'default'),
      project_id: projectId,
    },
  })

export const searchPapers = ({ query, yearMin, yearMax, category, authors, topK = 10 }) =>
  requestJson({
    path: '/api/v1/papers/search',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        year_min: yearMin,
        year_max: yearMax,
        category,
        authors,
        top_k: topK,
      }),
    },
  })

export const addNote = ({
  paperId = '',
  projectId = '',
  conversationId = '',
  sessionId = '',
  content = '',
  noteType = 'reading_note',
  sourceType = 'manual',
  citation = '',
  excerpt = '',
  evidenceReference = {},
} = {}) =>
  requestJson({
    path: '/api/v1/notes',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paper_id: paperId || undefined,
        project_id: projectId || undefined,
        conversation_id: conversationId || undefined,
        session_id: sessionId || undefined,
        content,
        note_type: noteType,
        source_type: sourceType,
        citation: citation || undefined,
        excerpt: excerpt || undefined,
        evidence_reference: evidenceReference && Object.keys(evidenceReference).length ? evidenceReference : undefined,
      }),
    },
  })

export const listNotes = ({
  paperId = '',
  projectId = '',
  conversationId = '',
  sessionId = '',
  noteType = '',
  sourceType = '',
  includeProjectPapers = false,
  includeProjectSessions = false,
  limit = 200,
} = {}) => {
  const useLegacyPaperRoute = paperId
    && !projectId
    && !conversationId
    && !sessionId
    && !noteType
    && !sourceType
    && !includeProjectPapers
    && !includeProjectSessions

  const path = useLegacyPaperRoute
    ? `/api/v1/notes/${encodeURIComponent(paperId)}`
    : (() => {
        const params = new URLSearchParams()
        if (paperId) params.set('paper_id', paperId)
        if (projectId) params.set('project_id', projectId)
        if (conversationId) params.set('conversation_id', conversationId)
        if (sessionId) params.set('session_id', sessionId)
        if (noteType) params.set('note_type', noteType)
        if (sourceType) params.set('source_type', sourceType)
        if (includeProjectPapers) params.set('include_project_papers', 'true')
        if (includeProjectSessions) params.set('include_project_sessions', 'true')
        if (limit) params.set('limit', String(limit))
        const query = params.toString()
        return `/api/v1/notes${query ? `?${query}` : ''}`
      })()

  return requestJson({ path }).then((data) => ({
    ...data,
    notes: normalizeNoteCollection(data?.notes),
  }))
}

export const getCitations = ({ paperId, depth = 1 }) =>
  requestJson({ path: `/api/v1/citations/${encodeURIComponent(paperId)}?depth=${depth}&include_graph_paths=true` })
