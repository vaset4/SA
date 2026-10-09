import { requestJson } from './core'

// 研究记忆层前端 API：薄封装，全部基于 requestJson，路径前缀 /api/v1
// 覆盖研究问题、记忆条目、证据、关系、归属、确认、失效、成熟度、版本、字面搜索、成熟度计数

// ─── 研究问题 ───

export const listQuestions = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${encodeURIComponent(projectId)}/questions` })

export const createQuestion = ({ projectId, title, body = '', isPurpose = false, position = 0 }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/questions`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, body, is_purpose: isPurpose, position }),
    },
  })

export const updateQuestion = ({ projectId, questionId, patch }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/questions/${encodeURIComponent(questionId)}`,
    init: {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  })

export const deleteQuestion = ({ projectId, questionId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/questions/${encodeURIComponent(questionId)}?expected_version=${encodeURIComponent(expectedVersion)}`,
    init: { method: 'DELETE' },
  })

// ─── 记忆条目 ───

export const listMemory = ({ projectId, filters = {} }) => {
  const params = new URLSearchParams()
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') params.append(key, value)
  })
  const query = params.toString()
  return requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory${query ? `?${query}` : ''}`,
  })
}

export const searchMemory = ({ projectId, keyword, staleState }) => {
  const params = new URLSearchParams({ q: keyword })
  if (staleState) params.append('stale_state', staleState)
  return requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/search?${params.toString()}`,
  })
}

export const getMaturityCounts = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/maturity_counts` })

export const createMemory = ({ projectId, payload }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  })

export const getMemory = ({ projectId, memoryId }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}`,
  })

export const updateMemory = ({ projectId, memoryId, patch }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}`,
    init: {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  })

export const deleteMemory = ({ projectId, memoryId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}?expected_version=${encodeURIComponent(expectedVersion)}`,
    init: { method: 'DELETE' },
  })

export const confirmMemory = ({ projectId, memoryId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/confirm`,
    init: { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expected_version: expectedVersion }) },
  })

export const setMemoryMaturity = ({ projectId, memoryId, maturity, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/maturity`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ maturity, expected_version: expectedVersion }),
    },
  })

export const markMemoryStale = ({ projectId, memoryId, reasons = [], expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/stale`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reasons, expected_version: expectedVersion }),
    },
  })

export const listMemoryVersions = ({ projectId, memoryId }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/versions`,
  })

// ─── 证据 ───

export const addEvidence = ({ projectId, memoryId, payload }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/evidence`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  })

export const removeEvidence = ({ projectId, memoryId, refId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/evidence/${encodeURIComponent(refId)}?expected_version=${encodeURIComponent(expectedVersion)}`,
    init: { method: 'DELETE' },
  })

// ─── 记忆关系 ───

export const listMemoryLinks = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory_links` })

export const addMemoryLink = ({ projectId, memoryId, dstMemoryId, linkType, note = '', expectedVersion, dstExpectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/links`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dst_memory_id: dstMemoryId, link_type: linkType, note, expected_version: expectedVersion, dst_expected_version: dstExpectedVersion }),
    },
  })

export const removeMemoryLink = ({ projectId, linkId, expectedVersion, dstExpectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory_links/${encodeURIComponent(linkId)}?expected_version=${encodeURIComponent(expectedVersion)}&dst_expected_version=${encodeURIComponent(dstExpectedVersion)}`,
    init: { method: 'DELETE' },
  })

// ─── 归属研究问题 ───

export const attachQuestion = ({ projectId, memoryId, questionId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/questions/${encodeURIComponent(questionId)}`,
    init: { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expected_version: expectedVersion }) },
  })

export const detachQuestion = ({ projectId, memoryId, questionId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/questions/${encodeURIComponent(questionId)}?expected_version=${encodeURIComponent(expectedVersion)}`,
    init: { method: 'DELETE' },
  })

// ─── 待审区（AI 草稿整理，B6 置信度） ───

export const curateMemory = ({ projectId, scope }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/curate`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scope),
    },
  })

export const listReviewQueue = ({ projectId }) =>
  requestJson({ path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/review` })

export const acceptDraft = ({ projectId, memoryId, expectedVersion, patch = {} }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/accept`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...patch, expected_version: expectedVersion }),
    },
  })

export const dismissDraft = ({ projectId, memoryId, expectedVersion }) =>
  requestJson({
    path: `/api/v1/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(memoryId)}/dismiss`,
    init: { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expected_version: expectedVersion }) },
  })
