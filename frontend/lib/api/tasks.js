import { requestJson, uploadWithProgress } from './core'

export const listTasks = ({ authContext, signal } = {}) =>
  requestJson({ path: '/api/v1/tasks', authContext, init: { signal } })

export const createUploadTask = ({ file, requestId, onProgress, projectId = '', libraryId = '', graphRequested = false, authContext, signal }) => {
  const resolvedLibraryId = projectId ? libraryId : (libraryId || 'default')
  return uploadWithProgress({
    path: '/api/v1/tasks/upload',
    file,
    onProgress,
    authContext, signal,
    fields: { library_id: resolvedLibraryId, project_id: projectId, request_id: requestId, graph_requested: graphRequested },
  })
}

export const getTask = ({ taskId, authContext, signal }) =>
  requestJson({ path: `/api/v1/tasks/${encodeURIComponent(taskId)}`, authContext, init: { signal } })

export const cancelTask = ({ taskId, authContext }) =>
  requestJson({
    path: `/api/v1/tasks/${encodeURIComponent(taskId)}/cancel`,
    init: { method: 'POST' },
    authContext,
  })

/** @param {object} options 上传意图和原会话 @returns {Promise<object>} 原受理任务 */
export const getUploadReceipt = ({ requestId, authContext, signal }) =>
  requestJson({ path: `/api/v1/tasks/upload/requests/${encodeURIComponent(requestId)}`, authContext, init: { signal } })

/** @param {object} options 任务和原会话 @returns {Promise<object>} 继续后的同一任务 */
export const resumeUploadTask = ({ taskId, authContext, signal }) =>
  requestJson({ path: `/api/v1/tasks/${encodeURIComponent(taskId)}/resume`, authContext, init: { method: 'POST', signal } })

/** @param {object} options 论文、操作和原会话 @returns {Promise<object>} 独立处理任务 */
export const requestUploadOperation = ({ paperId, operation, authContext, signal }) =>
  requestJson({ path: `/api/v1/papers/${encodeURIComponent(paperId)}/upload/${operation}`, authContext, init: { method: 'POST', signal } })

/** @param {object} options 论文和原会话 @returns {Promise<object>} 基础上传及增强状态 */
export const getPaperUploadStatus = ({ paperId, authContext, signal }) =>
  requestJson({ path: `/api/v1/papers/${encodeURIComponent(paperId)}/upload`, authContext, init: { signal } })

export const getTaskEvents = ({ taskId, limit } = {}) => {
  const params = new URLSearchParams()
  if (Number.isFinite(limit) && Number(limit) > 0) {
    params.set('limit', String(Math.trunc(limit)))
  }
  const query = params.toString()
  return requestJson({
    path: `/api/v1/tasks/${encodeURIComponent(taskId)}/events${query ? `?${query}` : ''}`,
  })
}
