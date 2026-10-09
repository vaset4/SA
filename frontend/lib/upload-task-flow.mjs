/** 上传受理查回与任务观察协议，作者：xxx */
export const UPLOAD_TERMINAL_STATUSES = new Set(['completed', 'paused', 'failed', 'cancelled'])
export const UPLOAD_POLL_INTERVAL_MS = 2500

/** @param {object} task 持久任务 @returns {string} 用户可理解的当前阶段 */
export function uploadStage(task) {
  if (task?.status === 'completed') return (task.upload ? task.upload.searchable : task.result?.searchable) ? 'done' : 'completed'
  if (['paused', 'cancelled', 'failed', 'queued'].includes(task?.status)) return task.status
  return task?.progress >= 60 ? 'vectorizing' : 'parsing'
}

/** @param {number} ms 观察间隔 @param {AbortSignal} signal 取消信号 @returns {Promise<void>} 等待完成 */
export function waitForUploadPoll(ms, signal) {
  return new Promise((resolve, reject) => {
    const onAbort = () => { clearTimeout(timer); reject(signal.reason) }
    const timer = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve() }, ms)
    signal.addEventListener('abort', onAbort, { once: true })
    if (signal.aborted) onAbort()
  })
}

/** @param {object} options 原意图、API 端口和请求参数 @returns {Promise<object>} 唯一原任务 */
export async function acceptOrRecoverUpload({ api, request }) {
  try {
    return await api.createUploadTask(request)
  } catch (error) {
    if (request.signal?.aborted || error?.code === 'AUTH_SESSION_CHANGED' || (error.status && error.status < 500)) throw error
    try {
      return await api.getUploadReceipt(request)
    } catch (lookupError) {
      if (lookupError?.status !== 404) throw lookupError
      const pending = new Error('暂时无法确认上传是否受理，已保留上传标识。可以查回状态，或使用原文件继续发送。')
      pending.code = 'UPLOAD_RECEIPT_PENDING'
      throw pending
    }
  }
}

/** @param {object} options 原任务、API、固定会话和展示回调 @returns {Promise<object>} 持久终态 */
export async function observeUploadTask({ task, api, authContext, signal, onTask, wait = waitForUploadPoll }) {
  if (!task?.task_id) throw new Error('服务端没有返回上传任务标识')
  let current = task
  while (true) {
    signal.throwIfAborted()
    onTask(current)
    if (UPLOAD_TERMINAL_STATUSES.has(current.status)) return current
    await wait(UPLOAD_POLL_INTERVAL_MS, signal)
    current = await api.getTask({ taskId: current.task_id, authContext, signal })
  }
}
