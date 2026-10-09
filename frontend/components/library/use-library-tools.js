'use client'

import { useState } from 'react'
import { deletePaper } from '../../lib/api/papers'
import { getTask } from '../../lib/api/tasks'
import { assertAuthContext, captureAuthContext } from '../../lib/auth-client'
import { getError, toMessage } from '../../lib/study-assistant-workspace.mjs'
import { useUploadTask } from './use-upload-task'

const DELETE_TASK_POLL_INTERVAL_MS = 1200
const TERMINAL_TASK_STATUS = new Set(['completed', 'failed', 'cancelled'])

/** @param {number} ms 刷新间隔 @returns {Promise<void>} 等待结束 */
function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)) }

/** @param {Function} pushFeed 消息通知 @returns {object} 论文上传与删除操作 */
export function useLibraryTools(pushFeed) {
  const uploadTools = useUploadTask(pushFeed)
  const uploadBusy = uploadTools.uploadBusy
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [libraryError, setLibraryError] = useState('')
  const [deleteResult, setDeleteResult] = useState(null)
  const [deleteStatusMessage, setDeleteStatusMessage] = useState('')

  const remove = async (paperId, refresh) => {
    if (!paperId || uploadBusy || deleteBusy) return
    setDeleteBusy(true)
    setLibraryError('')
    setDeleteResult(null)
    setDeleteStatusMessage('删除任务创建中')
    const authContext = captureAuthContext()
    try {
      let latestTask = await deletePaper({ paperId })
      setDeleteResult(latestTask)
      setDeleteStatusMessage(latestTask?.message || '删除任务已入队')
      await refresh('')
      while (latestTask?.task_id && !TERMINAL_TASK_STATUS.has(latestTask?.status)) {
        await sleep(DELETE_TASK_POLL_INTERVAL_MS)
        assertAuthContext(authContext)
        latestTask = await getTask({ taskId: latestTask.task_id, authContext })
        setDeleteResult(latestTask)
        setDeleteStatusMessage(latestTask?.message || '删除任务执行中')
      }
      if (latestTask?.status !== 'completed') {
        throw new Error(latestTask?.error || latestTask?.message || '删除任务执行失败')
      }
      pushFeed(
        toMessage(
          'assistant',
          `论文 ${paperId} 的删除任务已完成。`,
          'Paper deleted'
        )
      )
    } catch (reason) {
      setLibraryError(getError(reason, '删除失败'))
    } finally {
      setDeleteBusy(false)
    }
  }

  return { ...uploadTools, deleteBusy, deleteResult, deleteStatusMessage,
    libraryError: libraryError || uploadTools.libraryError, remove }
}
