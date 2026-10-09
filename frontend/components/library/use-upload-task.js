'use client'

import { useEffect, useRef, useState } from 'react'
import * as api from '../../lib/api/tasks'
import { assertAuthContext, captureAuthContext, isCurrentAuthContext } from '../../lib/auth-client'
import { acceptOrRecoverUpload, observeUploadTask, uploadStage } from '../../lib/upload-task-flow.mjs'
import { toMessage } from '../../lib/study-assistant-workspace.mjs'

const INITIAL_STATE = { uploadFile: null, uploadBusy: false, uploadTask: null, uploadResult: null,
  uploadProgress: 0, uploadStage: 'idle', uploadStatusMessage: '', uploadElapsed: 0, libraryError: '' }
const TRANSFER_PROGRESS_LIMIT = 18
const INTENT_STORAGE_PREFIX = 'sa:upload-intent:v1:'

/** @param {object} intent 受理意图 @param {object} request 当前文件及目标 @returns {boolean} 是否为原意图的再次发送 */
function sameIntent(intent, request) {
  return intent?.filename === request.file.name && intent?.size === request.file.size
    && intent?.lastModified === request.file.lastModified && intent?.projectId === request.projectId
}

/** @param {object} request 原文件与目标 @param {object} context 当前账号 @returns {object} 可重载查回的意图 */
function newIntent(request, context) {
  return { requestId: crypto.randomUUID(), userId: context.userId, filename: request.file.name,
    size: request.file.size, lastModified: request.file.lastModified, projectId: request.projectId }
}

/** @param {Function} action 单次受理或继续 @param {Function|undefined} refresh 论文刷新 @returns {Promise<void>} 观察到终态 */
async function runUploadAction({ action, refresh, controllerRef, startedRef, setState, pushFeed }) {
  controllerRef.current?.abort()
  const controller = new AbortController()
  controllerRef.current = controller
  const authContext = captureAuthContext()
  const options = { authContext, signal: controller.signal }
  startedRef.current = Date.now()
  setState((value) => ({ ...value, uploadBusy: true, libraryError: '', uploadElapsed: 0 }))
  try {
    const task = await action(options)
    const finished = await observeUploadTask({ task, api, ...options, onTask: (next) => {
      assertAuthContext(authContext)
      setState((value) => ({ ...value, uploadTask: next, uploadResult: next.result || null,
        uploadStage: uploadStage(next), uploadProgress: next.progress || 0, uploadStatusMessage: next.message || '' }))
    } })
    assertAuthContext(authContext)
    if (uploadStage(finished) === 'done' && refresh) {
      await refresh(finished.result.paper_id)
      assertAuthContext(authContext)
      pushFeed(toMessage('assistant', `${finished.result.title || '论文'} 已完成核验，可以检索和提问。`, 'Paper uploaded'))
    }
  } catch (error) {
    if (!controller.signal.aborted && isCurrentAuthContext(authContext)) {
      setState((value) => ({ ...value, libraryError: error.message || '无法获取上传状态，请查回原任务' }))
    }
  } finally {
    if (!controller.signal.aborted && isCurrentAuthContext(authContext)) setState((value) => ({ ...value, uploadBusy: false }))
  }
}

/** @param {object} options 组件生命周期及本账号上传记录 @returns {void} 恢复原任务观察 */
function useRestoredUpload({ intentRef, controllerRef, runAction, setState }) {
  useEffect(() => {
    const context = captureAuthContext()
    try {
      const stored = window.localStorage.getItem(INTENT_STORAGE_PREFIX + context.userId)
      if (stored) {
        const intent = JSON.parse(stored)
        if (!intent.requestId || intent.userId !== context.userId) throw new Error('上传记录损坏，请从任务中心查看')
        intentRef.current = intent
        runAction((options) => api.getUploadReceipt({ ...options, requestId: intent.requestId }))
      }
    } catch (error) {
      setState((value) => ({ ...value, libraryError: error.message }))
    }
    return () => controllerRef.current?.abort()
  }, [])
}

/** @param {object} options 当前原文件和受理意图 @returns {Promise<object>} 已受理或查回的任务 */
async function sendUpload({ request, intentRef, setState }) {
  if (!request.authContext.userId) throw new Error('请先登录后上传论文')
  const intent = sameIntent(intentRef.current, request) ? intentRef.current : newIntent(request, request.authContext)
  intentRef.current = intent
  window.localStorage.setItem(INTENT_STORAGE_PREFIX + request.authContext.userId, JSON.stringify(intent))
  setState((value) => ({ ...value, uploadStage: 'uploading', uploadProgress: 0, uploadTask: null }))
  return acceptOrRecoverUpload({ api, request: { ...request, requestId: intent.requestId, onProgress: ({ percent }) => {
    assertAuthContext(request.authContext)
    setState((value) => ({ ...value, uploadProgress: Math.round(percent / 100 * TRANSFER_PROGRESS_LIMIT) }))
  } } })
}

/** @param {Function} pushFeed 消息提示端口 @returns {object} 持久上传状态及操作 */
export function useUploadTask(pushFeed) {
  const [state, setState] = useState(INITIAL_STATE)
  const controllerRef = useRef(null)
  const intentRef = useRef(null)
  const startedRef = useRef(null)

  useRestoredUpload({ intentRef, controllerRef, runAction: (action) => runUploadAction({ action, controllerRef, startedRef, setState, pushFeed }), setState })

  useEffect(() => {
    if (!state.uploadBusy) return undefined
    const timer = setInterval(() => setState((value) => ({ ...value,
      uploadElapsed: Math.floor((Date.now() - startedRef.current) / 1000) })), 1000)
    return () => clearInterval(timer)
  }, [state.uploadBusy])

  const runAction = (action, refresh) => runUploadAction({ action, refresh, controllerRef, startedRef, setState, pushFeed })

  /** @param {Function} refresh 论文刷新 @param {string} projectId 目标项目 @returns {Promise<void>} 上传执行 */
  async function upload(refresh, projectId = '') {
    if (!state.uploadFile || state.uploadBusy) return
    await runAction((options) => sendUpload({ request: { file: state.uploadFile, projectId, ...options }, intentRef, setState }), refresh)
  }

  const seconds = state.uploadElapsed
  return { ...state, uploadSteps: [], formattedElapsed: seconds < 60 ? `${seconds} 秒` : `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`,
    setUploadFile(file) { intentRef.current = null; setState((value) => ({ ...value, uploadFile: file })) },
    upload,
    resumeUpload: (refresh) => runAction((options) => api.resumeUploadTask({ ...options, taskId: state.uploadTask.task_id }), refresh),
    rebuildUpload: (refresh) => runAction((options) => api.requestUploadOperation({ ...options,
      paperId: state.uploadTask.payload.paper_id, operation: 'rebuild' }), refresh),
    checkUpload: () => runAction((options) => api.getUploadReceipt({ ...options, requestId: intentRef.current.requestId })),
    canCheckUpload: Boolean(intentRef.current?.requestId) }
}
