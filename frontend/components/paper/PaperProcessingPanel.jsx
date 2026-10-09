'use client'

import { useEffect, useRef, useState } from 'react'
import { getPaperUploadStatus, requestUploadOperation, resumeUploadTask } from '../../lib/api/tasks'
import { assertAuthContext, captureAuthContext, isCurrentAuthContext } from '../../lib/auth-client'
import { UploadTaskStatus } from '../library/UploadTaskStatus'

const POLL_INTERVAL_MS = 2500
const ACTIVE_STATES = new Set(['queued', 'running', 'processing'])
const RESUMABLE_STATES = new Set(['paused', 'cancelled', 'failed'])

/** @param {string} paperId 论文 @param {Function} onPublished 发布刷新 @returns {object} 独立增强与重建操作 */
function usePaperProcessing(paperId, onPublished) {
  const [task, setTask] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [revision, setRevision] = useState(0)
  const generation = useRef(0)
  useEffect(() => {
    generation.current += 1
    setTask(null); setError(''); setBusy(false)
    const controller = new AbortController()
    const authContext = captureAuthContext()
    let timer
    let lastBatch = ''
    const poll = async () => {
      try {
        const next = await getPaperUploadStatus({ paperId, authContext, signal: controller.signal })
        setTask(next)
        if (lastBatch && lastBatch !== next.upload?.published_batch_id) await onPublished()
        lastBatch = next.upload?.published_batch_id || ''
        const active = [next.status, next.upload?.metadata_status, next.upload?.graph_status].some((value) => ACTIVE_STATES.has(value))
        if (active && !controller.signal.aborted) timer = setTimeout(poll, POLL_INTERVAL_MS)
      } catch (error) { if (!controller.signal.aborted && isCurrentAuthContext(authContext)) setError(error.status === 404 ? '这篇旧论文尚无恢复记录，可重新上传原文件建立核验结果。' : error.message) }
    }
    poll()
    return () => { generation.current += 1; controller.abort(); clearTimeout(timer) }
  }, [paperId, revision])
  const act = async (operation) => {
    const authContext = captureAuthContext()
    const currentGeneration = generation.current
    setBusy(true); setError('')
    try {
      const current = task?.upload || {}
      const taskId = operation === 'resume' ? task.task_id : current[`${operation}_task_id`]
      const shouldResume = operation === 'resume' || RESUMABLE_STATES.has(current[`${operation}_status`])
      const next = shouldResume ? await resumeUploadTask({ taskId, authContext })
        : await requestUploadOperation({ paperId, operation, authContext })
      assertAuthContext(authContext)
      if (currentGeneration !== generation.current) return
      setTask(next); setRevision((value) => value + 1)
    } catch (error) { if (currentGeneration === generation.current && isCurrentAuthContext(authContext)) setError(error.message) }
    finally { if (currentGeneration === generation.current && isCurrentAuthContext(authContext)) setBusy(false) }
  }
  return { task, error, busy, act }
}

/** @param {object} props 当前论文及刷新回调 @returns {JSX.Element} 基础可用与可选能力 */
export function PaperProcessingPanel({ paperId, onPublished }) {
  const { task, error, busy, act } = usePaperProcessing(paperId, onPublished)
  const upload = task?.upload || {}
  const processing = [task?.status, upload.metadata_status, upload.graph_status].some((value) => ACTIVE_STATES.has(value))
  return (
    <section className="sa-section-card sa-section-card--compact">
      <h3 style={{ margin: 0 }}>论文处理状态</h3>
      <UploadTaskStatus task={task} busy={busy || processing} onResume={() => act('resume')} onRebuild={() => act('rebuild')} />
      {error ? <p style={{ color: 'var(--sa-danger)' }}>{error}</p> : null}
      {upload.metadata_error ? <p className="sa-note">论文信息待补全：{upload.metadata_error}</p> : null}
      {upload.graph_error ? <p className="sa-note">图谱尚未就绪：{upload.graph_error}；检索仍使用已核验正文。</p> : null}
      {upload.searchable ? <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {upload.metadata_status !== 'completed' ? <button type="button" className="sa-button" disabled={busy || processing} onClick={() => act('metadata')}>继续补全论文信息</button> : null}
        {upload.graph_status !== 'completed' ? <button type="button" className="sa-button sa-button--ghost" disabled={busy || processing} onClick={() => act('graph')}>{upload.graph_status === 'not_requested' ? '构建图谱（可选）' : '继续构建图谱'}</button> : null}
      </div> : null}
    </section>
  )
}
