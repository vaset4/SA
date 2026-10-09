'use client'

import { useCallback, useEffect, useState } from 'react'
import { WorkspaceLayout } from '../../../components/layout/WorkspaceLayout'
import { EmptyState } from '../../../components/shared/EmptyState'
import { ErrorState } from '../../../components/shared/ErrorState'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'
import { UploadTaskStatus } from '../../../components/library/UploadTaskStatus'
import { cancelTask, listTasks, resumeUploadTask, requestUploadOperation } from '../../../lib/api/tasks'
import { captureAuthContext, assertAuthContext, isCurrentAuthContext } from '../../../lib/auth-client'

const TASK_REFRESH_MS = 8000
const STATUS_LABELS = { queued: '排队中', running: '运行中', completed: '已完成', paused: '已暂停', failed: '失败', cancelled: '已取消' }
const OPERATION_LABELS = { ingest: '上传论文', rebuild: '重新处理论文', metadata: '论文信息补全', graph: '可选图谱构建' }

/** @param {object} props 持久任务和操作 @returns {JSX.Element} 本账号任务卡片 */
function TaskCard({ task, act, busy }) {
  const status = task.status
  const isUpload = task.task_type === 'upload_ingest'
  const canCancel = task.task_type !== 'paper_delete' && ['running', 'queued', 'paused'].includes(status)
  return (
    <article className="sa-list-card" style={{ cursor: 'default', display: 'grid', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <strong>{isUpload ? OPERATION_LABELS[task.payload?.operation || 'ingest'] : task.task_type}</strong>
        <StatusBadge tone={status === 'completed' ? 'success' : status === 'failed' ? 'danger' : 'warning'}>{STATUS_LABELS[status] || status}</StatusBadge>
      </div>
      <span className="sa-note">{task.payload?.filename || task.paper_title || ''}</span>
      {isUpload ? <UploadTaskStatus task={task} busy={busy}
        onResume={() => act(task, 'resume')} onRebuild={() => act(task, 'rebuild')} />
        : <p className="sa-note">{task.error || task.message}</p>}
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <span className="sa-note">{task.created_at ? new Date(task.created_at).toLocaleString('zh-CN') : ''}</span>
        {canCancel ? <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => act(task, 'cancel')}>取消任务</button> : null}
      </div>
    </article>
  )
}

/** @returns {object} 本账号任务列表及显式操作，不在页面自动恢复暂停任务 */
function useTaskList() {
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')
  const refresh = useCallback(async (options = {}) => {
    const authContext = options.authContext || captureAuthContext()
    try { setTasks(await listTasks({ authContext, signal: options.signal })); setError('') }
    catch (error) { if (!options.signal?.aborted && isCurrentAuthContext(authContext)) setError(error.message || '任务列表加载失败') }
    finally { if (!options.signal?.aborted && isCurrentAuthContext(authContext)) setLoading(false) }
  }, [])
  useEffect(() => {
    let active = true
    let timer
    const controller = new AbortController()
    const authContext = captureAuthContext()
    const poll = async () => { await refresh({ authContext, signal: controller.signal }); if (active) timer = setTimeout(poll, TASK_REFRESH_MS) }
    poll()
    return () => { active = false; controller.abort(); clearTimeout(timer) }
  }, [refresh])
  const act = async (task, action) => {
    const authContext = captureAuthContext()
    setBusyId(task.task_id)
    try {
      if (action === 'cancel') await cancelTask({ taskId: task.task_id, authContext })
      else if (action === 'resume') await resumeUploadTask({ taskId: task.task_id, authContext })
      else await requestUploadOperation({ paperId: task.payload.paper_id, operation: 'rebuild', authContext })
      assertAuthContext(authContext)
      await refresh()
    } catch (error) { if (isCurrentAuthContext(authContext)) setError(error.message || '任务操作失败') }
    finally { if (isCurrentAuthContext(authContext)) setBusyId('') }
  }
  return { tasks, loading, error, busyId, refresh, act }
}

/** @returns {JSX.Element} 普通登录用户可访问的任务中心 */
export default function TasksPage() {
  const state = useTaskList()
  const [filter, setFilter] = useState('all')
  const tasks = filter === 'all' ? state.tasks : state.tasks.filter((task) => task.status === filter)
  return (
    <WorkspaceLayout focusMode header={<PageHeader eyebrow="Tasks" variant="workspace" title="任务中心"
      description="查看上传进度、失败原因，继续已保存的处理任务。"
      actions={<button type="button" className="sa-button" onClick={state.refresh}>刷新</button>} />}>
      <div className="sa-workbench">
        <section className="sa-section-card sa-section-card--accent"><div className="sa-mode-group">
          {['all', 'running', 'queued', 'paused', 'completed', 'failed', 'cancelled'].map((value) => (
            <button key={value} type="button" className="sa-mode-button" data-active={filter === value} onClick={() => setFilter(value)}>{value === 'all' ? '全部' : STATUS_LABELS[value]}</button>
          ))}
        </div></section>
        {state.error ? <ErrorState title="任务操作未完成" description={state.error} onRetry={state.refresh} /> : null}
        {state.loading ? <p>加载中...</p> : tasks.length ? <div style={{ display: 'grid', gap: 12 }}>
          {tasks.map((task) => <TaskCard key={task.task_id} task={task} act={state.act} busy={state.busyId === task.task_id} />)}
        </div> : <EmptyState title="暂无任务" description="上传、信息补全及图谱任务会显示在这里。" />}
      </div>
    </WorkspaceLayout>
  )
}
