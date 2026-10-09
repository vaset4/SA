'use client'

import Link from 'next/link'
import { StatusBadge } from '../shared/StatusBadge'

const ENRICHMENT_LABELS = { not_requested: '未启用', pending: '待补全', queued: '排队中', processing: '处理中',
  completed: '已完成', incomplete: '仍有缺项', paused: '已暂停', cancelled: '已取消' }

/** @param {object} props 持久任务、允许操作及忙碌状态 @returns {JSX.Element|null} 用户可见状态 */
export function UploadTaskStatus({ task, onResume, onRebuild, busy = false }) {
  if (!task) return null
  const upload = task.upload || {}
  const ready = task.upload ? Boolean(upload.searchable) : Boolean(task.result?.searchable)
  return (
    <div style={{ display: 'grid', gap: 8 }} aria-live="polite">
      <StatusBadge tone={ready ? 'success' : 'warning'}>{ready ? '已可检索' : '尚不可检索'}</StatusBadge>
      {upload.using_previous_result ? <p className="sa-note">重新处理中，目前继续使用上一次核验通过的结果。</p> : null}
      <p className="sa-note">{task.message}</p>
      {task.error ? <p style={{ color: 'var(--sa-danger)', margin: 0 }}>{task.error}</p> : null}
      {ready ? <p className="sa-note">论文信息：{ENRICHMENT_LABELS[upload.metadata_status] || '待确认'} · 图谱（可选）：{ENRICHMENT_LABELS[upload.graph_status] || '未启用'}</p> : null}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {task.allowed_actions?.includes('resume') && onResume ? <button className="sa-button" disabled={busy} onClick={onResume}>继续处理</button> : null}
        {task.allowed_actions?.includes('rebuild') && onRebuild ? <button className="sa-button sa-button--ghost" disabled={busy} onClick={onRebuild}>重新处理原文件</button> : null}
        <Link className="sa-button sa-button--ghost" href="/tasks">查看全部任务</Link>
      </div>
    </div>
  )
}
