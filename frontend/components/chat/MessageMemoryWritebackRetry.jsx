'use client'

import { useState } from 'react'
import { retryRagRunMemoryWriteback } from '../../lib/api/rag-runs'
import { hasPendingCandidates } from '../../lib/result-delivery.mjs'
import { getError } from '../../lib/study-assistant-workspace.mjs'
import { useWorkspaceStore } from '../../stores/workspace-store'

/**
 * 【项目记忆】【重试候选】对已提交答案重试失败的候选处理，不重新生成研究答案
 * @param {{message: object}} props 原答案和候选处理事实
 * @returns {JSX.Element|null} 服务端明确允许的重试操作
 */
export function MessageMemoryWritebackRetry({ message }) {
  const workspace = useWorkspaceStore()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  /** @returns {Promise<void>} 恢复同一已提交版本的候选处理并更新当前显示 */
  async function retry() {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const result = await retryRagRunMemoryWriteback({ runId: message.run_id })
      workspace.applyMemoryWriteback({ conversationId: workspace.conversationId, result })
      workspace.reloadCandidates(message.run_id)
    } catch (reason) {
      setError(getError(reason, '候选处理重试失败'))
    } finally { setBusy(false) }
  }

  // 1. 【项目记忆】【提交后操作】未提交答案只允许补存，不查询或重试尚未开始的候选处理
  if (['pending', 'not_staged'].includes(message.persistence_status)) return null

  if (workspace.candidateQueryRunId === message.run_id && workspace.candidateQueryError) return (
    <div><p role="alert">{workspace.candidateQueryError}</p>
      <button type="button" className="sa-button sa-button--ghost" onClick={() => workspace.reloadCandidates(message.run_id)}>重新查询候选状态</button>
    </div>
  )

  if (hasPendingCandidates(message)) return (
    <button type="button" className="sa-button sa-button--ghost" onClick={() => workspace.reloadCandidates(message.run_id)}>查询候选状态</button>
  )

  if (message.memory_writeback?.status !== 'failed' || message.memory_writeback?.retryable !== true) return null
  return (
    <div>
      <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={retry}>
        {busy ? '处理候选中' : '重试生成记忆候选'}
      </button>
      {error ? <p role="alert" style={{ color: 'var(--sa-danger)' }}>{error}</p> : null}
    </div>
  )
}
