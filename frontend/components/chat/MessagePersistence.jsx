'use client'

import { useRef, useState } from 'react'
import { retryRagRunSave } from '../../lib/api/rag-runs'
import { describePersistence, deliveryResponsePatch } from '../../lib/result-delivery.mjs'
import { createId, getError } from '../../lib/study-assistant-workspace.mjs'
import { useWorkspaceStore } from '../../stores/workspace-store'

/**
 * 【结果交付】【重试保存】展示真实保存状态，使用同一请求身份补存原答案
 * @param {{message: object}} props 当前回答及服务端恢复能力
 * @returns {JSX.Element|null} 保存状态和补存操作
 */
export function MessagePersistence({ message }) {
  const workspace = useWorkspaceStore()
  const requestKey = useRef('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const status = describePersistence(message)

  /** @returns {Promise<void>} 提交已暂存答案并按目标及版本应用返回值 */
  async function retrySave() {
    if (busy) return
    requestKey.current ||= createId('save')
    setBusy(true)
    setError('')
    try {
      const result = await retryRagRunSave({ runId: message.run_id, idempotencyKey: requestKey.current })
      if (['failed', 'cancelled', 'timed_out'].includes(result.operation?.status)) requestKey.current = ''
      workspace.applyResultDelivery({ conversationId: workspace.conversationId,
        messageId: message.id, runId: message.run_id, patch: deliveryResponsePatch(result) })
      requestKey.current = ''
    } catch (reason) {
      setError(getError(reason, '保存失败'))
    } finally {
      setBusy(false)
    }
  }

  if (!status) return null
  return (
    <div role="status" style={{ fontSize: 12, marginTop: 8 }}>
      <span>{status.label}{status.detail ? `：${status.detail}` : ''}</span>
      {status.canRetry ? <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={retrySave}>
        {busy ? '保存中' : '重试保存'}
      </button> : null}
      {error ? <p style={{ color: 'var(--sa-danger)' }}>{error}</p> : null}
    </div>
  )
}
