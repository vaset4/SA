'use client'

import { useState } from 'react'
import { acceptDraft, updateMemory } from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'

/**
 * 【项目记忆】【编辑确认】编辑当前版本，明确保存草稿或确认修改后的判断
 * @param {{projectId: string, item: object, onDone: Function}} props 当前记忆和刷新入口
 * @returns {JSX.Element} 编辑与确认表单
 */
export function MemoryEditForm({ projectId, item, onDone }) {
  const [title, setTitle] = useState(item.title)
  const [body, setBody] = useState(item.body || '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  /** @param {boolean} confirm 是否确认新判断 @returns {Promise<void>} 完成版本核对后的保存 */
  async function save(confirm) {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const patch = { title, body }
      if (confirm && item.review_state === 'pending_review') {
        await acceptDraft({ projectId, memoryId: item.memory_id, expectedVersion: item.version, patch })
      } else {
        await updateMemory({ projectId, memoryId: item.memory_id,
          patch: { ...patch, expected_version: item.version, confirm } })
      }
      onDone?.()
    } catch (reason) {
      setError(getError(reason, '记忆保存失败'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <label>判断标题<input className="sa-input" value={title} onChange={(event) => setTitle(event.target.value)} /></label>
      <label>判断正文<textarea className="sa-textarea" rows={5} value={body} onChange={(event) => setBody(event.target.value)} /></label>
      <p style={{ fontSize: 12 }}>修改正文后需重新确认；文献型判断同时核对所绑定的依据。</p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => save(false)}>保存修改</button>
        <button type="button" className="sa-button" disabled={busy} onClick={() => save(true)}>{item.review_state === 'pending_review' ? '保存并纳入' : '保存并确认'}</button>
      </div>
      {error ? <p role="alert" style={{ color: 'var(--sa-danger)' }}>{error}</p> : null}
    </div>
  )
}
