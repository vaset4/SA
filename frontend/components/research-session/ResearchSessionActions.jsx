'use client'

import { useRef, useState } from 'react'
import Link from 'next/link'
import { Button } from '@heroui/react'
import {
  cancelResearchSession,
  retryResearchSession,
  resumeResearchSession,
} from '../../lib/api/research'
import { ReportExport } from '../research/ReportExport'
import { StatusBadge } from '../shared/StatusBadge'
import { researchModeLabel } from '../../lib/research-session-view'
import { retryRagRunSave } from '../../lib/api/rag-runs'
import { createId } from '../../lib/study-assistant-workspace.mjs'

function errorMessage(reason, fallback) {
  return reason instanceof Error ? reason.message : fallback
}

function commandLabel(action) {
  if (action === 'resume') return '提交澄清并继续'
  if (action === 'cancel') return '取消研究'
  if (action === 'save') return '重试保存'
  return '重试研究'
}

function ResumeForm({ answer, setAnswer, busyAction, onExecute }) {
  return (
    <div style={{ display: 'grid', gap: 8, minWidth: 280 }}>
      <textarea
        className="sa-textarea"
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
        placeholder="补充澄清信息后继续研究"
        rows={3}
        disabled={Boolean(busyAction)}
      />
      <Button size="sm" color="primary" onPress={() => onExecute('resume')} isDisabled={Boolean(busyAction)}>
        {busyAction === 'resume' ? '提交中' : commandLabel('resume')}
      </Button>
    </div>
  )
}

function CommandButton({ action, busyAction, onExecute }) {
  const color = action === 'cancel' ? 'danger' : 'primary'
  const pendingLabel = action === 'cancel' ? '取消中' : action === 'retry' ? '重试中' : '提交中'
  return (
    <Button size="sm" color={color} variant="flat" onPress={() => onExecute(action)} isDisabled={Boolean(busyAction)}>
      {busyAction === action ? pendingLabel : commandLabel(action)}
    </Button>
  )
}

function SessionCommands({ model, onSessionChange }) {
  const [answer, setAnswer] = useState('')
  const [busyAction, setBusyAction] = useState('')
  const [error, setError] = useState('')
  const [cancelling, setCancelling] = useState(false)
  const request = useRef(null)
  const actions = Array.isArray(model.allowedActions) ? model.allowedActions : []

  const execute = async (action) => {
    if (action === 'cancel' ? cancelling : busyAction || cancelling) return
    if (action === 'resume' && !answer.trim()) {
      setError('请先填写澄清答案')
      return
    }
    if (action === 'cancel') setCancelling(true)
    else setBusyAction(action)
    setError('')
    try {
      // 1. 网络重发沿用同次身份；内容或动作变化才产生新命令
      const identity = JSON.stringify([model.sessionId, action, action === 'resume' ? answer.trim() : model.resultRunId])
      if (request.current?.identity !== identity) request.current = { identity, key: createId(action) }
      const idempotencyKey = request.current.key
      if (action === 'resume') await resumeResearchSession({ sessionId: model.sessionId, answer: answer.trim(), idempotencyKey })
      if (action === 'retry') await retryResearchSession({ sessionId: model.sessionId, idempotencyKey })
      if (action === 'cancel') await cancelResearchSession({ sessionId: model.sessionId })
      if (action === 'save') await retryRagRunSave({ runId: model.resultRunId, idempotencyKey })
      setAnswer('')
      request.current = null
      onSessionChange?.()
    } catch (reason) {
      setError(errorMessage(reason, `${commandLabel(action)}失败`))
    } finally {
      if (action === 'cancel') setCancelling(false)
      else setBusyAction('')
    }
  }

  if (!actions.length && !model.lastError && !model.persistence && !busyAction) return null
  return (
    <div style={{ display: 'grid', gap: 8, width: '100%' }}>
      {model.lastError ? <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>后端记录：{model.lastError}</p> : null}
      {actions.includes('resume') ? <ResumeForm answer={answer} setAnswer={setAnswer} busyAction={busyAction} onExecute={execute} /> : null}
      {model.persistence ? <p role="status">{model.persistence.label}：{model.persistence.detail}</p> : null}
      {model.cancelRequested ? <p role="status">正在停止，完成清理后才能重试</p> : null}
      {actions.includes('cancel') || busyAction === 'resume' || busyAction === 'retry'
        ? <CommandButton action="cancel" busyAction={cancelling || model.cancelRequested ? 'cancel' : ''} onExecute={execute} /> : null}
      {actions.includes('retry') ? <CommandButton action="retry" busyAction={busyAction} onExecute={execute} /> : null}
      {model.persistence?.canRetry ? <CommandButton action="save" busyAction={busyAction} onExecute={execute} /> : null}
      {error ? <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>{error}</p> : null}
    </div>
  )
}

export function ResearchSessionActions({ model, onSessionChange }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <StatusBadge tone="neutral">{researchModeLabel(model.researchMode)}</StatusBadge>
      <StatusBadge tone="neutral">{model.status}</StatusBadge>
      {model.attemptCount > 0 ? <StatusBadge tone="neutral">第 {model.attemptCount} 次尝试</StatusBadge> : null}
      <ReportExport content={model.report} filename={`research-${model.sessionId}.md`} />
      <ReportExport content={model.traceAppendixMarkdown} filename={`research-${model.sessionId}-trace-appendix.md`} label="导出证据附录" color="default" variant="light" />
      <ReportExport content={model.traceBundleJson} filename={`research-${model.sessionId}-events.json`} label="导出过程 JSON" color="default" variant="light" mimeType="application/json;charset=utf-8" defaultExtension="json" />
      <SessionCommands model={model} onSessionChange={onSessionChange} />
      <Link href="/research" className="sa-button sa-button--ghost">返回研究</Link>
      <Link href="/chat" className="sa-button sa-button--ghost">返回会话</Link>
    </div>
  )
}
