'use client'

import { useState } from 'react'
import Link from 'next/link'

import { createMemory, curateMemory } from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'
import { useWorkspaceStore } from '../../stores/workspace-store'

/**
 * 从助手消息提炼一条「判断」标题：优先 answer 首行，否则研究问题，否则截断正文。
 * @param {object} message
 * @returns {string}
 */
function deriveMemoryTitle(message) {
  const content = String(message?.content || '').trim()
  if (content) {
    const firstLine = content.split('\n').map((line) => line.trim()).find(Boolean) || ''
    const cleaned = firstLine.replace(/^#+\s*/, '').replace(/【注意】.*/, '').trim()
    if (cleaned) return cleaned.slice(0, 120)
  }
  const question = String(message?.research_plan?.question || message?.answer_brief?.question || '').trim()
  if (question) return question.slice(0, 120)
  return '来自对话的研究判断'
}

/**
 * 组装记忆正文：答案 + 可选问题上下文。
 * @param {object} message
 * @returns {string}
 */
function deriveMemoryBody(message) {
  const parts = []
  const question = String(message?.research_plan?.question || '').trim()
  if (question) parts.push(`研究问题：${question}`)
  const content = String(message?.content || '').trim()
  if (content) parts.push(content)
  return parts.join('\n\n')
}

/**
 * 聊天主路径：把当前回答沉淀到项目研究记忆（快速保存 / AI 待审）。
 * compact=true 时只渲染工具条按钮（嵌在答案主卡底栏），不单独占一块说明卡。
 * @param {{ message: object, compact?: boolean }} props
 */
export function MessageMemoryCapture({ message, compact = false }) {
  const workspace = useWorkspaceStore()
  const projectId = String(workspace.projectId || '').trim()
  const conversationId = String(workspace.conversationId || '').trim()
  const [busy, setBusy] = useState('')
  const [status, setStatus] = useState('')

  // 用户消息或无正文时不展示沉淀条
  if (message?.role === 'user') return null
  if (message?.pending) return null
  if (!String(message?.content || '').trim()) return null

  const memoryHref = projectId
    ? `/projects/${encodeURIComponent(projectId)}?tab=memory`
    : '/projects'

  const handleQuickSave = async () => {
    if (!projectId) {
      setStatus('请先绑定当前项目，再到项目「研究记忆」查看。')
      return
    }
    setBusy('save')
    setStatus('')
    try {
      // 1. 快速保存为用户草稿判断（不经 AI），成熟度 intuition
      await createMemory({
        projectId,
        payload: {
          title: deriveMemoryTitle(message),
          body: deriveMemoryBody(message),
          memory_type: 'decision_note',
          maturity: 'intuition',
          evidence_mode: 'user_note_only',
          tags: ['from_chat'],
          related_paper_ids: Array.isArray(message?.research_scope?.paper_ids)
            ? message.research_scope.paper_ids
            : [],
        },
      })
      setStatus('已保存到研究记忆（直觉）。可在项目页完善证据与成熟度。')
    } catch (reason) {
      setStatus(getError(reason, '保存到研究记忆失败'))
    } finally {
      setBusy('')
    }
  }

  const handleCurate = async () => {
    if (!projectId) {
      setStatus('请先绑定当前项目，再使用 AI 整理。')
      return
    }
    if (!conversationId) {
      setStatus('当前没有会话 ID，无法按对话圈定 AI 整理范围。')
      return
    }
    setBusy('curate')
    setStatus('')
    try {
      // 2. 用当前 conversation 圈定范围，产出 pending_review 待审草稿
      const drafts = await curateMemory({
        projectId,
        scope: { conversation_id: conversationId },
      })
      const count = Array.isArray(drafts) ? drafts.length : (drafts?.length || 0)
      setStatus(
        count
          ? `AI 已生成 ${count} 条待审记忆，请到项目「研究记忆 → AI 待审区」确认。`
          : 'AI 整理完成，未产生新草稿（可能与已有记忆重复或材料不足）。',
      )
    } catch (reason) {
      setStatus(getError(reason, 'AI 整理失败'))
    } finally {
      setBusy('')
    }
  }

  // 主卡底栏：主操作 2 个 + 记忆入口；状态用 toast 语义短文
  if (compact) {
    return (
      <>
        <button
          type="button"
          className="sa-button"
          disabled={Boolean(busy) || !projectId}
          onClick={handleQuickSave}
          title={!projectId ? '请先绑定项目' : undefined}
        >
          {busy === 'save' ? '保存中' : '留下判断'}
        </button>
        <button
          type="button"
          className="sa-button sa-button--ghost"
          disabled={Boolean(busy) || !projectId || !conversationId}
          onClick={handleCurate}
          title={!projectId ? '请先绑定项目' : undefined}
        >
          {busy === 'curate' ? '整理中' : 'AI 整理'}
        </button>
        <Link
          href={memoryHref}
          className="sa-answer-shell__text-link"
          style={{ fontSize: 12, color: 'var(--sa-text-faint)', alignSelf: 'center' }}
        >
          记忆
        </Link>
        {status ? (
          <span className="sa-answer-shell__status" role="status">{status}</span>
        ) : null}
      </>
    )
  }

  return (
    <div
      className="sa-stack"
      style={{
        marginTop: 10,
        padding: 10,
        border: '1px dashed var(--sa-border)',
        borderRadius: 10,
        background: 'var(--sa-surface-soft, #fafafa)',
      }}
    >
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>沉淀为研究记忆</p>
        {!projectId ? (
          <span className="sa-chip" style={{ fontSize: 12 }}>需绑定项目</span>
        ) : null}
      </div>
      <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>
        把本轮结论留下：可快速保存为「我的判断」，或让 AI 按本对话整理成待审草稿。
      </p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button
          type="button"
          className="sa-button"
          disabled={Boolean(busy) || !projectId}
          onClick={handleQuickSave}
        >
          {busy === 'save' ? '保存中' : '快速保存判断'}
        </button>
        <button
          type="button"
          className="sa-button sa-button--ghost"
          disabled={Boolean(busy) || !projectId || !conversationId}
          onClick={handleCurate}
        >
          {busy === 'curate' ? 'AI 整理中' : 'AI 整理本对话'}
        </button>
        <Link href={memoryHref} className="sa-button sa-button--ghost" style={{ display: 'inline-flex', alignItems: 'center' }}>
          打开研究记忆
        </Link>
      </div>
      {status ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>{status}</p>
      ) : null}
    </div>
  )
}
