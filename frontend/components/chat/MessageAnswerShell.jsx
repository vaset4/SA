'use client'

import { useState } from 'react'

import { MessageContent } from './MessageContent'
import { MessageEvidencePanel } from './MessageEvidencePanel'
import { MessageMemoryCapture } from './MessageMemoryCapture'
import { MessageMemoryCandidates } from './MessageMemoryCandidates'
import { MessageResearchPanel } from './MessageResearchPanel'
import { MessageResearchSteps } from './MessageResearchSteps'
import { MessageTracePanel } from './MessageTracePanel'
import { MessageTurnSummary } from './MessageTurnSummary'
import { MessagePersistence } from './MessagePersistence'
import { MessageMemoryWritebackRetry } from './MessageMemoryWritebackRetry'
import { turnDepthView } from '../../lib/research-turn.mjs'
import { currentProviderWait, formatResearchTraceDetail } from '../../lib/research-trace'
import { useWorkspaceStore } from '../../stores/workspace-store'

// 点开后的两块面板：依据看原文，过程看研究说明与轨迹
function AnswerPanels({ message, activePanel }) {
  if (activePanel === 'evidence') {
    return (
      <div className="sa-answer-shell__panel">
        <MessageEvidencePanel message={message} embedded />
      </div>
    )
  }
  if (activePanel === 'process') {
    return (
      <div className="sa-answer-shell__panel sa-answer-shell__panel--process">
        {/* 步骤已在正文上方；此处只补充研究说明与轨迹 */}
        <MessageResearchPanel message={message} embedded />
        <details className="sa-answer-shell__trace">
          <summary>轨迹（可选）</summary>
          <MessageTracePanel message={message} embedded />
        </details>
      </div>
    )
  }
  return null
}

/**
 * 助手答案主卡：默认只见正文 + 主操作；依据/过程点开再看（学 neo-chat 克制）。
 * @param {{ message: object }} props
 */
export function MessageAnswerShell({ message }) {
  const workspace = useWorkspaceStore()
  const running = Boolean(message?.pending)
  // 1. 互斥：'' | evidence | process
  const [activePanel, setActivePanel] = useState('')
  // 2. 按本轮深度决定露出哪些入口；非项目科研回合拿到的是全露出，行为与原先一致
  const depthView = turnDepthView(message?.turn_depth)
  const waiting = running ? currentProviderWait(workspace.researchTrace?.events,
    message?.run_id || workspace.activeRun?.run_id) : null

  const togglePanel = (key) => {
    setActivePanel((current) => (current === key ? '' : key))
  }

  return (
    <div className="sa-answer-shell">
      {/* 3. 【模型服务】【等待展示】所有深度直接显示当前额度等待，不隐藏在研究步骤或过程抽屉 */}
      {waiting ? (
        <div role="status" style={{ marginBottom: 12, color: 'var(--sa-text-soft)', fontSize: 13 }}>
          <strong>等待模型服务额度</strong>
          <p style={{ margin: '4px 0 0' }}>{formatResearchTraceDetail(waiting)}</p>
        </div>
      ) : null}
      {/* 步骤条固定在正文上方：运行中展开进度；结束后默认 4/4 一行，不挤答案。
          直接协助档不摆进度条（那一档只该露回答与记忆标识），要看仍可从「过程」点开 */}
      {depthView.showProgressSteps ? <MessageResearchSteps message={message} /> : null}
      <MessageContent message={message} />
      {!running ? <MessagePersistence message={message} /> : null}
      {!running ? <MessageMemoryWritebackRetry message={message} /> : null}

      {!running ? (
        <>
          {/* 本轮各能力的真实终态：深度、记忆用没用、检索有没有命中、候选写回结果 */}
          <MessageTurnSummary message={message} />
          {/* 写回真产出了候选才渲染，就地纳入/忽略 */}
          <MessageMemoryCandidates message={message} />

          <div className="sa-answer-shell__toolbar" role="toolbar" aria-label="答案操作">
            <MessageMemoryCapture message={message} compact />
            <span className="sa-answer-shell__toolbar-spacer" aria-hidden="true" />
            {/* 直接协助档没发过论文检索，不给依据入口 —— 点开就是空的等于摆一个没发生的动作 */}
            {depthView.showEvidenceToggle ? (
              <button
                type="button"
                className="sa-button sa-button--ghost sa-answer-shell__quiet-btn"
                aria-expanded={activePanel === 'evidence'}
                onClick={() => togglePanel('evidence')}
              >
                依据
              </button>
            ) : null}
            <button
              type="button"
              className="sa-button sa-button--ghost sa-answer-shell__quiet-btn"
              aria-expanded={activePanel === 'process'}
              onClick={() => togglePanel('process')}
            >
              过程
            </button>
          </div>

          <AnswerPanels message={message} activePanel={activePanel} />
        </>
      ) : null}
    </div>
  )
}
