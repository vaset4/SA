'use client'

import { MessageContent } from '../chat/MessageContent'
import { MessageEvidencePanel } from '../chat/MessageEvidencePanel'
import { describeMemoryWriteback } from '../../lib/research-turn.mjs'
import { EmptyState } from '../shared/EmptyState'

/** @param {object} props 已提交研究结果模型 @returns {JSX.Element} 原样正文与同次来源 */
export function ResearchReportBody({ model }) {
  const candidates = describeMemoryWriteback(model.resultMessage?.memory_writeback)
  if (!model.report) {
    return <EmptyState title="尚无已保存的研究结果" description="当前操作和保存状态可在上方查看。" />
  }
  return (
    <section className="sa-research-session-paper">
      <p className="sa-research-room-label">Session Report</p>
      <h2>{model.status === 'awaiting_clarification' ? '等待澄清问题' : '研究结论'}</h2>
      {model.persistence ? <p role="status">{model.persistence.label}{model.persistence.detail ? `：${model.persistence.detail}` : ''}</p> : null}
      {candidates ? <p role="status">{candidates.label}：{candidates.detail}</p> : null}
      <div className="sa-research-session-body">
        <MessageContent
          message={model.resultMessage}
          fallback="报告生成中..."
        />
      </div>
      {model.resultMessage.evidence_records?.length ? <MessageEvidencePanel message={model.resultMessage} embedded /> : null}
    </section>
  )
}
