'use client'

import { buildPaperReadinessSteps } from '../../lib/paper-readiness-steps.mjs'

const STATUS_LABELS = {
  ready: '就绪',
  running: '处理中',
  pending: '待处理',
  failed: '失败',
  unknown: '未知',
}

function tone(status) {
  if (status === 'ready') return 'success'
  if (status === 'running' || status === 'pending') return 'warning'
  if (status === 'failed') return 'danger'
  return 'neutral'
}

export function PaperReadinessTimeline({ paper }) {
  const steps = buildPaperReadinessSteps(paper)
  return (
    <div className="sa-chip-list" aria-label="论文就绪时间线">
      {steps.map((step) => (
        <span key={step.key} className="sa-chip sa-progress-accent-chip" data-tone={tone(step.status)} title={step.hint || undefined}>
          {step.label}：{STATUS_LABELS[step.status] || step.status}
        </span>
      ))}
    </div>
  )
}
