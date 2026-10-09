'use client'

import { TraceReplayPanel } from '../research/TraceReplayPanel'
import { StatusBadge } from '../shared/StatusBadge'

function StageRow({ stage }) {
  return (
    <article className="sa-research-process-row">
      <div>
        <strong>{stage.label}</strong>
        <p>{stage.detail}</p>
      </div>
      <StatusBadge tone={stage.active ? 'warning' : stage.done ? 'success' : 'neutral'}>
        {stage.active ? '进行中' : stage.done ? '完成' : '待开始'}
      </StatusBadge>
    </article>
  )
}

export function ResearchProcessSection({ model, isDeveloper }) {
  return (
    <section className="sa-section-card">
      <div className="sa-section-card__header">
        <span className="sa-section-card__eyebrow">Process</span>
        <h3 className="sa-section-card__title">研究过程</h3>
      </div>
      <div className="sa-research-process-list">
        {model.stages.map((stage) => <StageRow key={stage.id} stage={stage} />)}
      </div>
      {isDeveloper ? (
        <details>
          <summary className="sa-button sa-button--ghost">展开技术过程</summary>
          <TraceReplayPanel events={model.traceEvents} />
        </details>
      ) : null}
    </section>
  )
}
