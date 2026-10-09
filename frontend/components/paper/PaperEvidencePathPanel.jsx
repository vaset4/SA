'use client'

import { GraphEvidenceCard } from '../chat/GraphEvidenceCard'
import { EmptyState } from '../shared/EmptyState'

export function PaperEvidencePathPanel({ graphEvidence = [], evidenceSummary = '' }) {
  if (!graphEvidence.length) {
    return (
      <EmptyState
        title="暂无图证据路径"
        description="当前论文尚未命中可解释路径，或图谱中还没有相关连接。"
      />
    )
  }

  return (
    <section className="sa-stack">
      {evidenceSummary ? <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>{evidenceSummary}</p> : null}
      {graphEvidence.map((item) => (
        <GraphEvidenceCard key={item.path_id} item={item} />
      ))}
    </section>
  )
}
