'use client'

import { StatusBadge } from '../shared/StatusBadge'

export function GraphInterpretationPanel({ interpretation }) {
  if (!interpretation) return null
  return (
    <section className="sa-evidence-desk sa-evidence-desk-panel sa-evidence-desk-panel--wide">
      <header className="sa-evidence-desk__head">
        <h3 className="sa-evidence-desk__title">图谱解读</h3>
        <span className="sa-evidence-desk__badge sa-evidence-desk__badge--blue">
          {interpretation.totalRelations} edges
        </span>
      </header>
      <div className="sa-chip-list">
        <StatusBadge tone={interpretation.lowSignal ? 'warning' : 'success'}>
          {interpretation.lowSignal ? '关系不足' : '可解读'}
        </StatusBadge>
        <span className="sa-chip">论文 {interpretation.totalPapers}</span>
        <span className="sa-chip">关系 {interpretation.totalRelations}</span>
      </div>
      {interpretation.byRelation.length ? (
        <div className="sa-evidence-desk__grid">
          {interpretation.byRelation.map((item) => (
            <article key={item.type} className="sa-evidence-desk__card">
              <span className="sa-evidence-desk__label">{item.label} {item.count}</span>
              <strong>{item.meaning}</strong>
            </article>
          ))}
        </div>
      ) : (
        <p className="sa-note">当前还没有稳定关系边，暂不生成趋势判断。</p>
      )}
      {interpretation.topSharedItems.length ? (
        <p className="sa-note">
          高频共享线索：{interpretation.topSharedItems.map((item) => `${item.item}(${item.count})`).join('，')}
        </p>
      ) : null}
    </section>
  )
}
