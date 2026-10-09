'use client'

import { formatRelationLabel } from '../../lib/relation-meanings.mjs'

export function GraphFilters({
  query = '',
  onQueryChange,
  yearBounds,
  yearRange,
  onYearRangeChange,
  relationTypes = [],
  selectedRelations = [],
  onToggleRelation,
  onReset,
  onFitView,
  onFocusSelected,
  activeNodeTitle = '',
}) {
  const hasKnownYears = Number.isFinite(yearBounds?.min) && Number.isFinite(yearBounds?.max)
  const [yearMin, yearMax] = yearRange || [yearBounds?.min || 0, yearBounds?.max || 0]

  return (
    <div className="sa-panel-card" style={{ padding: 16, display: 'grid', gap: 16 }}>
      <div style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <strong style={{ fontSize: 14 }}>筛选器</strong>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button type="button" className="sa-button sa-button--ghost" onClick={onFitView}>
              适配视图
            </button>
            <button
              type="button"
              className="sa-button sa-button--ghost"
              onClick={onFocusSelected}
              disabled={!activeNodeTitle}
            >
              聚焦当前论文
            </button>
            <button type="button" className="sa-button sa-button--ghost" onClick={onReset}>
              重置
            </button>
          </div>
        </div>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="搜索论文标题或 paper_id"
          className="sa-input"
        />
      </div>

      <section style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
          <strong style={{ fontSize: 14 }}>年份范围</strong>
          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
            {hasKnownYears ? `${yearMin} - ${yearMax}` : '暂无可用年份'}
          </span>
        </div>
        {hasKnownYears ? (
          <div style={{ display: 'grid', gap: 10 }}>
            <input
              type="range"
              min={yearBounds.min}
              max={yearBounds.max}
              value={yearMin}
              onChange={(event) => {
                const nextMin = Number(event.target.value)
                onYearRangeChange([Math.min(nextMin, yearMax), Math.max(nextMin, yearMax)])
              }}
            />
            <input
              type="range"
              min={yearBounds.min}
              max={yearBounds.max}
              value={yearMax}
              onChange={(event) => {
                const nextMax = Number(event.target.value)
                onYearRangeChange([Math.min(yearMin, nextMax), Math.max(yearMin, nextMax)])
              }}
            />
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>
            当前图谱节点还没有写入年份字段，先保留全部节点展示。
          </p>
        )}
      </section>

      <section style={{ display: 'grid', gap: 10 }}>
        <strong style={{ fontSize: 14 }}>关系类型</strong>
        {!relationTypes.length ? (
          <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>当前图谱还没有可筛选的关系类型。</p>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {relationTypes.map((relation) => (
              <label
                key={relation}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontSize: 13,
                  color: 'var(--sa-text)',
                }}
              >
                <input
                  type="checkbox"
                  checked={selectedRelations.includes(relation)}
                  onChange={() => onToggleRelation(relation)}
                />
                <span>{formatRelationLabel(relation)}</span>
              </label>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
