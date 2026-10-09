'use client'

import { useCallback, useEffect, useState } from 'react'
import { WorkspaceLayout } from '../../../components/layout/WorkspaceLayout'
import { EmptyState } from '../../../components/shared/EmptyState'
import { ErrorState } from '../../../components/shared/ErrorState'
import { PageHeader } from '../../../components/shared/PageHeader'
import { getEvolutionGraph } from '../../../lib/api/research'

const VIEW_MODES = [
  { id: 'paper', label: '论文' },
  { id: 'method', label: '方法' },
  { id: 'task', label: '任务' },
]

function EvolutionNode({ node }) {
  return (
    <article className="sa-timeline__item">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <strong style={{ fontSize: 14 }}>{node.label || node.name || node.id}</strong>
        {node.year ? <span className="sa-chip">{node.year}</span> : null}
      </div>
      {node.description ? <span className="sa-note">{node.description}</span> : null}
      {node.papers_count ? (
        <span style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>{node.papers_count} 篇相关论文</span>
      ) : null}
    </article>
  )
}

export default function EvolutionPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [viewMode, setViewMode] = useState('paper')
  const [yearRange, setYearRange] = useState([2018, 2026])

  const fetchEvolution = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setData(await getEvolutionGraph({
        viewMode,
        yearStart: yearRange[0],
        yearEnd: yearRange[1],
      }))
    } catch (error) {
      setError(error instanceof Error ? error.message : '研究演化加载失败')
    }
    setLoading(false)
  }, [viewMode, yearRange])

  useEffect(() => { fetchEvolution() }, [fetchEvolution])

  const nodes = data?.nodes || data?.entries || []

  return (
    <WorkspaceLayout
      focusMode
      header={(
        <PageHeader
          eyebrow="Evolution"
          variant="workspace"
          title="研究演化"
          description="追踪论文、方法和任务的时间线演化。"
          actions={<button type="button" className="sa-button sa-button--ghost" onClick={fetchEvolution}>重新加载</button>}
        />
      )}
    >
      <div className="sa-workbench">
        <section className="sa-section-card sa-section-card--accent">
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            <div className="sa-mode-group">
              {VIEW_MODES.map((m) => (
                <button key={m.id} type="button" className="sa-mode-button" data-active={viewMode === m.id} onClick={() => setViewMode(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 13, color: 'var(--sa-text-soft)' }}>
              <input type="number" className="sa-input" style={{ width: 72, minHeight: 36, padding: '6px 10px' }} value={yearRange[0]} onChange={(e) => setYearRange([+e.target.value, yearRange[1]])} />
              <span>—</span>
              <input type="number" className="sa-input" style={{ width: 72, minHeight: 36, padding: '6px 10px' }} value={yearRange[1]} onChange={(e) => setYearRange([yearRange[0], +e.target.value])} />
            </div>
          </div>
        </section>

        {error ? (
          <ErrorState title="研究演化加载失败" description={error} onRetry={fetchEvolution} />
        ) : loading ? (
          <p style={{ color: 'var(--sa-text-soft)', textAlign: 'center', padding: 40 }}>加载中...</p>
        ) : nodes.length === 0 ? (
          <EmptyState
            title="暂无演化数据"
            description="请先上传论文并完成分析，形成可追踪的时间线节点。"
            action={<button type="button" className="sa-button sa-button--ghost" onClick={fetchEvolution}>重新加载</button>}
          />
        ) : (
          <div className="sa-timeline">
            {nodes.map((node, i) => (
              <EvolutionNode key={node.id || i} node={node} />
            ))}
          </div>
        )}
      </div>
    </WorkspaceLayout>
  )
}
