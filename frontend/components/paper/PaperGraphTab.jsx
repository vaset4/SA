'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { EmptyState } from '../shared/EmptyState'
import { NoteCaptureCard } from '../notes/NoteCaptureCard'
import { StatusBadge } from '../shared/StatusBadge'
import { getPaperSubgraph } from '../../lib/api/research'
import { formatRelationLabel } from '../../lib/relation-meanings.mjs'
import { PaperGraphCanvas } from '../paper-graph/PaperGraphCanvas'

function relationLabel(type) {
  return formatRelationLabel(String(type || '').toUpperCase())
}

function buildGraphObservationNote({ paperId, title, activeNode, activeEdges, nodes }) {
  const lines = []
  if (activeNode?.id === `paper:${paperId}`) {
    lines.push(`图谱观察对象：${title || activeNode?.title || paperId}`)
    lines.push('观察重点：当前中心论文在研究对象图中的高信号关系。')
  } else {
    lines.push(`图谱观察对象：${activeNode?.title || activeNode?.id || '当前节点'}`)
    lines.push(`中心论文：${title || paperId}`)
  }
  if (activeEdges.length) {
    lines.push('关联关系：')
    activeEdges.slice(0, 5).forEach((edge) => {
      const peerId = edge.source === activeNode?.id ? edge.target : edge.source
      const peerNode = nodes.find((node) => node.id === peerId)
      lines.push(`- ${relationLabel(edge.type)} · ${peerNode?.title || peerId}`)
    })
  } else {
    lines.push('关联关系：当前节点暂无更多可见关系。')
  }
  return lines.join('\n')
}

export function PaperGraphTab({ paperId, title = '', projectId = '' }) {
  const [state, setState] = useState({
    loading: true,
    error: '',
    graph: null,
    activeNodeId: '',
  })

  useEffect(() => {
    if (!paperId) return
    let active = true
    setState({ loading: true, error: '', graph: null, activeNodeId: '' })
    getPaperSubgraph({ paperId })
      .then((graph) => {
        if (!active) return
        setState({
          loading: false,
          error: '',
          graph,
          activeNodeId: `paper:${paperId}`,
        })
      })
      .catch((reason) => {
        if (!active) return
        setState({
          loading: false,
          error: reason instanceof Error ? reason.message : '论文子图加载失败',
          graph: null,
          activeNodeId: '',
        })
      })
    return () => {
      active = false
    }
  }, [paperId])

  const canvasData = useMemo(() => {
    if (!state.graph) return { nodes: [], edges: [] }
    const centerId = `paper:${paperId}`
    const entityNodes = (state.graph.nodes || []).map((node) => ({
      id: node.id,
      title: node.name || node.label || node.id,
      paper_id: `${nodeLabel(node.label)} · ${node.name || node.id}`,
      year: null,
    }))
    return {
      nodes: [
        {
          id: centerId,
          title: title || state.graph.paper?.title || paperId,
          paper_id: paperId,
          year: null,
        },
        ...entityNodes,
      ],
      edges: (state.graph.edges || []).map((edge) => ({
        source: edge.source,
        target: edge.target,
        type: edge.type,
        weight: 1,
        shared_items: [],
      })),
    }
  }, [paperId, state.graph, title])

  const activeNode = useMemo(
    () => canvasData.nodes.find((node) => node.id === state.activeNodeId) || canvasData.nodes[0] || null,
    [canvasData.nodes, state.activeNodeId],
  )

  const activeEdges = useMemo(
    () => canvasData.edges.filter((edge) => edge.source === activeNode?.id || edge.target === activeNode?.id),
    [activeNode?.id, canvasData.edges],
  )
  const graphNoteDraft = useMemo(
    () => buildGraphObservationNote({ paperId, title, activeNode, activeEdges, nodes: canvasData.nodes }),
    [activeEdges, activeNode, canvasData.nodes, paperId, title],
  )

  if (state.loading) {
    return (
      <section style={{ border: '1px solid var(--sa-border)', borderRadius: 12, padding: 16 }}>
        <p style={{ margin: 0, color: 'var(--sa-text-soft)', fontSize: 13 }}>论文子图加载中...</p>
      </section>
    )
  }

  if (state.error) {
    return <EmptyState title="论文子图加载失败" description={state.error} />
  }

  if (!canvasData.nodes.length || !state.graph?.meta?.node_count) {
    return (
      <EmptyState
        title="当前论文还没有子图"
        description="如果这篇论文刚上传，可能需要等待关系结构写入完成；也可能当前还没有抽取到研究对象实体。"
      />
    )
  }

  return (
    <section style={{ display: 'grid', gap: 14 }}>
      <div className="sa-tool-card" style={{ padding: 14, display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <StatusBadge tone="neutral">实体 {state.graph.meta?.node_count || 0}</StatusBadge>
          <StatusBadge tone="neutral">关系 {state.graph.meta?.edge_count || 0}</StatusBadge>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link href="/graph" className="sa-button sa-button--ghost">
            打开完整图谱
          </Link>
        </div>
      </div>

      <div style={{ display: 'grid', gap: 14, gridTemplateColumns: 'minmax(0, 1.4fr) minmax(260px, 0.8fr)' }}>
        <PaperGraphCanvas
          nodes={canvasData.nodes}
          edges={canvasData.edges}
          activeNodeId={activeNode?.id || ''}
          onSelect={(node) => setState((current) => ({ ...current, activeNodeId: node.id }))}
        />

        <aside className="sa-panel-card" style={{ padding: 16, display: 'grid', gap: 12, alignContent: 'start' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16 }}>{activeNode?.title || '图谱节点'}</h3>
            <p style={{ margin: '8px 0 0', fontSize: 12, color: 'var(--sa-text-soft)' }}>
              {activeNode?.id === `paper:${paperId}`
                ? '中心节点代表当前论文；周围节点仅展示高信号研究对象，例如方法、数据集、指标和任务。'
                : activeNode?.paper_id || activeNode?.id}
            </p>
          </div>

          <NoteCaptureCard
            title="保存图谱观察"
            description="把当前节点和关系观察沉淀为研究笔记，后续可以在项目笔记里继续整理。"
            placeholder="记录当前节点的重要关系、可疑点或后续追问"
            initialContent={graphNoteDraft}
            resetKey={`${paperId}:${activeNode?.id || 'root'}`}
            notePayload={{
              paperId,
              projectId,
              evidenceReference: {
                graph_node_id: activeNode?.id || '',
                graph_neighbors: activeEdges.slice(0, 5).map((edge) => ({
                  relation: edge.type,
                  peer_id: edge.source === activeNode?.id ? edge.target : edge.source,
                })),
              },
            }}
            defaultNoteType={activeNode?.id === `paper:${paperId}` ? 'reading_note' : 'comparison_note'}
            sourceType="manual"
            submitLabel="保存观察"
          />

          <div style={{ display: 'grid', gap: 8 }}>
            {activeEdges.length ? activeEdges.map((edge) => {
              const peerId = edge.source === activeNode?.id ? edge.target : edge.source
              const peerNode = canvasData.nodes.find((node) => node.id === peerId)
              return (
                <article
                  key={`${edge.source}-${edge.target}-${edge.type}`}
                  style={{
                    padding: 12,
                    borderRadius: 12,
                    border: '1px solid var(--sa-border)',
                    background: 'rgba(255,255,255,0.82)',
                    display: 'grid',
                    gap: 4,
                  }}
                >
                  <strong style={{ fontSize: 13 }}>{peerNode?.title || peerId}</strong>
                  <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{relationLabel(edge.type)}</span>
                </article>
              )
            }) : (
              <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>当前节点暂无更多关系细节。</p>
            )}
          </div>
        </aside>
      </div>
    </section>
  )
}

function nodeLabel(label) {
  const mapping = {
    Method: '方法',
    Dataset: '数据集',
    Metric: '指标',
    Task: '任务',
  }
  return mapping[String(label || '')] || '实体'
}
