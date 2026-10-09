'use client'

import { useEffect, useMemo, useRef } from 'react'
import { Graph, NodeEvent } from '@antv/g6'

import { LINK_TYPE_META } from '../../lib/research-memory.mjs'

// 三种记忆关系的边样式：基于(蓝实线)/冲突(红虚线)/修正(橙实线箭头)
const LINK_STYLE = {
  based_on: { stroke: '#2563eb', lineDash: [] },
  conflicts_with: { stroke: '#dc2626', lineDash: [4, 4] },
  supersedes: { stroke: '#d97706', lineDash: [] },
}

// 节点按成熟度着色，与台账主轴一致
const MATURITY_FILL = {
  intuition: '#94a3b8',
  leaning: '#2563eb',
  established: '#16a34a',
}

function toGraphData(items, links) {
  const nodes = items.map((item) => ({
    id: String(item.memory_id),
    data: { label: item.title, maturity: item.maturity },
    style: {
      size: 28,
      fill: MATURITY_FILL[item.maturity] || '#94a3b8',
    },
  }))
  const edges = links.map((link) => ({
    id: link.link_id,
    source: String(link.src_memory_id),
    target: String(link.dst_memory_id),
    data: { linkType: link.link_type },
    style: {
      stroke: (LINK_STYLE[link.link_type] || {}).stroke || '#94a3b8',
      lineDash: (LINK_STYLE[link.link_type] || {}).lineDash || [],
      endArrow: link.link_type !== 'conflicts_with',
    },
  }))
  return { nodes, edges }
}

function createGraph(container) {
  return new Graph({
    container,
    autoResize: true,
    animation: false,
    layout: {
      type: 'd3-force',
      link: { distance: 140 },
      manyBody: { strength: -240 },
    },
    node: {
      type: 'circle',
      style: {
        size: (d) => Number(d.style?.size || 28),
        fill: (d) => String(d.style?.fill || '#94a3b8'),
        stroke: '#ffffff',
        lineWidth: 2,
        labelText: (d) => {
          const text = String(d.data?.label || d.id)
          return text.length <= 16 ? text : `${text.slice(0, 15)}…`
        },
        labelPlacement: 'bottom',
        labelFontSize: 11,
        labelFill: '#0f172a',
        cursor: 'pointer',
      },
    },
    edge: {
      type: 'line',
      style: {
        stroke: (d) => String(d.style?.stroke || '#94a3b8'),
        lineDash: (d) => d.style?.lineDash || [],
        endArrow: (d) => Boolean(d.style?.endArrow),
        lineWidth: 1.5,
      },
    },
    behaviors: [{ type: 'drag-canvas' }, { type: 'zoom-canvas' }, { type: 'drag-element' }],
  })
}

export function MemoryGraphView({ items, links, onSelect }) {
  const containerRef = useRef(null)
  const graphRef = useRef(null)
  const onSelectRef = useRef(onSelect)
  const graphData = useMemo(() => toGraphData(items || [], links || []), [items, links])

  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    const container = containerRef.current
    if (!container || graphRef.current) return
    const graph = createGraph(container)
    graphRef.current = graph
    graph.on(NodeEvent.CLICK, (event) => {
      const elementId = event?.target?.id || event?.data?.id
      if (elementId) onSelectRef.current?.(String(elementId))
    })
    return () => {
      graph.destroy()
      graphRef.current = null
    }
  }, [])

  useEffect(() => {
    const graph = graphRef.current
    if (!graph) return
    let cancelled = false
    const sync = async () => {
      graph.setData(graphData)
      await graph.render()
      if (!cancelled && graphData.nodes?.length) {
        await graph.fitView({ padding: 32 })
      }
    }
    sync().catch((error) => { console.error('记忆关系图渲染失败:', error) })
    return () => { cancelled = true }
  }, [graphData])

  return (
    <div className="sa-panel-card" style={{ padding: 0, overflow: 'hidden' }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', padding: '10px 14px', fontSize: 12, borderBottom: '1px solid var(--sa-border)' }}>
        {Object.entries(LINK_TYPE_META).map(([type, label]) => (
          <span key={type} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 16, height: 0, borderTop: `2px ${type === 'conflicts_with' ? 'dashed' : 'solid'} ${(LINK_STYLE[type] || {}).stroke}` }} />
            {label}
          </span>
        ))}
      </div>
      <div ref={containerRef} role="img" aria-label="记忆关系图" style={{ width: '100%', height: 480 }} />
    </div>
  )
}
