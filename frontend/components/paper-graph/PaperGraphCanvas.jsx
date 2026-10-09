'use client'

import { useEffect, useMemo, useRef } from 'react'
import { CanvasEvent, Graph, NodeEvent } from '@antv/g6'

const DEFAULT_EDGE_COLOR = 'rgba(59,130,246,0.42)'
const EDGE_COLORS = {
  USES_METHOD: '#2563eb',
  STUDIES_TASK: '#0f766e',
  USES_DATASET: '#d97706',
  PUBLISHED_IN: '#475569',
}

function truncateLabel(value, limit = 28) {
  const text = String(value || '').trim()
  if (text.length <= limit) return text
  return `${text.slice(0, limit - 1)}…`
}

function edgeId(edge) {
  return `edge:${edge.source}:${edge.target}:${edge.type}`
}

function buildYearBounds(nodes) {
  const years = nodes
    .map((node) => Number(node.year))
    .filter((year) => Number.isFinite(year) && year > 0)
  if (!years.length) {
    return { min: null, max: null }
  }
  return {
    min: Math.min(...years),
    max: Math.max(...years),
  }
}

function nodeColor(node, yearBounds) {
  const year = Number(node.year)
  if (!Number.isFinite(year) || year <= 0 || yearBounds.min === null || yearBounds.max === null) {
    return '#94a3b8'
  }
  if (yearBounds.min === yearBounds.max) return '#1d4ed8'
  const ratio = (year - yearBounds.min) / (yearBounds.max - yearBounds.min)
  if (ratio >= 0.66) return '#0f766e'
  if (ratio >= 0.33) return '#1d4ed8'
  return '#d97706'
}

function toGraphData(nodes, edges) {
  const degrees = new Map()
  edges.forEach((edge) => {
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1)
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1)
  })

  const yearBounds = buildYearBounds(nodes)
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      data: {
        ...node,
        degree: degrees.get(node.id) || 0,
      },
      style: {
        size: 24 + Math.min(degrees.get(node.id) || 0, 6) * 3,
        fill: nodeColor(node, yearBounds),
      },
    })),
    edges: edges.map((edge) => ({
      id: edgeId(edge),
      source: edge.source,
      target: edge.target,
      data: edge,
      style: {
        stroke: EDGE_COLORS[edge.type] || DEFAULT_EDGE_COLOR,
        lineWidth: 1 + Math.min(Number(edge.weight || 1), 4),
      },
    })),
  }
}

function buildStateMap(nodes, edges, activeNodeId) {
  if (!activeNodeId) return {}
  const adjacentNodes = new Set([activeNodeId])
  const stateMap = {}

  edges.forEach((edge) => {
    const activeEdge = edge.source === activeNodeId || edge.target === activeNodeId
    const nextEdgeId = edgeId(edge)
    if (activeEdge) {
      adjacentNodes.add(edge.source)
      adjacentNodes.add(edge.target)
      stateMap[nextEdgeId] = ['highlighted']
      return
    }
    stateMap[nextEdgeId] = ['dimmed']
  })

  nodes.forEach((node) => {
    if (node.id === activeNodeId) {
      stateMap[node.id] = ['selected']
      return
    }
    if (adjacentNodes.has(node.id)) {
      stateMap[node.id] = ['neighbor']
      return
    }
    stateMap[node.id] = ['dimmed']
  })

  return stateMap
}

function buildLayout(container) {
  const width = container?.clientWidth || 960
  const height = container?.clientHeight || 560
  return {
    type: 'd3-force',
    animation: false,
    center: [width / 2, height / 2],
    link: {
      distance: 160,
      strength: 0.18,
    },
    manyBody: {
      strength: -260,
    },
    collide: {
      radius: 30,
      iterations: 2,
    },
  }
}

function createGraph(container) {
  return new Graph({
    container,
    autoResize: true,
    animation: false,
    node: {
      type: 'circle',
      style: {
        size: (datum) => Number(datum.style?.size || 26),
        fill: (datum) => String(datum.style?.fill || '#1d4ed8'),
        stroke: '#ffffff',
        lineWidth: 2,
        shadowColor: 'rgba(15, 23, 42, 0.15)',
        shadowBlur: 12,
        cursor: 'pointer',
        labelText: (datum) => truncateLabel(datum.data?.title || datum.data?.paper_id || datum.id),
        labelPlacement: 'bottom',
        labelFontSize: 11,
        labelFill: '#0f172a',
      },
      state: {
        selected: {
          fill: '#d97706',
          stroke: '#7c2d12',
          lineWidth: 4,
          shadowColor: 'rgba(217, 119, 6, 0.38)',
          shadowBlur: 18,
        },
        neighbor: {
          stroke: '#0f766e',
          lineWidth: 3,
        },
        dimmed: {
          opacity: 0.24,
        },
      },
    },
    edge: {
      type: 'line',
      style: {
        stroke: (datum) => String(datum.style?.stroke || DEFAULT_EDGE_COLOR),
        lineWidth: (datum) => Number(datum.style?.lineWidth || 1),
        opacity: 0.46,
      },
      state: {
        highlighted: {
          opacity: 0.96,
          lineWidth: 4,
        },
        dimmed: {
          opacity: 0.08,
        },
      },
    },
    behaviors: [
      { type: 'drag-canvas' },
      { type: 'zoom-canvas', sensitivity: 0.18 },
      { type: 'drag-element-force' },
    ],
  })
}

export function PaperGraphCanvas({
  nodes,
  edges,
  activeNodeId = '',
  onSelect,
  onReady = () => {},
}) {
  const containerRef = useRef(null)
  const graphRef = useRef(null)
  const onSelectRef = useRef(onSelect)
  const onReadyRef = useRef(onReady)
  const autoFitRef = useRef(true)
  const nodesRef = useRef(nodes)
  const activeNodeIdRef = useRef(activeNodeId)
  const graphData = useMemo(() => toGraphData(nodes, edges), [nodes, edges])
  const stateMap = useMemo(() => buildStateMap(nodes, edges, activeNodeId), [nodes, edges, activeNodeId])

  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    nodesRef.current = nodes
  }, [nodes])

  useEffect(() => {
    activeNodeIdRef.current = activeNodeId
  }, [activeNodeId])

  useEffect(() => {
    onReadyRef.current = onReady
  }, [onReady])

  useEffect(() => {
    const container = containerRef.current
    if (!container || graphRef.current) return

    const graph = createGraph(container)
    graphRef.current = graph
    onReadyRef.current({
      fitView: () => graph.fitView({ padding: 32 }),
      focusNode: (nodeId) => {
        if (!nodeId) return
        graph.focusElement(nodeId, { duration: 280 })
      },
    })

    graph.on(NodeEvent.CLICK, async (event) => {
      const elementId = event?.target?.id || event?.data?.id
      if (!elementId) return
      const datum = graph.getElementData(elementId)
      const payload = datum?.data || null
      if (payload?.id) {
        onSelectRef.current?.(payload)
      }
    })

    graph.on(CanvasEvent.CLICK, () => {
      if (!nodesRef.current.length) return
      const fallback = nodesRef.current.find((node) => node.id === activeNodeIdRef.current) || nodesRef.current[0]
      if (fallback) {
        onSelectRef.current?.(fallback)
      }
    })

    return () => {
      onReadyRef.current(null)
      graph.destroy()
      graphRef.current = null
    }
  }, [])

  useEffect(() => {
    const graph = graphRef.current
    const container = containerRef.current
    if (!graph || !container) return

    let cancelled = false
    const sync = async () => {
      graph.setOptions({
        layout: buildLayout(container),
      })
      graph.setData(graphData)
      await graph.render()
      await graph.setElementState(stateMap, false)
      if (!cancelled && autoFitRef.current && graphData.nodes?.length) {
        autoFitRef.current = false
        await graph.fitView({ padding: 32 })
      }
    }

    sync().catch((error) => {
      console.error('Paper graph render failed:', error)
    })

    return () => {
      cancelled = true
    }
  }, [graphData, stateMap])

  return (
    <div
      className="sa-panel-card"
      style={{
        minHeight: 560,
        height: '100%',
        padding: 0,
        overflow: 'hidden',
        background: 'radial-gradient(circle at top, rgba(59,130,246,0.12), transparent 42%), linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
      }}
    >
      <div
        ref={containerRef}
        role="img"
        aria-label="论文关系图"
        style={{ width: '100%', height: 560 }}
      />
    </div>
  )
}
