import { formatRelationLabel, formatRelationMeaning } from './relation-meanings.mjs'

const TOP_ITEM_LIMIT = 5

function countSharedItems(edges) {
  const counts = new Map()
  for (const edge of edges) {
    for (const item of edge?.shared_items || []) {
      const text = String(item || '').trim()
      if (text) counts.set(text, (counts.get(text) || 0) + 1)
    }
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, TOP_ITEM_LIMIT)
    .map(([item, count]) => ({ item, count }))
}

function rankPapers(nodes, edges) {
  const degrees = new Map()
  for (const edge of edges) {
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1)
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1)
  }
  return nodes
    .map((node) => ({
      paper_id: node.paper_id || node.id,
      title: node.title || node.paper_id || node.id,
      degree: degrees.get(node.id) || 0,
    }))
    .filter((item) => item.degree > 0)
    .sort((left, right) => right.degree - left.degree)
    .slice(0, TOP_ITEM_LIMIT)
}

export function buildGraphInterpretation(graph = {}) {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : []
  const edges = Array.isArray(graph.edges) ? graph.edges : []
  const relationCounts = graph.meta?.relation_counts || {}
  const lowSignal = edges.length === 0
  const byRelation = Object.entries(relationCounts).map(([type, count]) => ({
    type,
    label: formatRelationLabel(type),
    meaning: formatRelationMeaning(type),
    count,
  }))
  return {
    totalPapers: graph.meta?.node_count ?? nodes.length,
    totalRelations: graph.meta?.edge_count ?? edges.length,
    byRelation,
    topPapers: rankPapers(nodes, edges),
    topSharedItems: countSharedItems(edges),
    lowSignal,
    insights: lowSignal ? [] : byRelation.slice(0, 3).map((item) => `${item.label} ${item.count} 条`),
  }
}
