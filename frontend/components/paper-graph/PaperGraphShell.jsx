'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Button } from '@heroui/react'
import { getPaperGraph } from '../../lib/api/research'
import { buildGraphInterpretation } from '../../lib/graph-interpretation.mjs'
import { formatRelationLabel, formatRelationMeaning } from '../../lib/relation-meanings.mjs'
import { buildResearchWorkspaceHref } from '../../lib/workspace-routes'
import { useConversationStore } from '../../stores/conversation-store'
import { useProjectStore } from '../../stores/project-store'
import { PageHeader } from '../shared/PageHeader'
import { EmptyState } from '../shared/EmptyState'
import { StatusBadge } from '../shared/StatusBadge'
import { GraphFilters } from './GraphFilters'
import { GraphInterpretationPanel } from './GraphInterpretationPanel'
import { PaperGraphCanvas } from './PaperGraphCanvas'

function summarizeNode(node, edges) {
  return edges.filter((edge) => edge.source === node.id || edge.target === node.id)
}

function resolveYearBounds(nodes) {
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

function isFullYearRange(yearRange, yearBounds) {
  if (!yearRange || yearBounds.min === null || yearBounds.max === null) return true
  return yearRange[0] === yearBounds.min && yearRange[1] === yearBounds.max
}

function hasGraphFilters(filters, relationTypes, yearBounds) {
  const hasQuery = Boolean(String(filters.query || '').trim())
  const hasYearFilter = !isFullYearRange(filters.yearRange, yearBounds)
  const hasRelationFilter = relationTypes.length > 0 && filters.selectedRelations.length > 0 && filters.selectedRelations.length < relationTypes.length
  return hasQuery || hasYearFilter || hasRelationFilter
}

export function PaperGraphShell({
  mode = 'global',
  projectId = '',
  showPageHeader = true,
  layoutHeight = 'calc(100vh - 64px)',
}) {
  const router = useRouter()
  const conversations = useConversationStore()
  const projectStore = useProjectStore()
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  const [graph, setGraph] = useState({ nodes: [], edges: [], meta: { relation_counts: {} } })
  const [activeNodeId, setActiveNodeId] = useState('')
  const [filters, setFilters] = useState({
    query: '',
    yearRange: null,
    selectedRelations: [],
  })
  const abortRef = useRef(null)
  const graphActionsRef = useRef(null)

  useEffect(() => {
    if (mode === 'project' && projectId && projectStore.activeProjectId !== projectId) {
      projectStore.setActiveProjectId(projectId)
    }
  }, [mode, projectId, projectStore])

  useEffect(() => {
    let active = true
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setBusy(true)
    setError('')
    getPaperGraph({
      projectId: mode === 'project' ? projectId : '',
      limit: mode === 'project' ? 120 : 220,
      signal: controller.signal,
    })
      .then((payload) => {
        if (!active) return
        setGraph(payload)
        setActiveNodeId(payload.nodes?.[0]?.id || '')
      })
      .catch((reason) => {
        if (!active || controller.signal.aborted) return
        setGraph({ nodes: [], edges: [], meta: { relation_counts: {} } })
        setError(reason instanceof Error ? reason.message : '论文关系图加载失败')
      })
      .finally(() => {
        if (active) setBusy(false)
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [mode, projectId])

  const relationTypes = useMemo(
    () => Object.keys(graph.meta?.relation_counts || {}).sort(),
    [graph.meta],
  )

  const yearBounds = useMemo(() => resolveYearBounds(graph.nodes), [graph.nodes])

  useEffect(() => {
    setFilters((current) => {
      const nextYearRange = current.yearRange && yearBounds.min !== null && yearBounds.max !== null
        ? [
          Math.max(yearBounds.min, Math.min(current.yearRange[0], yearBounds.max)),
          Math.max(yearBounds.min, Math.min(current.yearRange[1], yearBounds.max)),
        ]
        : yearBounds.min !== null && yearBounds.max !== null
          ? [yearBounds.min, yearBounds.max]
          : null
      const nextRelations = current.selectedRelations.length
        ? current.selectedRelations.filter((relation) => relationTypes.includes(relation))
        : relationTypes
      return {
        ...current,
        yearRange: nextYearRange,
        selectedRelations: nextRelations,
      }
    })
  }, [relationTypes, yearBounds.max, yearBounds.min])

  const visibleNodes = useMemo(() => {
    const keyword = filters.query.trim().toLowerCase()
    const [yearMin, yearMax] = filters.yearRange || [yearBounds.min, yearBounds.max]
    const hasYearFilter = yearBounds.min !== null && yearBounds.max !== null
    const fullYearRange = !hasYearFilter || (yearMin === yearBounds.min && yearMax === yearBounds.max)
    return graph.nodes.filter((node) => {
      const title = String(node.title || '').toLowerCase()
      const paperId = String(node.paper_id || '').toLowerCase()
      const matchesKeyword = !keyword || title.includes(keyword) || paperId.includes(keyword)
      if (!matchesKeyword) return false
      if (!hasYearFilter) return true
      if (node.year == null) return fullYearRange
      return node.year >= yearMin && node.year <= yearMax
    })
  }, [filters.query, filters.yearRange, graph.nodes, yearBounds.max, yearBounds.min])

  const visibleNodeIds = useMemo(
    () => new Set(visibleNodes.map((node) => node.id)),
    [visibleNodes],
  )

  const visibleEdges = useMemo(
    () => {
      const allowedRelations = filters.selectedRelations.length ? new Set(filters.selectedRelations) : null
      return graph.edges.filter((edge) => (
        visibleNodeIds.has(edge.source)
        && visibleNodeIds.has(edge.target)
        && (!allowedRelations || allowedRelations.has(edge.type))
      ))
    },
    [filters.selectedRelations, graph.edges, visibleNodeIds],
  )

  const activeNode = useMemo(
    () => visibleNodes.find((node) => node.id === activeNodeId) || visibleNodes[0] || null,
    [activeNodeId, visibleNodes],
  )

  const activeEdges = useMemo(
    () => (activeNode ? summarizeNode(activeNode, visibleEdges) : []),
    [activeNode, visibleEdges],
  )

  const interpretation = useMemo(() => buildGraphInterpretation(graph), [graph])
  const relationCounts = graph.meta?.relation_counts || {}
  const scopedPaperCount = graph.meta?.scoped_paper_count || 0
  const projectName = projectStore.projects.find((item) => item.project_id === projectId)?.name || projectId
  const title = mode === 'project' ? '项目论文图' : '全局论文图'
  const description = mode === 'project'
    ? `当前只展示项目 ${projectName || projectId} 绑定论文形成的子图。主节点是论文，边来自共享方法、任务、数据集或刊物。`
    : '这里展示数据库中的论文关系网络。主节点是论文，边来自共享方法、任务、数据集或刊物。'
  const relationSummary = useMemo(() => {
    if (!activeEdges.length) return []
    const counts = activeEdges.reduce((acc, edge) => {
      acc[edge.type] = (acc[edge.type] || 0) + 1
      return acc
    }, {})
    return Object.entries(counts)
      .sort((left, right) => right[1] - left[1])
      .map(([relation, count]) => `${formatRelationLabel(relation)} ${count}`)
  }, [activeEdges])
  const hasFilteredGraph = useMemo(
    () => hasGraphFilters(filters, relationTypes, yearBounds),
    [filters, relationTypes, yearBounds],
  )
  const hasLowSignalGraph = !busy && visibleNodes.length > 0 && !visibleEdges.length
  const graphStatus = useMemo(() => {
    if (error) return { tone: 'danger', label: '加载失败' }
    if (!busy && !visibleNodes.length) {
      if (hasFilteredGraph) return { tone: 'warning', label: '筛选后为空' }
      if (mode === 'project' && !scopedPaperCount) return { tone: 'warning', label: '项目待补论文' }
      return { tone: 'warning', label: '低信号图谱' }
    }
    if (hasLowSignalGraph) {
      return { tone: 'warning', label: hasFilteredGraph ? '筛选后低信号' : '低信号图谱' }
    }
    return { tone: 'success', label: '图谱可用' }
  }, [busy, error, hasFilteredGraph, hasLowSignalGraph, mode, scopedPaperCount, visibleNodes.length])
  const emptyStateConfig = useMemo(() => {
    if (error) {
      return {
        title: '论文关系图加载失败',
        description: error,
        kind: 'error',
      }
    }
    if (hasFilteredGraph) {
      return {
        title: '当前筛选条件下没有可展示的节点',
        description: '这通常是筛选条件过严，而不是图谱损坏。先重置筛选，再判断当前范围是否真的没有结构关系。',
        kind: 'filtered',
      }
    }
    if (mode === 'project' && !scopedPaperCount) {
      return {
        title: '当前项目还没有进入图谱的论文',
        description: '先把论文绑定到项目，项目图才会出现真正的关系和证据空白。',
        kind: 'project-empty',
      }
    }
    if (scopedPaperCount > 0) {
      return {
        title: '当前范围还没有形成高信号关系图',
        description: '这通常表示论文已进入范围，但还没有抽取到稳定的共享方法、任务或数据集关系。不是坏数据，也不是完整图。',
        kind: 'low-signal-empty',
      }
    }
    return {
      title: '当前范围没有可展示的论文关系',
      description: '可以先上传论文、补项目范围，或稍后再看结构化关系写入结果。',
      kind: 'global-empty',
    }
  }, [error, hasFilteredGraph, mode, scopedPaperCount])
  const lowSignalConfig = useMemo(() => {
    if (!hasLowSignalGraph) return null
    if (hasFilteredGraph) {
      return {
        title: '当前筛选后只剩论文节点',
        description: '这通常是筛选条件过滤掉了关系边，不是图谱坏了。先重置筛选，再判断当前图的真实稀疏度。',
      }
    }
    return {
      title: '当前图里只有论文节点，还没有高置信关系边',
      description: '这通常表示已有论文范围，但结构化抽取还没产出稳定关系。不是坏数据，也不是完整图。',
    }
  }, [hasFilteredGraph, hasLowSignalGraph])

  useEffect(() => {
    if (!activeNode && activeNodeId) {
      setActiveNodeId('')
      return
    }
    if (activeNode && activeNode.id !== activeNodeId) {
      setActiveNodeId(activeNode.id)
    }
  }, [activeNode, activeNodeId])

  const resetFilters = () => {
    setFilters({
      query: '',
      yearRange: yearBounds.min !== null && yearBounds.max !== null ? [yearBounds.min, yearBounds.max] : null,
      selectedRelations: relationTypes,
    })
    graphActionsRef.current?.fitView?.()
  }

  const ensureConversation = async (title = `图谱问答 · ${activeNode?.title || activeNode?.paper_id || '论文节点'}`) => {
    if (conversations.activeId) return conversations.activeId
    const created = await conversations.createNew(title, { returnConversation: true })
    return typeof created === 'string' ? created : created?.id || ''
  }

  return (
    <div style={{ height: layoutHeight, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {showPageHeader ? (
        <PageHeader
          eyebrow={mode === 'project' ? 'Project Graph' : 'Global Graph'}
          variant="workspace"
          title={title}
          description={description}
          status={(
            <>
              <StatusBadge tone={graphStatus.tone}>{graphStatus.label}</StatusBadge>
              <span className="sa-chip">节点 {visibleNodes.length}</span>
              <span className="sa-chip">边 {visibleEdges.length}</span>
            </>
          )}
          actions={(
            <>
              {mode === 'project' ? (
                <Link href="/projects" className="sa-button sa-button--ghost">返回项目</Link>
              ) : null}
              <Link href="/papers" className="sa-button sa-button--ghost">查看论文库</Link>
            </>
          )}
        />
      ) : null}

      <main className="sa-workbench" style={{ gridTemplateColumns: 'minmax(260px, 320px) minmax(0, 1.7fr) minmax(320px, 0.9fr)', display: 'grid', gap: 16 }}>
        <GraphInterpretationPanel interpretation={interpretation} />

        <GraphFilters
          query={filters.query}
          onQueryChange={(query) => setFilters((current) => ({ ...current, query }))}
          yearBounds={yearBounds}
          yearRange={filters.yearRange}
          onYearRangeChange={(yearRange) => setFilters((current) => ({ ...current, yearRange }))}
          relationTypes={relationTypes}
          selectedRelations={filters.selectedRelations}
          onToggleRelation={(relation) => {
            setFilters((current) => {
              const selectedRelations = current.selectedRelations.includes(relation)
                ? current.selectedRelations.filter((item) => item !== relation)
                : [...current.selectedRelations, relation].sort()
              return { ...current, selectedRelations }
            })
          }}
          onReset={resetFilters}
          onFitView={() => graphActionsRef.current?.fitView?.()}
          onFocusSelected={() => activeNode && graphActionsRef.current?.focusNode?.(activeNode.id)}
          activeNodeTitle={activeNode?.title || ''}
        />

        <section style={{ display: 'grid', gap: 12 }}>
          <div className="sa-section-card" style={{ padding: 16, gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span className="sa-chip">节点 {visibleNodes.length}/{graph.meta?.node_count || 0}</span>
                <span className="sa-chip">边 {visibleEdges.length}/{graph.meta?.edge_count || 0}</span>
                <span className="sa-chip">范围论文 {scopedPaperCount}</span>
                {Object.entries(relationCounts).map(([relation, count]) => (
                  <span key={relation} className="sa-chip">{formatRelationLabel(relation)} {count}</span>
                ))}
              </div>
              <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                缩放、拖拽和平移都已启用；点击节点可在右侧查看完整关系。
              </span>
            </div>
          </div>

          {lowSignalConfig ? (
            <div
              className="sa-section-card"
              style={{
                padding: 16,
                gap: 10,
                background: 'linear-gradient(145deg, rgba(255,243,205,0.82), rgba(255,255,255,0.94))',
              }}
            >
              <div style={{ display: 'grid', gap: 6 }}>
                <strong style={{ fontSize: 14 }}>{lowSignalConfig.title}</strong>
                <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                  {lowSignalConfig.description}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {hasFilteredGraph ? (
                  <button type="button" className="sa-button sa-button--ghost" onClick={resetFilters}>
                    重置筛选
                  </button>
                ) : null}
                {mode === 'project' && projectId ? (
                  <Link href={`/projects/${encodeURIComponent(projectId)}`} className="sa-button sa-button--ghost">
                    回项目工作台
                  </Link>
                ) : null}
                <Link href="/daily-brief" className="sa-button sa-button--ghost">去每日速览</Link>
                <Link href="/research" className="sa-button sa-button--ghost">进入外部研究</Link>
              </div>
            </div>
          ) : null}

          {!busy && !visibleNodes.length ? (
            <EmptyState
              title={emptyStateConfig.title}
              description={emptyStateConfig.description}
              action={(
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                  {emptyStateConfig.kind === 'filtered' ? (
                    <button type="button" className="sa-button sa-button--ghost" onClick={resetFilters}>
                      重置筛选
                    </button>
                  ) : null}
                  {emptyStateConfig.kind === 'project-empty' ? (
                    <>
                      <Link href={`/projects/${encodeURIComponent(projectId)}`} className="sa-button sa-button--ghost">
                        去绑定论文
                      </Link>
                      <Link href={`/papers?projectId=${encodeURIComponent(projectId)}`} className="sa-button sa-button--ghost">
                        打开论文库
                      </Link>
                    </>
                  ) : null}
                  {emptyStateConfig.kind === 'low-signal-empty' ? (
                    <>
                      {mode === 'project' && projectId ? (
                        <Link href={`/projects/${encodeURIComponent(projectId)}`} className="sa-button sa-button--ghost">
                          回项目工作台
                        </Link>
                      ) : null}
                      <Link href="/daily-brief" className="sa-button sa-button--ghost">去每日速览</Link>
                      <Link href="/research" className="sa-button sa-button--ghost">进入外部研究</Link>
                    </>
                  ) : null}
                  {emptyStateConfig.kind === 'global-empty' || emptyStateConfig.kind === 'error' ? (
                    <Link href="/papers" className="sa-button sa-button--ghost">查看论文库</Link>
                  ) : null}
                </div>
              )}
            />
          ) : (
            <PaperGraphCanvas
              nodes={visibleNodes}
              edges={visibleEdges}
              activeNodeId={activeNode?.id || ''}
              onSelect={(node) => setActiveNodeId(node.id)}
              onReady={(actions) => {
                graphActionsRef.current = actions
              }}
            />
          )}
        </section>

        <aside className="sa-section-card" style={{ padding: 16, gap: 14, alignContent: 'start' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 18 }}>{activeNode?.title || '选择论文节点'}</h3>
            <p style={{ margin: '8px 0 0', color: 'var(--sa-text-soft)', fontSize: 13, lineHeight: 1.7 }}>
              {activeNode ? '优先查看论文标题、年份和图中关系；技术标识保留在操作链接里。' : '点击左侧节点后，这里会显示论文元信息和图中关系。'}
            </p>
          </div>

          {activeNode ? (
            <>
              <div style={{ display: 'grid', gap: 8 }}>
                <span className="sa-chip">年份 {activeNode.year || '年份未知'}</span>
                <span className="sa-chip">关联边 {activeEdges.length ? activeEdges.length : '尚未建立关联'}</span>
                {activeNode.library_id && activeNode.library_id !== 'default' ? (
                  <span className="sa-chip">来源库 {activeNode.library_id}</span>
                ) : null}
                {relationSummary.length ? (
                  <span className="sa-chip">关系概况 {relationSummary.slice(0, 2).join(' · ')}</span>
                ) : null}
              </div>

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <Link href={`/papers/${encodeURIComponent(activeNode.paper_id)}`} className="sa-button sa-button--ghost">
                  查看论文
                </Link>
                <Button
                  size="sm"
                  color="primary"
                  variant="flat"
                  onPress={async () => {
                    const conversationId = await ensureConversation(`图谱问答 · ${activeNode.title || activeNode.paper_id}`)
                    if (!conversationId) return
                    router.push(`/chat/${encodeURIComponent(conversationId)}?paperId=${encodeURIComponent(activeNode.paper_id)}`)
                  }}
                >
                  直接提问
                </Button>
                <Button
                  size="sm"
                  color="secondary"
                  variant="flat"
                  onPress={async () => {
                    const researchPrompt = `请围绕论文《${activeNode.title || activeNode.paper_id}》开展深度研究，结合图谱关系说明它的关联论文、潜在研究脉络和后续值得追踪的问题。`
                    router.push(buildResearchWorkspaceHref({
                      prompt: researchPrompt,
                      paperId: activeNode.paper_id,
                      projectId: mode === 'project' ? projectId : '',
                      source: 'graph',
                      focus: 'external',
                    }))
                  }}
                >
                  送入外部研究
                </Button>
                <button
                  type="button"
                  className="sa-button sa-button--ghost"
                  onClick={() => graphActionsRef.current?.focusNode?.(activeNode.id)}
                >
                  图中聚焦
                </button>
                {mode === 'global' && projectStore.activeProjectId ? (
                  <Button
                    size="sm"
                    color="primary"
                    variant="flat"
                    onPress={async () => {
                      await projectStore.addPaper(activeNode.paper_id)
                    }}
                  >
                    加入当前项目
                  </Button>
                ) : null}
              </div>

              <section style={{ display: 'grid', gap: 10 }}>
                <strong style={{ fontSize: 14 }}>图中关系</strong>
                {!activeEdges.length ? (
                  <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>当前节点尚未建立可展示的论文关系。</p>
                ) : (
                  activeEdges.map((edge) => {
                    const peerId = edge.source === activeNode.id ? edge.target : edge.source
                    const peerNode = graph.nodes.find((node) => node.id === peerId)
                    return (
                      <article
                        key={`${edge.source}-${edge.target}-${edge.type}`}
                        style={{
                          padding: 12,
                          borderRadius: 14,
                          border: '1px solid var(--sa-border)',
                          background: 'rgba(255,255,255,0.78)',
                          display: 'grid',
                          gap: 6,
                        }}
                      >
                        <strong style={{ fontSize: 13 }}>{peerNode?.title || peerNode?.paper_id || peerId}</strong>
                        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                          {formatRelationLabel(edge.type)} · {formatRelationMeaning(edge.type)} · 权重 {edge.weight}
                        </span>
                        {edge.shared_items?.length ? (
                          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                            共享依据：{edge.shared_items.slice(0, 3).join('，')}
                          </span>
                        ) : null}
                      </article>
                    )
                  })
                )}
              </section>
            </>
          ) : (
            <EmptyState title="暂无节点详情" description="左侧节点加载完成后，选择一个节点查看详情。" />
          )}
        </aside>
      </main>
    </div>
  )
}
