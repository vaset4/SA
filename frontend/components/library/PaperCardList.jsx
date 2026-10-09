'use client'

import Link from 'next/link'
import { Button } from '@heroui/react'
import { EmptyState } from '../shared/EmptyState'
import { StatusBadge } from '../shared/StatusBadge'

function sortPapers(items, sortBy) {
  const cloned = [...items]
  if (sortBy === 'year-desc') {
    return cloned.sort((a, b) => Number(b?.year || 0) - Number(a?.year || 0))
  }
  if (sortBy === 'year-asc') {
    return cloned.sort((a, b) => Number(a?.year || 0) - Number(b?.year || 0))
  }
  return cloned.sort((a, b) => (a?.title || '').localeCompare(b?.title || ''))
}

function authorSummary(paper) {
  return Array.isArray(paper?.authors) && paper.authors.length
    ? paper.authors.slice(0, 4).join(', ')
    : '未知作者'
}

function keywordSummary(paper) {
  const values = paper?.analysis?.keywords || paper?.keywords || []
  return Array.isArray(values) ? values.filter(Boolean).slice(0, 4) : []
}

function summaryText(paper) {
  return paper?.analysis?.tldr || paper?.analysis?.problem || ''
}

export function PaperCardList({
  papers,
  loading,
  query,
  setQuery,
  sortBy,
  setSortBy,
  viewMode = 'card',
  setViewMode,
  onAddToConversation,
  onAddToProject,
  onOpenGraph,
  onDelete,
}) {
  const filtered = sortPapers(
    papers.filter((paper) => {
      const keyword = query.trim().toLowerCase()
      if (!keyword) return true
      const title = String(paper?.title || '').toLowerCase()
      const authors = Array.isArray(paper?.authors) ? paper.authors.join(' ').toLowerCase() : ''
      return title.includes(keyword) || authors.includes(keyword)
    }),
    sortBy,
  )

  return (
    <section className="sa-section-card sa-library-list">
      <div className="sa-library-list__toolbar">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="按标题或作者筛选论文"
          style={{
            flex: 1,
            minWidth: 220,
            height: 38,
            border: '1px solid var(--sa-border)',
            borderRadius: 12,
            padding: '0 12px',
            background: 'rgba(255,255,255,0.92)',
            color: 'var(--sa-text)',
          }}
        />
        <select
          value={sortBy}
          onChange={(event) => setSortBy(event.target.value)}
          style={{
            height: 38,
            border: '1px solid var(--sa-border)',
            borderRadius: 12,
            padding: '0 12px',
            background: 'rgba(255,255,255,0.92)',
            color: 'var(--sa-text)',
          }}
        >
          <option value="title">按标题</option>
          <option value="year-desc">按年份降序</option>
          <option value="year-asc">按年份升序</option>
        </select>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <Button size="sm" variant={viewMode === 'card' ? 'flat' : 'light'} onPress={() => setViewMode?.('card')}>
            卡片视图
          </Button>
          <Button size="sm" variant={viewMode === 'list' ? 'flat' : 'light'} onPress={() => setViewMode?.('list')}>
            列表视图
          </Button>
        </div>
        <StatusBadge tone="neutral">共 {filtered.length} 篇</StatusBadge>
      </div>

      <div className="sa-library-list__items">
        {loading ? (
          <p style={{ margin: 0, color: 'var(--sa-text-soft)', fontSize: 13 }}>论文列表加载中...</p>
        ) : filtered.length === 0 ? (
          <EmptyState
            title="当前筛选条件下没有论文"
            description="可以调整筛选关键词，或先上传新的论文文档。"
            action={query ? (
              <button type="button" className="sa-button sa-button--ghost" onClick={() => setQuery('')}>
                清空筛选
              </button>
            ) : null}
          />
        ) : (
          filtered.map((paper) => {
            const paperId = paper?.paper_id || ''
            const keywords = keywordSummary(paper)
            const summary = summaryText(paper)
            const actions = (
              <div style={{ display: 'flex', gap: 8, marginTop: viewMode === 'card' ? 12 : 0, flexWrap: 'wrap' }}>
                <Link href={`/papers/${paperId}`} prefetch={false}>
                  <Button size="sm" variant="flat">查看详情</Button>
                </Link>
                <Button size="sm" color="primary" variant="flat" onPress={() => onAddToConversation(paperId)}>
                  添加到会话
                </Button>
                {onAddToProject ? (
                  <Button size="sm" color="secondary" variant="flat" onPress={() => onAddToProject(paperId)}>
                    添加到项目
                  </Button>
                ) : null}
                {onOpenGraph ? (
                  <Button size="sm" color="warning" variant="flat" onPress={() => onOpenGraph(paperId)}>
                    查看图谱
                  </Button>
                ) : null}
                <Button size="sm" color="danger" variant="light" onPress={() => onDelete(paperId)}>
                  删除
                </Button>
              </div>
            )

            if (viewMode === 'list') {
              return (
                <article
                  key={paperId}
                  className="sa-list-card"
                  style={{
                    gap: 12,
                    gridTemplateColumns: 'minmax(0, 1.2fr) 120px minmax(260px, 0.9fr)',
                    alignItems: 'center',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <h4 style={{ margin: 0, fontSize: 15, lineHeight: 1.4 }}>{paper?.title || paperId}</h4>
                    <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)', fontSize: 12 }}>
                      {authorSummary(paper)}
                    </p>
                    {summary ? (
                      <p style={{ margin: '8px 0 0', color: 'var(--sa-text-soft)', fontSize: 12, lineHeight: 1.7 }}>
                        {summary}
                      </p>
                    ) : null}
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                      {paper?.project_role ? <span className="sa-chip">{paper.project_role}</span> : null}
                      {keywords.map((item) => <span key={`${paperId}-${item}`} className="sa-chip">{item}</span>)}
                    </div>
                  </div>
                  <div style={{ display: 'grid', gap: 6, justifyItems: 'start' }}>
                    <StatusBadge tone="neutral">{paper?.year || 'N/A'}</StatusBadge>
                    <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{paperId}</span>
                  </div>
                  {actions}
                </article>
              )
            }

            return (
              <article key={paperId} className="sa-list-card" style={{ gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
                  <div style={{ minWidth: 0 }}>
                    <h4 style={{ margin: 0, fontSize: 15, lineHeight: 1.4 }}>{paper?.title || paperId}</h4>
                    <p style={{ margin: '6px 0 0', color: 'var(--sa-text-soft)', fontSize: 12 }}>
                      {authorSummary(paper)}
                    </p>
                    {summary ? (
                      <p style={{ margin: '8px 0 0', color: 'var(--sa-text-soft)', fontSize: 12, lineHeight: 1.7 }}>
                        {summary}
                      </p>
                    ) : null}
                  </div>
                  <StatusBadge tone="neutral">{paper?.year || 'N/A'}</StatusBadge>
                </div>

                {(paper?.project_role || keywords.length) ? (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                    {paper?.project_role ? <span className="sa-chip">{paper.project_role}</span> : null}
                    {keywords.map((item) => <span key={`${paperId}-${item}`} className="sa-chip">{item}</span>)}
                  </div>
                ) : null}

                {actions}
              </article>
            )
          })
        )}
      </div>
    </section>
  )
}
