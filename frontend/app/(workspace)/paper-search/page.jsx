'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Button } from '@heroui/react'
import { searchArxiv } from '../../../lib/api/research'
import { buildResearchWorkspaceHref } from '../../../lib/workspace-routes'
import { useProjectStore } from '../../../stores/project-store'
import { EmptyState } from '../../../components/shared/EmptyState'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'

function authorSummary(item) {
  return Array.isArray(item?.authors) && item.authors.length
    ? item.authors.slice(0, 4).join(', ')
    : '未知作者'
}

export default function PaperSearchPage() {
  const projectStore = useProjectStore()
  const [query, setQuery] = useState('')
  const [state, setState] = useState({
    loading: false,
    error: '',
    report: '',
    papers: [],
  })

  const runSearch = async () => {
    const cleanQuery = String(query || '').trim()
    if (!cleanQuery || state.loading) return
    setState((current) => ({ ...current, loading: true, error: '', report: '', papers: [] }))
    try {
      const data = await searchArxiv({
        query: cleanQuery,
        sessionId: '',
        conversationId: '',
        projectId: projectStore.activeProjectId || '',
      })
      setState({
        loading: false,
        error: '',
        report: String(data?.report || '').trim(),
        papers: Array.isArray(data?.papers) ? data.papers : [],
      })
    } catch (reason) {
      setState({
        loading: false,
        error: reason instanceof Error ? reason.message : 'arXiv 搜索失败',
        report: '',
        papers: [],
      })
    }
  }

  return (
    <div style={{ minHeight: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      <PageHeader
        eyebrow="Paper Search"
        variant="workspace"
        title="论文搜索"
        description="搜索外部候选论文，再决定是否送入研究流程或沉淀到当前项目。"
        status={(
          <>
            <StatusBadge tone={state.error ? 'danger' : state.loading ? 'warning' : 'neutral'}>
              {state.error ? '搜索失败' : state.loading ? '搜索中' : 'arXiv 已接入'}
            </StatusBadge>
            <span className="sa-chip">当前项目 {projectStore.activeProject?.name || '未绑定'}</span>
            <span className="sa-chip">结果 {state.papers.length}</span>
          </>
        )}
        actions={(
          <>
            <Link href="/chat" className="sa-button sa-button--ghost">
              回对话页
            </Link>
            <Link href="/research" className="sa-button sa-button--ghost">
              去 Deep Research
            </Link>
          </>
        )}
      />

      <main className="sa-workbench">
        <section className="sa-section-card sa-section-card--accent">
          <div className="sa-section-card__header">
            <span className="sa-section-card__eyebrow">arXiv Search</span>
            <div>
              <h3 className="sa-section-card__title">先找候选论文，再决定怎么研究</h3>
              <p className="sa-section-card__description">
                这一步先负责外部论文搜索，不直接替代内部研究。找到候选论文后，再决定是继续外部研究，还是回到对话里围绕它做研究。
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gap: 12 }}>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="例如：retrieval augmented generation survey"
              className="sa-input"
            />
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="sa-chip-list">
                <span className="sa-chip">搜索源 arXiv</span>
                {projectStore.activeProject?.name ? <span className="sa-chip">当前项目 {projectStore.activeProject.name}</span> : null}
              </div>
              <Button size="sm" color="primary" onPress={runSearch} isDisabled={!query.trim() || state.loading}>
                {state.loading ? '搜索中' : '开始搜索'}
              </Button>
            </div>
            {state.error ? <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>{state.error}</p> : null}
          </div>
        </section>

        {state.report ? (
          <section className="sa-section-card">
            <div className="sa-section-card__header">
              <span className="sa-section-card__eyebrow">Search Summary</span>
              <div>
                <h3 className="sa-section-card__title">搜索摘要</h3>
                <p className="sa-section-card__description">这里展示当前这轮 arXiv 搜索返回的摘要性结果，方便先快速判断方向。</p>
              </div>
            </div>
            <p className="sa-note" style={{ whiteSpace: 'pre-wrap' }}>{state.report}</p>
          </section>
        ) : null}

        {!state.loading && !state.error && !state.papers.length ? (
          <EmptyState
            title="还没有搜索结果"
            description="先输入主题或方法关键词，arXiv 返回结果后会在这里列出候选论文。"
          />
        ) : null}

        {state.papers.length ? (
          <section className="sa-section-card">
            <div className="sa-section-card__header">
              <span className="sa-section-card__eyebrow">Results</span>
              <div>
                <h3 className="sa-section-card__title">候选论文</h3>
                <p className="sa-section-card__description">先看题目、作者和摘要，再决定要不要把这个问题送进 Deep Research 做正式外部研究。</p>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 12 }}>
              {state.papers.map((paper, index) => {
                const paperId = paper.paper_id || paper.arxiv_id || `arxiv-${index}`
                const researchPrompt = `请围绕论文《${paper.title || paperId}》开展 Deep Research，整理它的研究问题、方法位置、代表性对比工作和后续值得继续跟踪的方向。`
                return (
                  <article key={paperId} className="sa-list-card" style={{ gap: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                      <div style={{ display: 'grid', gap: 6, minWidth: 0 }}>
                        <strong style={{ fontSize: 15, lineHeight: 1.6 }}>{paper.title || paperId}</strong>
                        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{authorSummary(paper)}</span>
                      </div>
                      <div className="sa-chip-list">
                        {paper.published_date ? <span className="sa-chip">{paper.published_date}</span> : null}
                        <span className="sa-chip">{paper.arxiv_id || paperId}</span>
                      </div>
                    </div>

                    {paper.abstract ? (
                      <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)', lineHeight: 1.75 }}>
                        {paper.abstract}
                      </p>
                    ) : null}

                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {paper.pdf_url ? (
                        <Link href={paper.pdf_url} target="_blank" className="sa-button sa-button--ghost">
                          打开 PDF
                        </Link>
                      ) : null}
                      <Link
                        href={buildResearchWorkspaceHref({
                          prompt: researchPrompt,
                          paperId: paperId,
                          projectId: projectStore.activeProjectId || '',
                          source: 'paper_search',
                        })}
                        className="sa-button sa-button--primary"
                      >
                        送入 Deep Research
                      </Link>
                    </div>
                  </article>
                )
              })}
            </div>
          </section>
        ) : null}
      </main>
    </div>
  )
}
