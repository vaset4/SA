'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'

const MAX_VISIBLE_PAPERS = 20

function paperTitle(paper) {
  return paper?.title || paper?.paper_id || '未命名论文'
}

function paperSubtitle(paper) {
  const authors = Array.isArray(paper?.authors) ? paper.authors.slice(0, 2).join(', ') : ''
  return [paper?.year, authors].filter(Boolean).join(' · ') || ''
}

/**
 * 轻量就绪态：供右栏展示「能不能引用」，不暴露工程枚举原文。
 * @param {object} paper
 * @returns {{ label: string, tone: 'ok' | 'warn' | 'mute' }}
 */
function paperReadiness(paper) {
  const raw = `${paper?.status || paper?.readiness || paper?.index_status || paper?.pipeline_status || ''}`.toLowerCase()
  if (!raw) {
    return { label: '未知', tone: 'mute' }
  }
  if (/(ready|success|done|complete|indexed|available|ok)/.test(raw)) {
    return { label: '可引用', tone: 'ok' }
  }
  if (/(fail|error|broken)/.test(raw)) {
    return { label: '失败', tone: 'warn' }
  }
  if (/(pend|run|process|analy|queue|wait|progress)/.test(raw)) {
    return { label: '分析中', tone: 'warn' }
  }
  return { label: '处理中', tone: 'mute' }
}

function filterPapers(papers, query) {
  const keyword = query.trim().toLowerCase()
  const items = keyword
    ? papers.filter((paper) => {
      const title = paperTitle(paper).toLowerCase()
      const authors = Array.isArray(paper?.authors) ? paper.authors.join(' ').toLowerCase() : ''
      return title.includes(keyword) || authors.includes(keyword)
    })
    : papers
  return items.slice(0, MAX_VISIBLE_PAPERS)
}

function PaperRow({ paper, active, onAddPaper, onRemovePaper }) {
  const paperId = paper?.paper_id || ''
  if (!paperId) return null
  const readiness = paperReadiness(paper)
  const subtitle = paperSubtitle(paper)
  return (
    <article className="sa-chat-rail__paper">
      <div className="sa-chat-rail__paper-main">
        <Link href={`/papers/${encodeURIComponent(paperId)}`} className="sa-chat-rail__paper-title">
          {paperTitle(paper)}
        </Link>
        {subtitle ? <span>{subtitle}</span> : null}
        <span className={`sa-chat-rail__ready sa-chat-rail__ready--${readiness.tone}`}>
          {readiness.label}
        </span>
      </div>
      <button
        type="button"
        className={`sa-button ${active ? 'sa-button--ghost' : 'sa-button--primary'}`}
        onClick={() => (active ? onRemovePaper(paperId) : onAddPaper(paperId))}
      >
        {active ? '移除' : '加入'}
      </button>
    </article>
  )
}

/**
 * 聊天右栏：只放知识库论文（筛选 / 就绪 / 加入会话）。
 * 上下文散文卡与会话产物列表已移除，避免与答案主区抢视线。
 */
export function ChatResearchRail({
  library,
  activePaperIds = [],
  onAddPaper,
  onRemovePaper,
}) {
  const [query, setQuery] = useState('')
  const activeSet = useMemo(() => new Set(activePaperIds), [activePaperIds])
  const papers = Array.isArray(library.papers) ? library.papers : []
  const visiblePapers = useMemo(() => filterPapers(papers, query), [papers, query])
  const activeCount = activePaperIds.length

  return (
    <div className="sa-chat-rail sa-chat-rail--papers-only">
      <div className="sa-chat-rail__head">
        <h2 className="sa-section-card__title">知识库论文</h2>
        <p className="sa-chat-rail__meta">
          {papers.length} 篇
          {activeCount ? ` · 本会话 ${activeCount}` : ''}
        </p>
      </div>

      <input
        className="sa-input sa-chat-rail__search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="筛选论文"
        aria-label="筛选论文"
      />

      {library.loading ? <p className="sa-note">加载中…</p> : null}
      {!library.loading && !visiblePapers.length ? (
        <p className="sa-note">没有匹配的论文。</p>
      ) : null}

      <div className="sa-chat-rail__paper-list">
        {visiblePapers.map((paper) => (
          <PaperRow
            key={paper.paper_id}
            paper={paper}
            active={activeSet.has(paper.paper_id)}
            onAddPaper={onAddPaper}
            onRemovePaper={onRemovePaper}
          />
        ))}
      </div>

      <Link href="/papers" className="sa-chat-rail__library-link">
        打开论文库
      </Link>
    </div>
  )
}
