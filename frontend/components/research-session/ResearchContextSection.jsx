'use client'

import { MessageContent } from '../chat/MessageContent'

function PaperRow({ paper }) {
  return (
    <article className="sa-list-card">
      <strong>{paper.title || paper.paper_id}</strong>
      <span className="sa-note">{paper.year || '年份未知'} · {paper.method || '方法待补充'}</span>
      {paper.tldr ? <span className="sa-note">{paper.tldr}</span> : null}
    </article>
  )
}

function HistoryRow({ item, index }) {
  return (
    <article className="sa-list-card">
      <strong>{item.role === 'assistant' ? 'Assistant' : 'User'} {index + 1}</strong>
      <MessageContent message={{ ...item, renderMode: item.renderMode || item.render_mode }} fallback="" />
    </article>
  )
}

export function ResearchContextSection({ model }) {
  return (
    <section className="sa-detail-grid sa-detail-grid--split">
      <div className="sa-section-card">
        <h3 className="sa-section-card__title">项目 / 论文上下文</h3>
        {!model.selectedPapers.length ? <p className="sa-note">当前 session 没有绑定项目背景论文。</p> : null}
        {model.selectedPapers.map((paper) => <PaperRow key={paper.paper_id} paper={paper} />)}
      </div>
      <div className="sa-section-card">
        <h3 className="sa-section-card__title">会话历史</h3>
        {!model.chatHistory.length ? <p className="sa-note">当前 checkpoint 没有持久化历史消息。</p> : null}
        {model.chatHistory.map((item, index) => <HistoryRow key={`${item.role}-${index}`} item={item} index={index} />)}
      </div>
    </section>
  )
}
