'use client'

import Link from 'next/link'
import { NoteCaptureCard } from '../notes/NoteCaptureCard'
import { buildNotePayload, buildResearchNoteDraft } from '../../lib/research-session-view'

function ActionLinks({ model }) {
  return (
    <div className="sa-stack">
      <Link href="/chat" className="sa-button sa-button--ghost">返回会话</Link>
      <Link href="/papers" className="sa-button sa-button--ghost">补充材料</Link>
      {model.projectId ? <Link href={`/projects/${encodeURIComponent(model.projectId)}`} className="sa-button sa-button--ghost">查看项目</Link> : null}
    </div>
  )
}

function SignalList({ title, items }) {
  if (!items.length) return null
  return (
    <section className="sa-context-card">
      <strong>{title}</strong>
      {items.slice(0, 4).map((item) => <p key={item}>{item}</p>)}
    </section>
  )
}

export function ResearchSessionSidePanel({ model }) {
  const noteDraft = buildResearchNoteDraft(model)
  return (
    <aside className="sa-research-session-side">
      <ActionLinks model={model} />
      <SignalList title="关键发现" items={model.keyFindings} />
      <SignalList title="下一步问题" items={model.followUpQuestions} />
      <SignalList title="外部空白" items={model.externalGaps} />
      <NoteCaptureCard
        title="保存研究结论"
        placeholder="整理关键结论、证据判断和下一步问题"
        initialContent={noteDraft}
        resetKey={`${model.sessionId}:${model.updatedAt}`}
        notePayload={buildNotePayload(model)}
        defaultNoteType="evidence_note"
        sourceType="deep_research"
        submitLabel="保存研究笔记"
      />
    </aside>
  )
}
