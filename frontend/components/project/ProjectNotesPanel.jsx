'use client'

import { useEffect, useMemo, useState } from 'react'

import { EmptyState } from '../shared/EmptyState'
import { NoteCaptureCard } from '../notes/NoteCaptureCard'
import { listNotes } from '../../lib/api/papers'
import { formatNoteResourceLabels } from '../../lib/display-identifiers.mjs'
import { NOTE_SOURCE_OPTIONS, NOTE_TYPE_OPTIONS, formatNoteSource, formatNoteType } from '../../lib/notes'
import { getError } from '../../lib/study-assistant-workspace.mjs'

const fieldStyle = {
  border: '1px solid var(--sa-border)',
  borderRadius: 10,
  padding: '8px 10px',
  background: 'rgba(255,255,255,0.88)',
  color: 'var(--sa-text)',
}

function formatTime(value) {
  if (!value) return '刚刚'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚'
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

export function ProjectNotesPanel({ projectStore }) {
  const projectId = projectStore.activeProjectId
  const [state, setState] = useState({
    loading: true,
    error: '',
    notes: [],
  })
  const [noteTypeFilter, setNoteTypeFilter] = useState('')
  const [sourceTypeFilter, setSourceTypeFilter] = useState('')

  const refresh = async () => {
    if (!projectId) {
      setState({ loading: false, error: '', notes: [] })
      return
    }
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const data = await listNotes({
        projectId,
        noteType: noteTypeFilter,
        sourceType: sourceTypeFilter,
        includeProjectPapers: true,
        includeProjectSessions: true,
        limit: 100,
      })
      setState({ loading: false, error: '', notes: data?.notes ?? [] })
    } catch (reason) {
      setState({ loading: false, error: getError(reason, '项目笔记加载失败'), notes: [] })
    }
  }

  useEffect(() => {
    refresh().catch(() => {})
  }, [projectId, noteTypeFilter, sourceTypeFilter])

  const headerText = useMemo(() => {
    const projectName = projectStore.activeProject?.name || projectId || '当前项目'
    return `把项目里的手工记录、内部研究、外部研究和图谱观察收束到 ${projectName}。`
  }, [projectId, projectStore.activeProject?.name])

  if (!projectId) {
    return <EmptyState title="请先选择项目" description="选择项目后，才能按研究空间聚合笔记。" />
  }

  return (
    <section className="sa-panel-card sa-section-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <p className="sa-section-card__eyebrow">Project Notes</p>
          <h3 className="sa-section-card__title">项目研究笔记</h3>
          <p className="sa-section-card__description">{headerText}</p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span className="sa-chip">当前项目 {projectStore.activeProject?.name || projectId}</span>
          <span className="sa-chip">笔记 {state.notes.length}</span>
        </div>
      </div>

      <NoteCaptureCard
        title="项目工作台手记"
        description="适合记录阶段性结论、后续问题和可执行想法。保存后会自动出现在当前项目的研究笔记列表。"
        placeholder="记录当前项目的研究判断、待验证问题或下一步实验想法"
        notePayload={{ projectId }}
        defaultNoteType="idea_note"
        sourceType="manual"
        submitLabel="保存到项目笔记"
        resetKey={projectId}
        onSaved={refresh}
      />

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <label style={{ display: 'grid', gap: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>类型筛选</span>
          <select value={noteTypeFilter} onChange={(event) => setNoteTypeFilter(event.target.value)} style={fieldStyle}>
            <option value="">全部类型</option>
            {NOTE_TYPE_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>
        <label style={{ display: 'grid', gap: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>来源筛选</span>
          <select value={sourceTypeFilter} onChange={(event) => setSourceTypeFilter(event.target.value)} style={fieldStyle}>
            <option value="">全部来源</option>
            {NOTE_SOURCE_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>
      </div>

      {state.error ? <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{state.error}</p> : null}
      {state.loading ? <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>项目笔记加载中...</p> : null}

      {!state.loading && !state.notes.length ? (
        <EmptyState
          title="当前项目还没有研究笔记"
          description="你可以先在这里手工记录，也可以从内部研究、外部研究或图谱观察里直接保存。"
          action={<button type="button" className="sa-button sa-button--ghost" onClick={refresh}>重新加载笔记</button>}
        />
      ) : null}

      {!state.loading && state.notes.length ? (
        <div style={{ display: 'grid', gap: 10 }}>
          {state.notes.map((note) => (
            <article
              key={note.note_id}
              className="sa-list-card"
            >
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span className="sa-chip">{formatNoteType(note.note_type)}</span>
                <span className="sa-chip">来源 {formatNoteSource(note.source_type)}</span>
                {formatNoteResourceLabels(note).map((label) => <span key={label} className="sa-chip">{label}</span>)}
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{note.content || ''}</div>
              {note.excerpt ? (
                <div style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                  摘录：{note.excerpt}
                </div>
              ) : null}
              {note.citation ? (
                <div style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                  引用：{note.citation}
                </div>
              ) : null}
              <div style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>
                {formatTime(note.updated_at || note.created_at)}
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
