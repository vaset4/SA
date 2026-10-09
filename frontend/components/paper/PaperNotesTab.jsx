'use client'

import { Button } from '@heroui/react'
import { EmptyState } from '../shared/EmptyState'
import { formatNoteSource, formatNoteType } from '../../lib/notes'

export function PaperNotesTab({ notesState }) {
  const {
    notes,
    noteDraft,
    setNoteDraft,
    busy,
    error,
    save,
  } = notesState

  return (
    <section style={{ border: '1px solid var(--sa-border)', borderRadius: 12, padding: 16 }}>
      <h3 style={{ margin: 0, fontSize: 15 }}>研究笔记</h3>

      <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
        <textarea
          value={noteDraft}
          onChange={(event) => setNoteDraft(event.target.value)}
          placeholder="记录你对本文的贡献、方法、实验和复现风险..."
          style={{
            minHeight: 96,
            border: '1px solid var(--sa-border)',
            borderRadius: 10,
            padding: 10,
            background: 'var(--sa-bg)',
            color: 'var(--sa-text)',
            resize: 'vertical',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button size="sm" color="primary" onPress={save} isDisabled={!noteDraft.trim() || busy}>
            {busy ? '保存中' : '保存笔记'}
          </Button>
        </div>
      </div>

      {error ? <p style={{ margin: '10px 0 0', color: 'var(--sa-danger)', fontSize: 12 }}>{error}</p> : null}

      <div style={{ marginTop: 14 }}>
        {notes.length === 0 ? (
          <EmptyState title="还没有笔记" description="先写下第一条研究记录，后续可持续追加。" />
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {notes.map((item, index) => (
              <article
                key={item.note_id || index}
                style={{
                  border: '1px solid var(--sa-border)',
                  borderRadius: 10,
                  background: 'var(--sa-surface-soft)',
                  padding: 10,
                }}
              >
                <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6 }}>{item.content || ''}</p>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                  <span className="sa-chip">{formatNoteType(item.note_type)}</span>
                  <span className="sa-chip">来源 {formatNoteSource(item.source_type)}</span>
                </div>
                {item.excerpt ? (
                  <p style={{ margin: '8px 0 0', fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                    摘录：{item.excerpt}
                  </p>
                ) : null}
                <p style={{ margin: '8px 0 0', color: 'var(--sa-text-faint)', fontSize: 11 }}>
                  {item.created_at || item.updated_at || '刚刚'}
                </p>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
