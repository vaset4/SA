'use client'

import { useEffect, useState } from 'react'
import { Button } from '@heroui/react'

import { addNote } from '../../lib/api/papers'
import { NOTE_TYPE_OPTIONS, formatNoteSource } from '../../lib/notes'
import { getError } from '../../lib/study-assistant-workspace.mjs'

const fieldStyle = {
  border: '1px solid var(--sa-border)',
  borderRadius: 10,
  padding: '8px 10px',
  background: 'rgba(255,255,255,0.9)',
  color: 'var(--sa-text)',
}

export function NoteCaptureCard({
  title = '保存研究笔记',
  description = '',
  placeholder = '写下要沉淀的研究结论、证据或问题',
  initialContent = '',
  resetKey = '',
  notePayload = {},
  defaultNoteType = 'reading_note',
  sourceType = 'manual',
  submitLabel = '保存笔记',
  successText = '已保存到研究笔记。',
  allowNoteTypeChange = true,
  onSaved = () => {},
}) {
  const [content, setContent] = useState(initialContent)
  const [noteType, setNoteType] = useState(defaultNoteType)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')

  useEffect(() => {
    setContent(initialContent)
    setNoteType(defaultNoteType)
    setStatus('')
  }, [defaultNoteType, initialContent, resetKey])

  const submit = async () => {
    const cleanContent = content.trim()
    if (!cleanContent || busy) return
    setBusy(true)
    setStatus('')
    try {
      await addNote({
        ...notePayload,
        content: cleanContent,
        noteType,
        sourceType,
      })
      setStatus(successText)
      onSaved()
    } catch (reason) {
      setStatus(getError(reason, '笔记保存失败'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section
      style={{
        border: '1px solid var(--sa-border)',
        borderRadius: 14,
        padding: 14,
        background: 'rgba(255,255,255,0.84)',
        display: 'grid',
        gap: 10,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'grid', gap: 4 }}>
          <strong style={{ fontSize: 14 }}>{title}</strong>
          {description ? (
            <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>{description}</span>
          ) : null}
        </div>
        <span className="sa-chip">来源 {formatNoteSource(sourceType)}</span>
      </div>

      {allowNoteTypeChange ? (
        <label style={{ display: 'grid', gap: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>笔记类型</span>
          <select value={noteType} onChange={(event) => setNoteType(event.target.value)} style={fieldStyle}>
            {NOTE_TYPE_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>
      ) : null}

      <textarea
        value={content}
        onChange={(event) => setContent(event.target.value)}
        placeholder={placeholder}
        style={{
          ...fieldStyle,
          minHeight: 112,
          resize: 'vertical',
        }}
      />

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button size="sm" color="primary" onPress={submit} isDisabled={!content.trim() || busy}>
          {busy ? '保存中' : submitLabel}
        </Button>
      </div>

      {status ? (
        <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>{status}</p>
      ) : null}
    </section>
  )
}
