'use client'

import { useState } from 'react'

import {
  MEMORY_TYPE_OPTIONS,
  EVIDENCE_MODE_OPTIONS,
  MATURITY_ORDER,
  formatMaturity,
} from '../../lib/research-memory.mjs'
import { createMemory } from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'

// 新建主张表单：标题 + 类型 + 成熟度 + 证据模式 + 归属问题 + 正文
export function MemoryCreateForm({ projectId, questions, onCreated }) {
  const [form, setForm] = useState({
    title: '',
    memory_type: 'decision_note',
    maturity: 'intuition',
    evidence_mode: 'user_note_only',
    question_id: '',
    body: '',
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const update = (field) => (event) => setForm((current) => ({ ...current, [field]: event.target.value }))

  const submit = async () => {
    if (!form.title.trim()) {
      setError('请先写下这条主张')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const payload = {
        title: form.title.trim(),
        memory_type: form.memory_type,
        maturity: form.maturity,
        evidence_mode: form.evidence_mode,
        body: form.body,
      }
      if (form.question_id) payload.question_id = form.question_id
      await createMemory({ projectId, payload })
      onCreated()
    } catch (reason) {
      setError(getError(reason, '主张创建失败'))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="sa-list-card" style={{ display: 'grid', gap: 10, marginTop: 12 }}>
      <input
        className="sa-input"
        placeholder="写下一条判断，如：RAGAS 是当前最可行的无参考评估框架"
        value={form.title}
        onChange={update('title')}
      />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <select className="sa-input" value={form.memory_type} onChange={update('memory_type')}>
          {MEMORY_TYPE_OPTIONS.map((item) => (
            <option key={item.value} value={item.value}>{item.label}</option>
          ))}
        </select>
        <select className="sa-input" value={form.maturity} onChange={update('maturity')}>
          {MATURITY_ORDER.map((value) => (
            <option key={value} value={value}>{formatMaturity(value)}</option>
          ))}
        </select>
        <select className="sa-input" value={form.evidence_mode} onChange={update('evidence_mode')}>
          {EVIDENCE_MODE_OPTIONS.map((item) => (
            <option key={item.value} value={item.value}>{item.label}</option>
          ))}
        </select>
        <select className="sa-input" value={form.question_id} onChange={update('question_id')}>
          <option value="">不归属问题</option>
          {questions.filter((q) => !q.is_purpose).map((q) => (
            <option key={q.question_id} value={q.question_id}>{q.title}</option>
          ))}
        </select>
      </div>
      <textarea
        className="sa-input"
        rows={3}
        placeholder="正文（markdown），可补充这条判断的依据与背景"
        value={form.body}
        onChange={update('body')}
      />
      {error ? <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 13 }}>{error}</p> : null}
      <div>
        <button type="button" className="sa-button" onClick={submit} disabled={submitting}>
          {submitting ? '保存中' : '保存主张'}
        </button>
      </div>
    </div>
  )
}
