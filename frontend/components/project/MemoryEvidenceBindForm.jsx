'use client'

import { useEffect, useState } from 'react'
import { addEvidence } from '../../lib/api/research-memory'
import { getProjectPapers } from '../../lib/api/projects'
import { getError } from '../../lib/study-assistant-workspace.mjs'

/** @param {object} props 项目论文及选择回调 @returns {JSX.Element} 论文选择 */
function PaperPicker({ papers, value, onChange }) {
  return (
    <select aria-label="依据所属论文" className="sa-input" value={value} onChange={onChange}>
      <option value="">请选择项目中的论文</option>
      {papers.map((paper) => <option key={paper.paper_id} value={paper.paper_id}>{paper.title || paper.paper_id}</option>)}
    </select>
  )
}

/** @param {object} props 后端歧义位置与选择回调 @returns {JSX.Element|null} 原文位置选择 */
function SourceChoices({ candidates, selected, onSelect }) {
  if (!candidates.length) return null
  return (
    <fieldset>
      <legend>找到多处相同原句，请选择所指位置</legend>
      {candidates.map((candidate, index) => <label key={`${candidate.chunk_id}-${candidate.source_start}`} style={{ display: 'block', marginTop: 8 }}>
        <input type="radio" name="memory-source" checked={selected === index} onChange={() => onSelect(index)} />
        {candidate.section || '原文片段'}{candidate.page_num ? ` · 第 ${candidate.page_num} 页` : ''}
        <div style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{candidate.text}</div>
      </label>)}
    </fieldset>
  )
}

/**
 * 【项目记忆】【绑定依据】从当前项目原文定位，再原子绑定并确认当前版本
 * @param {{projectId: string, item: object, onDone: Function}} props 项目、记忆与刷新入口
 * @returns {JSX.Element} 原文选择和确认表单
 */
export function MemoryEvidenceBindForm({ projectId, item, onDone }) {
  const [papers, setPapers] = useState([])
  const [paperId, setPaperId] = useState('')
  const [quote, setQuote] = useState('')
  const [candidates, setCandidates] = useState([])
  const [selected, setSelected] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    getProjectPapers({ projectId }).then((data) => {
      if (active) setPapers(data.papers)
    }).catch((reason) => { if (active) setError(getError(reason, '项目论文加载失败')) })
    return () => { active = false }
  }, [projectId])

  /** @returns {Promise<void>} 同事务绑定并确认，不伪造中间成功 */
  async function submit() {
    if (busy) return
    if (!paperId || !quote.trim()) { setError('请选择论文并填写原文摘录'); return }
    setBusy(true)
    setError('')
    try {
      const choice = selected === null ? null : candidates[selected]
      const location = choice ? { chunk_id: choice.chunk_id, source_field: choice.source_field,
        source_start: choice.source_start, source_end: choice.source_end } : {}
      await addEvidence({ projectId, memoryId: item.memory_id,
        payload: { paper_id: paperId, quote_text: quote, expected_version: item.version, confirm: true, ...location } })
      onDone?.()
    } catch (reason) {
      setError(getError(reason, '绑定和确认失败'))
      setCandidates(reason.body?.detail?.candidates || [])
      setSelected(null)
    } finally { setBusy(false) }
  }

  /** @param {Function} setter 表单更新 @param {string} value 新值 @returns {void} 清除旧定位 */
  function change(setter, value) { setter(value); setCandidates([]); setSelected(null) }

  return (
    <div style={{ display: 'grid', gap: 8, marginTop: 8, padding: 10, border: '1px dashed var(--sa-border)', borderRadius: 10 }}>
      <p style={{ margin: 0, fontSize: 12 }}>选择论文并粘贴原文，核对定位后绑定；所有已声明依据有效时才恢复使用。</p>
      <PaperPicker papers={papers} value={paperId} onChange={(event) => change(setPaperId, event.target.value)} />
      <textarea aria-label="原文摘录" className="sa-input" rows={3} value={quote} onChange={(event) => change(setQuote, event.target.value)} />
      <SourceChoices candidates={candidates} selected={selected} onSelect={setSelected} />
      {error ? <p role="alert" style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 13 }}>{error}</p> : null}
      <button type="button" className="sa-button" disabled={busy || (candidates.length > 0 && selected === null)} onClick={submit}>
        {busy ? '核对并保存中' : '绑定依据并确认'}
      </button>
    </div>
  )
}
