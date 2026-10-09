'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  MATURITY_ORDER,
  formatMaturity,
  formatMemoryType,
  formatStatus,
  formatStale,
  formatEvidenceMode,
  formatLinkType,
} from '../../lib/research-memory.mjs'
import {
  getMemory,
  confirmMemory,
  setMemoryMaturity,
  markMemoryStale,
  updateMemory,
  removeEvidence,
} from '../../lib/api/research-memory'
import { MemoryEditForm } from './MemoryEditForm'
import { MemoryEvidenceBindForm } from './MemoryEvidenceBindForm'
import { getError } from '../../lib/study-assistant-workspace.mjs'

const overlayStyle = {
  position: 'fixed',
  inset: 0,
  background: 'rgba(15,23,42,0.32)',
  zIndex: 40,
}

const drawerStyle = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 'min(520px, 92vw)',
  background: 'var(--sa-surface)',
  borderLeft: '1px solid var(--sa-border)',
  boxShadow: '-12px 0 32px rgba(15,23,42,0.18)',
  padding: 24,
  overflowY: 'auto',
  zIndex: 41,
}

// 抽屉内一个信息分区的标题
function SectionLabel({ children }) {
  return (
    <p style={{ margin: '18px 0 8px', fontSize: 12, fontWeight: 600, color: 'var(--sa-text-soft)', letterSpacing: 0.4 }}>
      {children}
    </p>
  )
}

// 单条证据引文块：斜体原句 + chunk 定位 + 校验状态
function EvidenceBlock({ evidenceRef }) {
  const stale = evidenceRef.stale_hint
  const locator = [evidenceRef.section_path || evidenceRef.section, evidenceRef.page_num ? `p.${evidenceRef.page_num}` : '', evidenceRef.chunk_id ? `chunk ${evidenceRef.chunk_id}` : '']
    .filter(Boolean)
    .join(' · ')
  return (
    <div style={{ borderLeft: `3px solid ${stale ? 'var(--sa-danger)' : 'var(--sa-info, #2563eb)'}`, padding: '6px 12px', margin: '8px 0', background: 'var(--sa-surface-soft)', borderRadius: 8 }}>
      <div style={{ fontStyle: 'italic', fontSize: 13, lineHeight: 1.7 }}>&ldquo;{evidenceRef.quote_text}&rdquo;</div>
      <div style={{ fontSize: 11, color: 'var(--sa-text-faint)', marginTop: 4 }}>
        {evidenceRef.paper_id}{locator ? ` · ${locator}` : ''} · {evidenceRef.ref_role || 'primary_support'}
      </div>
    </div>
  )
}

/** @param {object} props 当前记忆及修改操作 @returns {JSX.Element} 成熟度与审核动作 */
function MemoryReviewActions({ projectId, item, busy, runAction }) {
  const memoryId = item.memory_id
  return <>
            <SectionLabel>成熟度</SectionLabel>
            <div style={{ display: 'flex', gap: 6 }}>
              {MATURITY_ORDER.map((level) => (
                <button
                  key={level}
                  type="button"
                  className="sa-button sa-button--ghost"
                  data-active={item.maturity === level}
                  disabled={busy}
                  onClick={() => runAction(() => setMemoryMaturity({ projectId, memoryId, maturity: level, expectedVersion: item.version }), '成熟度调整失败')}
                >
                  {formatMaturity(level)}
                </button>
              ))}
            </div>

            <SectionLabel>操作</SectionLabel>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {item.status !== 'confirmed' || item.stale_state !== 'fresh' ? (
                <button type="button" className="sa-button" disabled={busy} onClick={() => runAction(() => confirmMemory({ projectId, memoryId, expectedVersion: item.version }), '确认失败')}>核对并确认</button>
              ) : null}
              <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => runAction(() => markMemoryStale({ projectId, memoryId, reasons: ['手动标记待重绑'], expectedVersion: item.version }), '标记失败')}>标记证据失效</button>
              {item.status === 'confirmed' ? (
                <button type="button" className="sa-button sa-button--ghost" disabled={busy} onClick={() => runAction(() => updateMemory({ projectId, memoryId, patch: { status: 'draft', expected_version: item.version } }), '回到草稿失败')}>回到草稿</button>
              ) : null}
            </div>
  </>
}

/** @param {object} props 当前记忆、依据与历史 @returns {JSX.Element} 可核对依据及变更历史 */
function MemoryEvidenceHistory({ projectId, item, detail, busy, runAction, onDone }) {
  const memoryId = item.memory_id
  const evidenceRefs = detail.evidence_refs || []
  const links = detail.links || []
  const versions = detail.versions || []
  return <>
            <SectionLabel>证据（{evidenceRefs.length}）</SectionLabel>
            {evidenceRefs.length
              ? evidenceRefs.map((ref) => <div key={ref.ref_id}><EvidenceBlock evidenceRef={ref} />
                  <button type="button" className="sa-button sa-button--ghost" disabled={busy}
                    onClick={() => runAction(() => removeEvidence({ projectId, memoryId, refId: ref.ref_id, expectedVersion: item.version }), '移除依据失败')}>移除此依据</button>
                </div>)
              : <p style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>暂无证据</p>}
            <MemoryEvidenceBindForm key={`bind-${item.version}`} projectId={projectId} item={item} onDone={onDone} />

            <SectionLabel>关系（{links.length}）</SectionLabel>
            {links.length
              ? links.map((link) => (
                  <div key={link.link_id} style={{ fontSize: 12, color: 'var(--sa-text-soft)', padding: '2px 0' }}>
                    {formatLinkType(link.link_type)}：{link.src_memory_id === memoryId ? `→ ${link.dst_memory_id}` : `← ${link.src_memory_id}`}
                  </div>
                ))
              : <p style={{ fontSize: 12, color: 'var(--sa-text-faint)' }}>暂无关系</p>}

            <SectionLabel>版本历史（{versions.length}）</SectionLabel>
            {versions.map((version) => (
              <div key={version.version_id} style={{ fontSize: 12, color: 'var(--sa-text-soft)', padding: '2px 0' }}>
                v{version.version} · {version.change_kind}{version.change_summary ? ` · ${version.change_summary}` : ''}
              </div>
            ))}


  </>
}

/** @param {object} props 当前版本与编辑状态 @returns {JSX.Element} 判断正文及编辑表单 */
function MemoryJudgment({ projectId, item, editing, setEditing, onDone }) {
  return <>
            <h2 style={{ margin: '10px 0 4px', fontSize: 19, lineHeight: 1.5 }}>{item.title}</h2>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              <span className="sa-chip">{formatMaturity(item.maturity)}</span>
              <span className="sa-chip">{formatMemoryType(item.memory_type)}</span>
              <span className="sa-chip">{formatStatus(item.status)}</span>
              <span className="sa-chip">{formatStale(item.stale_state)}</span>
              <span className="sa-chip">{formatEvidenceMode(item.evidence_mode)}</span>
              <span className="sa-chip">v{item.version}</span>
            </div>

            <div style={{ fontSize: 14, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{item.body || '（无正文）'}</div>
            <button type="button" className="sa-button sa-button--ghost" onClick={() => setEditing((value) => !value)}>{editing ? '收起编辑' : '编辑判断'}</button>
            {editing ? <MemoryEditForm key={item.version} projectId={projectId} item={item} onDone={onDone} /> : null}

    {item.stale_reasons?.length ? <p role="status" style={{ color: 'var(--sa-danger)', fontSize: 12 }}>
      待核对原因：{item.stale_reasons.join('；')}
    </p> : null}
  </>
}

/** @param {object} props 项目及记忆身份 @returns {JSX.Element} 持有当前版本的详情抽屉 */
export function MemoryDetailDrawer({ projectId, memoryId, onClose, onChanged }) {
  const [state, setState] = useState({ loading: true, error: '', detail: null })
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState(false)
  const requestSequence = useRef(0)

  /** @returns {Promise<void>} 读取当前身份，忽略切换项目或记忆后的迟到响应 */
  const refresh = useCallback(async () => {
    const sequence = ++requestSequence.current
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const detail = await getMemory({ projectId, memoryId })
      if (sequence === requestSequence.current) setState({ loading: false, error: '', detail })
    } catch (reason) {
      if (sequence === requestSequence.current) setState({ loading: false, error: getError(reason, '记忆详情加载失败'), detail: null })
    }
  }, [projectId, memoryId])

  useEffect(() => {
    refresh()
    return () => { requestSequence.current += 1 }
  }, [refresh])

  /** @param {Function} action 版本化操作 @param {string} failMessage 错误提示 @returns {Promise<void>} 修改后刷新 */
  const runAction = async (action, failMessage) => {
    if (busy) return
    setBusy(true)
    try {
      await action()
      await refresh()
      onChanged?.()
    } catch (reason) {
      if (reason.status === 409) await refresh()
      setState((current) => ({ ...current, error: getError(reason, failMessage) }))
    } finally {
      setBusy(false)
    }
  }

  const item = state.detail?.item
  /** @returns {void} 子表单保存后刷新当前版本 */
  function afterChange() { setEditing(false); refresh(); onChanged?.() }

  return (
    <>
      <div style={overlayStyle} onClick={onClose} />
      <aside style={drawerStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
          <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>记忆详情</p>
          <button type="button" className="sa-button sa-button--ghost" onClick={onClose}>关闭</button>
        </div>

        {state.error ? <p style={{ color: 'var(--sa-danger)', fontSize: 13 }}>{state.error}</p> : null}
        {state.loading ? <p style={{ fontSize: 13, color: 'var(--sa-text-soft)' }}>加载中...</p> : null}

        {item ? (
          <>
            <MemoryJudgment projectId={projectId} item={item} editing={editing} setEditing={setEditing} onDone={afterChange} />
            <MemoryReviewActions projectId={projectId} item={item} busy={busy} runAction={runAction} />
            <MemoryEvidenceHistory projectId={projectId} item={item} detail={state.detail} busy={busy} runAction={runAction} onDone={afterChange} />
          </>
        ) : null}
      </aside>
    </>
  )
}
