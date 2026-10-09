'use client'

import { useEffect, useMemo, useState } from 'react'

import { EmptyState } from '../shared/EmptyState'
import { MemoryDetailDrawer } from './MemoryDetailDrawer'
import { MemoryCreateForm } from './MemoryCreateForm'
import { MemoryReviewPanel } from './MemoryReviewPanel'
import { MemoryGraphView } from './MemoryGraphView'
import { MemoryStateBoard } from './MemoryStateBoard'
import {
  MATURITY_ORDER,
  formatMaturity,
  formatMemoryType,
  formatStale,
} from '../../lib/research-memory.mjs'
import { listMemory, listQuestions, getMaturityCounts, curateMemory, listMemoryLinks } from '../../lib/api/research-memory'
import { getError } from '../../lib/study-assistant-workspace.mjs'

// 研究目的置顶横幅：绿色大横幅 + 四类成熟度计数
function PurposeBanner({ purpose, counts }) {
  const pending = (counts.intuition || 0) + (counts.leaning || 0)
  return (
    <section style={{ background: 'var(--sa-accent, #16a34a)', color: '#fff', borderRadius: 16, padding: '20px 24px', marginBottom: 16 }}>
      <p style={{ margin: 0, fontSize: 12, opacity: 0.85 }}>研究目的</p>
      <h2 style={{ margin: '6px 0 12px', fontSize: 20, lineHeight: 1.5 }}>
        {purpose?.title || '尚未设定研究目的'}
      </h2>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 13 }}>
        <span>已确立 {counts.established || 0}</span>
        <span>倾向 {counts.leaning || 0}</span>
        <span>直觉 {counts.intuition || 0}</span>
        <span>待解 {pending}</span>
      </div>
    </section>
  )
}

// 已确立记忆：大卡形态，展示陈述 + 证据模式
function EstablishedCard({ item, onOpen }) {
  return (
    <article className="sa-list-card" style={{ borderLeft: '3px solid var(--sa-accent, #16a34a)', cursor: 'pointer' }} onClick={onOpen}>
      <p style={{ margin: 0, fontSize: 11, color: 'var(--sa-text-faint)' }}>我主张</p>
      <div style={{ fontSize: 15, lineHeight: 1.7, fontWeight: 500 }}>{item.title}</div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
        <span className="sa-chip">{formatMemoryType(item.memory_type)}</span>
        {item.stale_state !== 'fresh' ? <span className="sa-chip">{formatStale(item.stale_state)}</span> : null}
        <span className="sa-chip">v{item.version}</span>
      </div>
    </article>
  )
}

// 待解记忆：紧凑行形态，maturity 进度点 + 陈述 + 新鲜度
function PendingRow({ item, onOpen }) {
  const level = MATURITY_ORDER.indexOf(item.maturity) + 1
  return (
    <article className="sa-list-card" style={{ cursor: 'pointer', display: 'flex', gap: 10, alignItems: 'center' }} onClick={onOpen}>
      <span style={{ display: 'flex', gap: 3 }}>
        {MATURITY_ORDER.map((_, index) => (
          <span key={index} style={{ width: 7, height: 7, borderRadius: '50%', background: index < level ? 'var(--sa-accent, #16a34a)' : 'var(--sa-border)' }} />
        ))}
      </span>
      <span style={{ flex: 1, fontSize: 14, lineHeight: 1.6 }}>{item.title}</span>
      <span className="sa-chip">{formatMaturity(item.maturity)}</span>
      {item.stale_state !== 'fresh' ? <span className="sa-chip">{formatStale(item.stale_state)}</span> : null}
    </article>
  )
}

// 进度状态条：台账顶部一小条，显示各成熟度计数（established/leaning/待解）
function ProgressBar({ counts }) {
  const established = counts.established || 0
  const leaning = counts.leaning || 0
  const unresolved = (counts.intuition || 0) + leaning
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '4px 0 12px' }}>
      <span className="sa-chip">已确立 {established}</span>
      <span className="sa-chip">倾向 {leaning}</span>
      <span className="sa-chip">待解 {unresolved}</span>
    </div>
  )
}

// 台账视图：研究目的横幅 + 进度条 + 可写进论文（established）+ 待解（非 established）两段
function LedgerView({ state, established, pending, purpose, rqFilter, setRqFilter, showCreate, setShowCreate, projectId, onRefresh, onOpen }) {
  return (
    <>
      <PurposeBanner purpose={purpose} counts={state.counts} />
      <ProgressBar counts={state.counts} />

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 13 }}>
          研究问题
          <select className="sa-input" value={rqFilter} onChange={(event) => setRqFilter(event.target.value)}>
            <option value="">全部</option>
            {state.questions.filter((q) => !q.is_purpose).map((q) => (
              <option key={q.question_id} value={q.question_id}>{q.title}</option>
            ))}
          </select>
        </label>
        <button type="button" className="sa-button" onClick={() => setShowCreate((value) => !value)}>
          {showCreate ? '收起' : '新主张'}
        </button>
      </div>

      {showCreate ? (
        <MemoryCreateForm
          projectId={projectId}
          questions={state.questions}
          onCreated={() => { setShowCreate(false); onRefresh() }}
        />
      ) : null}

      {state.error ? <p style={{ margin: 0, color: 'var(--sa-danger)' }}>{state.error}</p> : null}
      {state.loading ? <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>加载中...</p> : null}

      <h3 className="sa-section-card__title" style={{ marginTop: 18 }}>可写进论文（{established.length}）</h3>
      {established.length ? (
        <div style={{ display: 'grid', gap: 10 }}>
          {established.map((item) => (
            <EstablishedCard key={item.memory_id} item={item} onOpen={() => onOpen(item.memory_id)} />
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 13, color: 'var(--sa-text-faint)', display: 'grid', gap: 8 }}>
          <p style={{ margin: 0 }}>还没有已确立的主张。把成熟的判断升级到「已确立」，它就会排进论证台账。</p>
          <p style={{ margin: 0 }}>
            从聊天沉淀：在研究对话回答下方点「快速保存判断」或「AI 整理本对话」，再到这里的「AI 待审区」确认。
          </p>
        </div>
      )}

      <h3 className="sa-section-card__title" style={{ marginTop: 18 }}>待解（{pending.length}）</h3>
      {pending.length ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {pending.map((item) => (
            <PendingRow key={item.memory_id} item={item} onOpen={() => onOpen(item.memory_id)} />
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 13, color: 'var(--sa-text-faint)', display: 'grid', gap: 8 }}>
          <p style={{ margin: 0 }}>暂无待解主张。</p>
          <p style={{ margin: 0 }}>
            可点「新主张」手写一条，或从聊天把本轮结论「快速保存判断」。
          </p>
        </div>
      )}
    </>
  )
}

export function ProjectMemoryPanel({ projectStore }) {
  const projectId = projectStore.activeProjectId
  const [state, setState] = useState({ loading: true, error: '', items: [], questions: [], counts: {} })
  const [rqFilter, setRqFilter] = useState('')
  const [activeMemoryId, setActiveMemoryId] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [view, setView] = useState('ledger')
  const [curating, setCurating] = useState(false)
  const [links, setLinks] = useState([])

  const refresh = async () => {
    if (!projectId) {
      setState({ loading: false, error: '', items: [], questions: [], counts: {} })
      setLinks([])
      return
    }
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const filters = { status: 'all', ...(rqFilter ? { question_id: rqFilter } : {}) }
      const [items, questions, counts, linkRows] = await Promise.all([
        listMemory({ projectId, filters }),
        listQuestions({ projectId }),
        getMaturityCounts({ projectId }),
        listMemoryLinks({ projectId }),
      ])
      setState({ loading: false, error: '', items: items || [], questions: questions || [], counts: counts || {} })
      setLinks(linkRows || [])
    } catch (reason) {
      setState({ loading: false, error: getError(reason, '研究记忆加载失败'), items: [], questions: [], counts: {} })
    }
  }

  useEffect(() => {
    refresh().catch(() => {})
  }, [projectId, rqFilter])

  // 触发 AI 整理：圈定当前 RQ 过滤（无则需用户在对话侧发起），产草稿后切到待审区
  const handleCurate = async () => {
    if (!projectId || curating) return
    // AI 整理：有研究问题则按问题圈定；否则不强制问题（仍须用户从聊天用 conversation 圈定更稳）
    const scope = rqFilter ? { question_id: rqFilter } : {}
    if (!scope.question_id) {
      setState((current) => ({
        ...current,
        error: 'AI 整理需先选定研究问题；或在聊天回答下使用「AI 整理本对话」按会话沉淀。',
      }))
      return
    }
    setCurating(true)
    try {
      await curateMemory({ projectId, scope })
      setView('review')
    } catch (reason) {
      setState((current) => ({ ...current, error: getError(reason, 'AI 整理失败') }))
    } finally {
      setCurating(false)
    }
  }

  const purpose = useMemo(() => state.questions.find((q) => q.is_purpose) || null, [state.questions])
  const established = useMemo(() => state.items.filter((item) => item.maturity === 'established'), [state.items])
  const pending = useMemo(() => state.items.filter((item) => item.maturity !== 'established'), [state.items])

  if (!projectId) {
    return <EmptyState title="请先选择项目" description="选择项目后，才能按成熟度浏览研究记忆。" />
  }

  return (
    <section className="sa-panel-card sa-section-card">
      <div className="sa-tab-strip" style={{ marginBottom: 12 }}>
        <button type="button" className="sa-tab-button" data-active={view === 'ledger'} onClick={() => setView('ledger')}>论证台账</button>
        <button type="button" className="sa-tab-button" data-active={view === 'states'} onClick={() => setView('states')}>状态</button>
        <button type="button" className="sa-tab-button" data-active={view === 'graph'} onClick={() => setView('graph')}>关系图谱</button>
        <button type="button" className="sa-tab-button" data-active={view === 'review'} onClick={() => setView('review')}>AI 待审区</button>
        <button type="button" className="sa-button sa-button--ghost" style={{ marginLeft: 'auto' }} disabled={curating} onClick={handleCurate}>
          {curating ? 'AI 整理中' : 'AI 整理'}
        </button>
      </div>

      {view === 'review' ? (
        <MemoryReviewPanel projectStore={projectStore} />
      ) : view === 'states' ? (
        <MemoryStateBoard
          items={state.items}
          projectId={projectId}
          loading={state.loading}
          error={state.error}
          onOpen={setActiveMemoryId}
          onRefresh={() => refresh().catch(() => {})}
        />
      ) : view === 'graph' ? (
        <MemoryGraphView items={state.items} links={links} onSelect={setActiveMemoryId} />
      ) : (
        <LedgerView
          state={state}
          established={established}
          pending={pending}
          purpose={purpose}
          rqFilter={rqFilter}
          setRqFilter={setRqFilter}
          showCreate={showCreate}
          setShowCreate={setShowCreate}
          projectId={projectId}
          onRefresh={() => refresh().catch(() => {})}
          onOpen={setActiveMemoryId}
        />
      )}

      {activeMemoryId ? (
        <MemoryDetailDrawer
          projectId={projectId}
          memoryId={activeMemoryId}
          onClose={() => setActiveMemoryId(null)}
          onChanged={() => refresh().catch(() => {})}
        />
      ) : null}
    </section>
  )
}
