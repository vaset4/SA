'use client'

import { useCallback, useEffect, useState } from 'react'
import { DeveloperAccessGate } from '../../../components/auth/DeveloperAccessGate'
import { EmptyState } from '../../../components/shared/EmptyState'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'
import {
  getLightragOverview,
  getNeo4jOverview,
  listLightragChunks,
  listLightragDocuments,
  listLightragEntities,
  listLightragRelations,
} from '../../../lib/api/graph-inspector'

const TABS = [
  { id: 'lightrag', label: 'LightRAG 自动抽取图' },
  { id: 'neo4j', label: 'Neo4j 研究对象图' },
]

const card = { border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 12, padding: 14, background: 'var(--sa-surface, #fff)' }
const cell = { padding: '8px 10px', borderBottom: '1px solid var(--sa-border, #eee)', textAlign: 'left', fontSize: 13, verticalAlign: 'top' }
const faint = { color: 'var(--sa-text-faint, #888)' }

// 单个统计卡
function Stat({ label, value, tone }) {
  const color = tone === 'danger' ? 'var(--sa-danger, #b45141)' : tone === 'warning' ? 'var(--sa-warning, #b9772b)' : 'var(--sa-text, #222)'
  return (
    <div style={{ ...card, minWidth: 116 }}>
      <div style={{ fontSize: 12, ...faint }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600, color }}>{value}</div>
    </div>
  )
}

// 诊断结论：把"类型在 graphml、不在 kv"这个真问题显性化
function FindingNote() {
  return (
    <div style={{ ...card, borderStyle: 'dashed', background: 'var(--sa-surface-soft, #faf7f2)', fontSize: 13, lineHeight: 1.7 }}>
      <strong>诊断结论（数据源真相）：</strong>实体的 <code>entity_type</code> 在 graphml 节点属性里；
      <code>kv_store_full_entities.json</code> 只是「上传→实体名」清单、<b>不含类型</b>。
      现有 <code>lint_lightrag_entities</code> 把后者当 <code>{'{名:{类型}}'}</code> 读是过时假设——本页类型一律以 graphml 为准。
    </div>
  )
}

// 不可用空态（文件缺失 / Neo4j 关闭）
function UnavailableNote({ payload }) {
  return (
    <EmptyState
      title="该图当前不可用"
      description={payload?.reason || '没有可读取的图数据。'}
    />
  )
}

// 概览卡片行 + 类型分布
function OverviewBar({ overview }) {
  if (!overview) return null
  if (overview.available === false) return <UnavailableNote payload={overview} />
  const byType = overview.by_type || {}
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Stat label="实体" value={overview.entity_total ?? 0} />
        <Stat label="关系" value={overview.relation_total ?? 0} />
        <Stat label="可疑(非学术类型)" value={overview.suspect_count ?? 0} tone={overview.suspect_count ? 'warning' : undefined} />
        <Stat label="孤儿(无关系)" value={overview.orphan_count ?? 0} tone={overview.orphan_count ? 'warning' : undefined} />
        <Stat label="重名分组" value={overview.dup_group_count ?? 0} tone={overview.dup_group_count ? 'warning' : undefined} />
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {Object.entries(byType).map(([type, count]) => (
          <span key={type} style={{ ...card, padding: '3px 10px', borderRadius: 999, fontSize: 12 }}>
            {type} · {count}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 11, ...faint }}>源文件：{overview.source_path}</div>
    </div>
  )
}

// 实体筛选条：搜索 / 仅可疑 / 类型
function EntityFilters({ filters, types, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', margin: '4px 0' }}>
      <input
        type="text"
        value={filters.q}
        onChange={(e) => onChange({ ...filters, q: e.target.value })}
        placeholder="搜索名称或描述"
        style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid var(--sa-border, #e5e7eb)', minWidth: 200 }}
      />
      <select
        value={filters.entityType}
        onChange={(e) => onChange({ ...filters, entityType: e.target.value })}
        style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid var(--sa-border, #e5e7eb)' }}
      >
        <option value="">全部类型</option>
        {types.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
        <input type="checkbox" checked={filters.suspect} onChange={(e) => onChange({ ...filters, suspect: e.target.checked })} />
        仅看可疑
      </label>
    </div>
  )
}

// 实体明细表
function EntityTable({ data }) {
  if (!data) return <p style={faint}>加载中…</p>
  if (data.available === false) return <UnavailableNote payload={data} />
  if (!data.items?.length) return <EmptyState title="无匹配实体" description="换个筛选条件试试。" />
  return (
    <div>
      <div style={{ fontSize: 12, ...faint, marginBottom: 4 }}>共 {data.total} 条，显示前 {data.items.length} 条</div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ ...cell, ...faint }}>名称</th>
            <th style={{ ...cell, ...faint }}>类型</th>
            <th style={{ ...cell, ...faint }}>描述</th>
          </tr>
        </thead>
        <tbody>
          {data.items.map((n, i) => (
            <tr key={`${n.name}-${i}`}>
              <td style={cell}>{n.name}</td>
              <td style={cell}>
                {n.is_suspect
                  ? <StatusBadge tone="warning">{n.entity_type || '(未标类型)'} · 可疑</StatusBadge>
                  : <span>{n.entity_type || '(未标类型)'}</span>}
              </td>
              <td style={{ ...cell, color: 'var(--sa-text-soft, #555)', maxWidth: 460 }}>{n.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// 关系明细表
function RelationTable({ data }) {
  if (!data) return <p style={faint}>加载中…</p>
  if (data.available === false) return <UnavailableNote payload={data} />
  if (!data.items?.length) return <EmptyState title="无关系" description="该图暂无关系边。" />
  return (
    <div>
      <div style={{ fontSize: 12, ...faint, marginBottom: 4 }}>共 {data.total} 条，显示前 {data.items.length} 条</div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ ...cell, ...faint }}>源</th>
            <th style={{ ...cell, ...faint }}>目标</th>
            <th style={{ ...cell, ...faint }}>描述 / 关键词</th>
            <th style={{ ...cell, ...faint }}>权重</th>
          </tr>
        </thead>
        <tbody>
          {data.items.map((r, i) => (
            <tr key={`${r.src}-${r.dst}-${i}`}>
              <td style={cell}>{r.src}</td>
              <td style={cell}>{r.dst}</td>
              <td style={{ ...cell, color: 'var(--sa-text-soft, #555)' }}>{r.description}{r.keywords ? ` · ${r.keywords}` : ''}</td>
              <td style={cell}>{r.weight}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// 文档摄入状态：图里到底进了几篇论文、哪些失败（"为什么实体这么少"的第一诊断面）
function DocStatusSection({ data }) {
  if (!data) return <p style={faint}>加载中…</p>
  if (data.available === false) return <UnavailableNote payload={data} />
  const byStatus = data.by_status || {}
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <h3 style={{ margin: 0, fontSize: 15 }}>文档摄入状态（图里进了几篇论文）</h3>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Stat label="文档总数" value={data.total ?? 0} />
        <Stat label="已入图" value={byStatus.processed ?? 0} />
        <Stat label="失败" value={byStatus.failed ?? 0} tone={byStatus.failed ? 'danger' : undefined} />
      </div>
      {data.items?.length ? (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ ...cell, ...faint }}>论文</th>
              <th style={{ ...cell, ...faint }}>状态</th>
              <th style={{ ...cell, ...faint }}>chunks</th>
              <th style={{ ...cell, ...faint }}>长度</th>
              <th style={{ ...cell, ...faint }}>错误</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((d) => (
              <tr key={d.paper_id}>
                <td style={cell}>{d.title || d.paper_id}</td>
                <td style={cell}>
                  {d.status === 'failed'
                    ? <StatusBadge tone="danger">failed</StatusBadge>
                    : <StatusBadge tone="neutral">{d.status || '—'}</StatusBadge>}
                </td>
                <td style={cell}>{d.chunks_count}</td>
                <td style={cell}>{d.content_length}</td>
                <td style={{ ...cell, color: 'var(--sa-danger, #b45141)', maxWidth: 360, fontSize: 12 }}>{d.error_msg}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  )
}

// 文本 chunk / 原文内容：LightRAG 实际摄入了什么（by_doc 覆盖全库，可能比图/状态多）
function ChunksSection({ data }) {
  if (!data) return <p style={faint}>加载中…</p>
  if (data.available === false) return <UnavailableNote payload={data} />
  const byDoc = data.by_doc || {}
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <h3 style={{ margin: 0, fontSize: 15 }}>原文 / Chunks（LightRAG 实际摄入的内容）</h3>
      <div style={{ fontSize: 12, ...faint }}>
        共 {data.total} 个 chunk · 覆盖 {data.doc_count} 篇论文（full_doc_id）· 显示前 {data.items?.length || 0} 个
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {Object.entries(byDoc).map(([doc, n]) => (
          <span key={doc} style={{ ...card, padding: '3px 10px', borderRadius: 999, fontSize: 12 }}>{doc} · {n} chunk</span>
        ))}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {data.items?.map((c) => (
          <div key={c.chunk_id} style={card}>
            <div style={{ display: 'flex', gap: 10, fontSize: 12, ...faint, marginBottom: 6, flexWrap: 'wrap' }}>
              <span>{c.full_doc_id}</span>
              <span>#{c.chunk_order_index}</span>
              <span>{c.tokens} tokens</span>
              <span style={{ fontFamily: 'monospace' }}>{(c.chunk_id || '').slice(0, 16)}…</span>
            </div>
            <div style={{ whiteSpace: 'pre-wrap', fontSize: 12.5, lineHeight: 1.6, maxHeight: 220, overflow: 'auto', color: 'var(--sa-text-soft, #555)' }}>
              {c.content}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

// LightRAG 整页：概览 + 诊断结论 + 实体表 + 关系表
function LightragTab({ libraryId }) {
  const [overview, setOverview] = useState(null)
  const [relations, setRelations] = useState(null)
  const [docs, setDocs] = useState(null)
  const [chunks, setChunks] = useState(null)
  const [entities, setEntities] = useState(null)
  const [filters, setFilters] = useState({ q: '', suspect: false, entityType: '' })
  const [error, setError] = useState(null)

  const loadStatic = useCallback(async () => {
    setError(null)
    try {
      const [ov, rel, doc, chk] = await Promise.all([
        getLightragOverview(libraryId),
        listLightragRelations({ libraryId }),
        listLightragDocuments(libraryId),
        listLightragChunks({ libraryId }),
      ])
      setOverview(ov)
      setRelations(rel)
      setDocs(doc)
      setChunks(chk)
    } catch (e) {
      setError(e?.message || '加载失败')
    }
  }, [libraryId])

  const loadEntities = useCallback(async () => {
    try {
      const res = await listLightragEntities({
        libraryId,
        suspect: filters.suspect ? true : undefined,
        entityType: filters.entityType || undefined,
        q: filters.q || undefined,
      })
      setEntities(res)
    } catch (e) {
      setError(e?.message || '加载失败')
    }
  }, [libraryId, filters])

  useEffect(() => { loadStatic() }, [loadStatic])
  useEffect(() => { loadEntities() }, [loadEntities])

  const types = Object.keys(overview?.by_type || {})
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {error && <p style={{ color: 'var(--sa-danger, red)' }}>{error}</p>}
      <DocStatusSection data={docs} />
      <OverviewBar overview={overview} />
      <FindingNote />
      <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>实体</h3>
        <EntityFilters filters={filters} types={types} onChange={setFilters} />
        <EntityTable data={entities} />
      </section>
      <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>关系</h3>
        <RelationTable data={relations} />
      </section>
      <ChunksSection data={chunks} />
    </div>
  )
}

// Neo4j 研究对象图：按 paper_ids 圈范围看节点/边计数
function Neo4jTab() {
  const [paperIds, setPaperIds] = useState('')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setError(null)
    try {
      setData(await getNeo4jOverview({ paperIds }))
    } catch (e) {
      setError(e?.message || '加载失败')
    }
  }, [paperIds])

  useEffect(() => { load() }, [])  // 首次加载（空 paper_ids → 0，提示输入范围）

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          type="text"
          value={paperIds}
          onChange={(e) => setPaperIds(e.target.value)}
          placeholder="paper_ids（逗号分隔，留空为全空范围）"
          style={{ flex: 1, minWidth: 260, padding: '6px 10px', borderRadius: 6, border: '1px solid var(--sa-border, #e5e7eb)' }}
        />
        <button type="button" className="sa-button" onClick={load}>查询</button>
      </div>
      {error && <p style={{ color: 'var(--sa-danger, red)' }}>{error}</p>}
      {data?.available === false && <UnavailableNote payload={data} />}
      {data?.available && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Stat label="节点" value={data.node_count ?? 0} />
          <Stat label="边" value={data.edge_count ?? 0} />
          <Stat label="论文" value={data.paper_count ?? 0} />
        </div>
      )}
      <p style={{ fontSize: 12, ...faint }}>研究对象图按论文圈定范围；输入 paper_ids 查看其节点/边规模。</p>
    </div>
  )
}

function GraphInspectorContent() {
  const [tab, setTab] = useState('lightrag')
  const [libraryId, setLibraryId] = useState('')

  return (
    <div style={{ padding: 24, maxWidth: 1040, overflowY: 'auto', height: '100%' }}>
      <PageHeader
        eyebrow="Developer"
        variant="workspace"
        title="图谱数据诊断"
        description="只读查看 LightRAG / Neo4j 两套图的真实内容与质量信号，仅 developer 可访问。"
        status={<StatusBadge tone="neutral">{TABS.find((t) => t.id === tab)?.label}</StatusBadge>}
      />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`sa-button${tab === t.id ? '' : ' sa-button--ghost'}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        {tab === 'lightrag' && (
          <input
            type="text"
            value={libraryId}
            onChange={(e) => setLibraryId(e.target.value)}
            placeholder="library_id（留空=默认库）"
            style={{ marginLeft: 'auto', padding: '6px 10px', borderRadius: 6, border: '1px solid var(--sa-border, #e5e7eb)' }}
          />
        )}
      </div>

      {tab === 'lightrag' ? <LightragTab libraryId={libraryId} /> : <Neo4jTab />}
    </div>
  )
}

export default function GraphInspectorPage() {
  return (
    <DeveloperAccessGate
      title="图谱数据诊断"
      description="只读查看两套图的内容与质量信号，仅 developer/admin 可访问。"
    >
      <GraphInspectorContent />
    </DeveloperAccessGate>
  )
}
