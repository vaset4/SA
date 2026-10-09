'use client'

import { useEffect, useState } from 'react'
import { listRagTraces } from '../../../lib/api/observability'
import { DeveloperAccessGate } from '../../../components/auth/DeveloperAccessGate'
import { EmptyState } from '../../../components/shared/EmptyState'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'

function ObservabilityContent() {
  const [traces, setTraces] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')

  async function load(query) {
    setLoading(true)
    setError(null)
    try {
      const params = query ? { q: query } : {}
      const data = await listRagTraces(params)
      setTraces(data?.traces || [])
    } catch (e) {
      setError(e?.message || 'Failed to load traces')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  function handleSearch(e) {
    e.preventDefault()
    load(search.trim())
  }

  return (
    <div style={{ padding: 24, maxWidth: 900, overflowY: 'auto', height: '100%' }}>
      <PageHeader
        eyebrow="Developer"
        variant="workspace"
        title="观测"
        description="查看检索与生成链路，仅 developer 可访问。"
        status={<StatusBadge tone={error ? 'danger' : loading ? 'warning' : 'neutral'}>{error ? '加载失败' : loading ? '加载中' : `记录 ${traces.length}`}</StatusBadge>}
      />
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="request_id or session_id"
          style={{ flex: 1, padding: '6px 10px', borderRadius: 6, border: '1px solid var(--sa-border, #e5e7eb)' }}
        />
        <button type="submit" className="sa-button">Search</button>
      </form>
      {error && <p style={{ color: 'var(--sa-error, red)' }}>{error}</p>}
      {loading && <p>Loading...</p>}
      {!loading && !traces.length ? (
        <EmptyState
          title="暂无观测记录"
          description="触发研究或问答后，相关链路会出现在这里。"
          action={<button type="button" className="sa-button sa-button--ghost" onClick={() => load(search.trim())}>重新加载</button>}
        />
      ) : null}
      {!loading && traces.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {traces.map((t) => (
            <a
              key={t.request_id}
              href={`/observability/${t.request_id}`}
              style={{ display: 'block', padding: 12, border: '1px solid var(--sa-border, #e5e7eb)', borderRadius: 8, textDecoration: 'none', color: 'inherit' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--sa-text-faint)' }}>
                <span>{t.request_id?.slice(0, 8)}...</span>
                <span>{t.created_at?.slice(0, 19)}</span>
              </div>
              <div style={{ marginTop: 4, fontSize: 14 }}>{t.query?.slice(0, 80) || '(no query)'}</div>
              {t.summary && (
                <div style={{ marginTop: 4, fontSize: 12, color: 'var(--sa-text-faint)' }}>
                  {t.summary.mode || ''} · {t.summary.latency_ms?.total || 0}ms · {t.summary.evidence_count || 0} evidence
                </div>
              )}
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ObservabilityPage() {
  return (
    <DeveloperAccessGate
      title="观测"
      description="查看检索与生成链路，仅 developer/admin 可访问。"
    >
      <ObservabilityContent />
    </DeveloperAccessGate>
  )
}
