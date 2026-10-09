'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { Button } from '@heroui/react'
import { listResearchSessions } from '../../lib/api/research'
import { externalResearchModeLabel, normalizeExternalResearchMode } from '../../lib/external-research-mode'
import { EmptyState } from '../shared/EmptyState'
import { StatusBadge } from '../shared/StatusBadge'

function formatTime(value) {
  if (!value) return '时间未知'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '时间未知'
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function resolveTone(status) {
  if (status === 'completed') return 'success'
  if (status === 'awaiting_clarification' || status === 'running') return 'warning'
  if (status === 'failed') return 'danger'
  return 'neutral'
}

function resolveLabel(status) {
  if (status === 'awaiting_clarification') return '等待澄清'
  if (status === 'running') return '进行中'
  if (status === 'completed') return '已完成'
  if (status === 'created') return '已创建'
  return status || '未知状态'
}

function resolveModeLabel(mode) {
  return externalResearchModeLabel(normalizeExternalResearchMode(mode))
}

export function ResearchSessionList({ projectStore, activeSessionId = '' }) {
  const [state, setState] = useState({
    loading: true,
    error: '',
    sessions: [],
  })

  const refresh = async () => {
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const data = await listResearchSessions()
      setState({
        loading: false,
        error: '',
        sessions: data?.sessions || [],
      })
    } catch (reason) {
      setState({
        loading: false,
        error: reason instanceof Error ? reason.message : '研究列表加载失败',
        sessions: [],
      })
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  const projectSessionIds = useMemo(
    () => new Set((projectStore.sessions || []).map((item) => item.session_id).filter(Boolean)),
    [projectStore.sessions],
  )

  return (
    <section className="sa-section-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <span className="sa-section-card__eyebrow">
            Session Index
          </span>
          <div>
            <h3 className="sa-section-card__title">历史研究</h3>
            <p className="sa-section-card__description">
              统一查看历史 session 的模式、证据规模、关键发现和是否属于当前项目。
            </p>
          </div>
        </div>
        <Button size="sm" variant="flat" onPress={refresh} isDisabled={state.loading}>
          {state.loading ? '刷新中' : '刷新列表'}
        </Button>
      </div>

      {state.error ? (
        <EmptyState title="研究列表加载失败" description={state.error} />
      ) : null}

      {!state.error && state.loading ? (
        <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>正在读取研究任务...</p>
      ) : null}

      {!state.error && !state.loading && !state.sessions.length ? (
        <EmptyState
          title="还没有研究记录"
          description="从上方输入课题即可创建第一条深度研究任务，之后会在这里持续沉淀。"
        />
      ) : null}

      {!state.error && !state.loading && state.sessions.length ? (
        <div style={{ display: 'grid', gap: 12 }}>
          {state.sessions.map((session) => {
            const isCurrent = session.session_id === activeSessionId
            const inProject = projectSessionIds.has(session.session_id)
            const keyFindings = Array.isArray(session.key_findings) ? session.key_findings : []
            return (
              <article
                key={session.session_id}
                className="sa-list-card"
                style={isCurrent
                  ? {
                    gap: 12,
                    borderColor: 'color-mix(in srgb, var(--sa-accent) 40%, white)',
                    background: 'linear-gradient(135deg, rgba(217,244,242,0.88), rgba(255,255,255,0.96))',
                    boxShadow: 'var(--sa-shadow-sm)',
                  }
                  : { gap: 12 }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                  <div style={{ display: 'grid', gap: 6, minWidth: 0 }}>
                    <strong style={{ fontSize: 15, lineHeight: 1.5 }}>{session.topic || '未命名研究任务'}</strong>
                    <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                      更新时间：{formatTime(session.updated_at || session.created_at)}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {inProject ? <StatusBadge tone="warning">当前项目</StatusBadge> : null}
                    <StatusBadge tone="neutral">{resolveModeLabel(session.research_mode)}</StatusBadge>
                    {session.paper_ids?.length ? <StatusBadge tone="neutral">背景 {session.paper_ids.length}</StatusBadge> : null}
                    <StatusBadge tone="neutral">证据 {session.evidence_count || 0}</StatusBadge>
                    <StatusBadge tone="neutral">补库 {session.candidate_paper_count || 0}</StatusBadge>
                    <StatusBadge tone={resolveTone(session.status)}>{resolveLabel(session.status)}</StatusBadge>
                  </div>
                </div>

                {keyFindings.length ? (
                  <div className="sa-detail-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                    {keyFindings.slice(0, 3).map((item) => (
                      <div key={item} className="sa-list-card" style={{ padding: 12, fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
                        {item}
                      </div>
                    ))}
                  </div>
                ) : null}

                {session.preview ? (
                  <p style={{ margin: 0, fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>{session.preview}</p>
                ) : null}

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <Link
                    href={`/research/${encodeURIComponent(session.session_id)}`}
                    className="sa-button sa-button--ghost"
                  >
                    查看详情
                  </Link>
                  {session.step_name ? <span className="sa-chip">阶段 {session.step_name}</span> : null}
                </div>
              </article>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}
