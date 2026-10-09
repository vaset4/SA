'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@heroui/react'
import { EXTERNAL_RESEARCH_MODE, externalResearchModeLabel } from '../../lib/external-research-mode'
import { resolveResearchSourceLabel } from '../../lib/workspace-routes'
import { StatusBadge } from '../shared/StatusBadge'

export function ResearchLauncher({ workspace, projectStore = null, variant = 'standalone', prefill = null }) {
  const router = useRouter()
  const [prompt, setPrompt] = useState('')
  const embedded = variant === 'embedded'
  const prefillPrompt = String(prefill?.prompt || '').trim()
  const prefillProjectId = String(prefill?.projectId || '').trim()
  const prefillPaperId = String(prefill?.paperId || '').trim()
  const prefillSourceLabel = resolveResearchSourceLabel(prefill?.source)
  const hasPrefill = Boolean(prefillPrompt || prefillSourceLabel)

  useEffect(() => {
    setPrompt(prefillPrompt)
  }, [prefillPrompt])

  const launch = async () => {
    const cleanPrompt = prompt.trim()
    if (!cleanPrompt || workspace.busy || workspace.restoring) return
    const projectId = prefillProjectId
    const projectPapers = projectStore?.activeProjectId === projectId && Array.isArray(projectStore.papers)
      ? projectStore.papers
      : []
    const papers = prefillPaperId
      ? [projectPapers.find((paper) => paper?.paper_id === prefillPaperId) || { paper_id: prefillPaperId }]
      : projectPapers
    await workspace.launchResearch({
      prompt: cleanPrompt,
      projectId,
      papers,
      researchMode: EXTERNAL_RESEARCH_MODE,
      onResearchSession(sessionId) {
        router.push(`/research/${encodeURIComponent(sessionId)}`)
      },
    })
    setPrompt('')
  }

  return (
    <section
      className="sa-panel-card sa-section-card sa-section-card--accent"
      style={{
        padding: embedded ? 16 : 18,
        background: 'linear-gradient(142deg, rgba(11,122,117,0.1), rgba(255,255,255,0.96) 48%, rgba(217,119,6,0.08) 100%)',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <div
        style={{
          position: 'absolute',
          right: -72,
          top: -72,
          width: 220,
          height: 220,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(11,122,117,0.18), rgba(11,122,117,0))',
          pointerEvents: 'none',
        }}
      />

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ display: 'grid', gap: 8, maxWidth: 780 }}>
          <span style={{ fontSize: 12, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--sa-text-soft)' }}>
            {embedded ? 'External Mode' : 'Deep Research'}
          </span>
          <div>
            <h3 style={{ margin: 0, fontSize: embedded ? 22 : 26, lineHeight: 1.08 }}>
              {embedded ? '外部研究模式' : '输入研究主题'}
            </h3>
          </div>
        </div>
        <div style={{ display: 'grid', gap: 8, justifyItems: 'end' }}>
          <StatusBadge tone={workspace.busy ? 'warning' : 'neutral'}>
            {workspace.busy ? '研究进行中' : '工作台待命'}
          </StatusBadge>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <span className="sa-chip">模式 {externalResearchModeLabel(EXTERNAL_RESEARCH_MODE)}</span>
            <span className="sa-chip">网络搜索</span>
            <span className="sa-chip">独立会话</span>
          </div>
        </div>
      </div>

      {hasPrefill ? (
        <div
          className="sa-list-card"
          style={{
            gap: 10,
            padding: 16,
            borderColor: 'color-mix(in srgb, var(--sa-accent) 22%, var(--sa-border))',
            background: 'rgba(255,255,255,0.8)',
          }}
        >
          <div style={{ display: 'grid', gap: 6 }}>
            <strong style={{ fontSize: 14 }}>已接住外部入口</strong>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {prefillSourceLabel ? <span className="sa-chip">来源 {prefillSourceLabel}</span> : null}
            {prefillPrompt ? <span className="sa-chip">已带入研究问题</span> : null}
          </div>
        </div>
      ) : null}

      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="例如：梳理最近两年多模态 RAG 的代表性方法、评测基准、主要争议点和后续值得继续研究的问题。"
        className="sa-textarea"
        style={{
          minHeight: 124,
          background: 'rgba(255,255,255,0.88)',
          borderColor: 'color-mix(in srgb, var(--sa-accent) 16%, var(--sa-border))',
        }}
        disabled={workspace.busy || workspace.restoring}
      />

      {workspace.error ? (
        <p style={{ margin: 0, color: 'var(--sa-danger)', fontSize: 12 }}>{workspace.error}</p>
      ) : null}

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>完成后自动进入研究详情页。</span>
        <Button
          size="sm"
          color="primary"
          onPress={launch}
          isDisabled={!prompt.trim() || workspace.busy || workspace.restoring}
        >
          {workspace.busy ? '研究启动中' : '开始研究'}
        </Button>
      </div>
    </section>
  )
}
