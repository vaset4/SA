'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Button } from '@heroui/react'
import { EmptyState } from '../shared/EmptyState'
import { StatusBadge } from '../shared/StatusBadge'
import { getPaperAnalysisStatus, updatePaperMetadata } from '../../lib/api/papers'

function buildDraft(paper) {
  return {
    title: paper?.title || '',
    authors: Array.isArray(paper?.authors) ? paper.authors.join('; ') : '',
    year: paper?.year || '',
    category: paper?.category || '',
    abstract: paper?.abstract || '',
    problem: paper?.analysis?.problem || '',
    method: paper?.analysis?.method || '',
    conclusion: paper?.analysis?.tldr || paper?.analysis?.conclusion || '',
  }
}

function cardStyle() {
  return {
    padding: 16,
    borderRadius: 18,
    border: '1px solid var(--sa-border)',
    background: 'rgba(255, 253, 249, 0.92)',
    display: 'grid',
    gap: 10,
  }
}

function Field({ label, value, placeholder, onChange, multiline = false }) {
  return (
    <label style={{ display: 'grid', gap: 6 }}>
      <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{label}</span>
      {multiline ? (
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="sa-textarea"
          style={{ minHeight: 120 }}
        />
      ) : (
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="sa-input"
        />
      )}
    </label>
  )
}

export function PaperOverviewTab({ paper, paperId = '', loading, error, onPaperUpdated }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(() => buildDraft(paper))
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')

  useEffect(() => {
    setDraft(buildDraft(paper))
    setEditing(false)
    setActionError('')
  }, [paper])

  if (loading) {
    return (
      <section style={cardStyle()}>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--sa-text-soft)' }}>论文详情加载中...</p>
      </section>
    )
  }

  if (error) {
    return (
      <EmptyState
        title="论文详情加载失败"
        description={error}
        action={<Link href="/papers" className="sa-button sa-button--ghost">返回论文库</Link>}
      />
    )
  }

  if (!paper) {
    return (
      <EmptyState
        title="未找到论文信息"
        description="请返回论文库重试，或刷新当前页面。"
        action={<Link href="/papers" className="sa-button sa-button--ghost">返回论文库</Link>}
      />
    )
  }

  const analysisStatus = getPaperAnalysisStatus(paper)
  const metadataComplete = Boolean(paper.year && paper.category && Array.isArray(paper.authors) && paper.authors.length && paper.abstract)

  const updateDraftField = (field, value) => {
    setDraft((current) => ({ ...current, [field]: value }))
  }

  const handleSave = async () => {
    if (!paperId || busy) return
    setBusy(true)
    setActionError('')
    try {
      const updated = await updatePaperMetadata({
        paperId,
        payload: {
          title: draft.title,
          authors: draft.authors,
          year: draft.year,
          category: draft.category,
          abstract: draft.abstract,
          problem: draft.problem,
          method: draft.method,
          tldr: draft.conclusion,
        },
      })
      onPaperUpdated?.(updated)
      setEditing(false)
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : '总览保存失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section
      className="sa-section-card"
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <StatusBadge tone={analysisStatus === 'analysis_ready' ? 'success' : 'warning'}>
              {analysisStatus === 'analysis_ready' ? '已具备研究材料' : '待补分析材料'}
            </StatusBadge>
            <span className="sa-chip">{metadataComplete ? '元数据完整' : '待补元数据'}</span>
            {paper.year ? <span className="sa-chip">年份 {paper.year}</span> : null}
            {paper.category ? <span className="sa-chip">{paper.category}</span> : null}
          </div>
          <p className="sa-section-card__description">
            总览页只保留基础信息和原始摘要，不再和 AI 分析页重复显示研究问题、方法和结论。
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {editing ? (
            <>
              <Button size="sm" variant="light" onPress={() => setEditing(false)} isDisabled={busy}>
                取消
              </Button>
              <Button size="sm" color="primary" onPress={handleSave} isDisabled={busy}>
                保存元数据
              </Button>
            </>
          ) : (
            <Button size="sm" color="primary" variant="flat" onPress={() => setEditing(true)} isDisabled={busy}>
              编辑元数据
            </Button>
          )}
        </div>
      </div>

      {actionError ? <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-danger)' }}>{actionError}</p> : null}

      <section style={cardStyle()}>
        <strong style={{ fontSize: 14 }}>基础信息</strong>
        <div className="sa-chip-list">
          {Array.isArray(paper.authors) && paper.authors.length ? (
            <span className="sa-chip">作者 {paper.authors.slice(0, 3).join('，')}{paper.authors.length > 3 ? ' 等' : ''}</span>
          ) : null}
          {paper.year ? <span className="sa-chip">年份 {paper.year}</span> : null}
          {paper.category ? <span className="sa-chip">{paper.category}</span> : null}
          <span className="sa-chip">{paper.source === 'upload' ? '上传论文' : '知识库论文'}</span>
        </div>
      </section>

      <section style={cardStyle()}>
        <strong style={{ fontSize: 14 }}>摘要</strong>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.8, color: 'var(--sa-text-soft)' }}>
          {paper.abstract || '当前还没有摘要，可先在这里手动整理一版，后续再补完整分析。'}
        </p>
      </section>

      {editing ? (
        <section style={{ ...cardStyle(), gap: 12 }}>
          <strong style={{ fontSize: 14 }}>编辑总览</strong>
          <Field label="标题" value={draft.title} placeholder="请输入论文标题" onChange={(value) => updateDraftField('title', value)} />
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <Field label="作者" value={draft.authors} placeholder="多个作者用分号分隔" onChange={(value) => updateDraftField('authors', value)} />
            <Field label="年份" value={draft.year} placeholder="如 2026" onChange={(value) => updateDraftField('year', value)} />
          </div>
          <Field label="分类" value={draft.category} placeholder="如 agent, education" onChange={(value) => updateDraftField('category', value)} />
          <Field label="摘要" value={draft.abstract} placeholder="补充原始摘要" onChange={(value) => updateDraftField('abstract', value)} multiline />
          <Field label="研究问题" value={draft.problem} placeholder="这篇论文主要在解决什么问题" onChange={(value) => updateDraftField('problem', value)} multiline />
          <Field label="核心方法" value={draft.method} placeholder="一句话说明方法路线" onChange={(value) => updateDraftField('method', value)} multiline />
          <Field label="主要结论" value={draft.conclusion} placeholder="先写本轮最值得带走的结论" onChange={(value) => updateDraftField('conclusion', value)} multiline />
        </section>
      ) : null}
    </section>
  )
}
