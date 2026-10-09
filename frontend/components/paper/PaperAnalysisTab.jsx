'use client'

import { Button } from '@heroui/react'
import { EmptyState } from '../shared/EmptyState'
import { StatusBadge } from '../shared/StatusBadge'
import { getPaperAnalysisStatus } from '../../lib/api/papers'

function panelStyle() {
  return {
    padding: 18,
    borderRadius: 18,
    border: '1px solid var(--sa-border)',
    background: 'rgba(255, 253, 249, 0.94)',
    display: 'grid',
    gap: 14,
  }
}

function asReadableText(value) {
  if (Array.isArray(value)) return value.filter(Boolean).join('；')
  return typeof value === 'string' ? value.trim() : ''
}

function AnalysisBlock({ title, value }) {
  const text = asReadableText(value)
  if (!text) return null
  return (
    <section
      style={{
        padding: 16,
        borderRadius: 16,
        border: '1px solid var(--sa-border)',
        background: 'rgba(246,245,244,0.72)',
        display: 'grid',
        gap: 8,
      }}
    >
      <strong style={{ fontSize: 14 }}>{title}</strong>
      <span style={{ fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{text}</span>
    </section>
  )
}

export function PaperAnalysisTab({ paper, onAnalyze = null, analyzing = false, analysisMessage = '' }) {
  if (!paper) {
    return <EmptyState title="论文详情尚未就绪" description="请先完成论文详情加载，再查看分析内容。" />
  }

  const analysis = paper.analysis || {}
  const analysisStatus = getPaperAnalysisStatus(paper)
  const statusLabel = (
    analysisStatus === 'analysis_ready' ? 'AI 分析已生成'
      : analysisStatus === 'queued' ? 'AI 分析已排队'
        : analysisStatus === 'running' ? 'AI 分析进行中'
          : analysisStatus === 'failed' ? 'AI 分析失败'
            : 'AI 分析待补齐'
  )
  const references = Array.isArray(analysis.references) ? analysis.references : []
  const methodHighlights = [
    analysis.method,
    analysis.methodology_type ? `方法类型：${analysis.methodology_type}` : '',
    analysis.methodology_detail,
  ].filter(Boolean)

  return (
    <section className="sa-section-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <StatusBadge tone={analysisStatus === 'analysis_ready' ? 'success' : 'warning'}>
              {statusLabel}
            </StatusBadge>
            <span className="sa-chip">方法重点</span>
            <span className="sa-chip">参考文献</span>
          </div>
          <p className="sa-section-card__description">
            这一页专门放 AI 生成的中文分析，重点看研究问题、方法重点、主要贡献、局限性和参考文献。
          </p>
        </div>
        {onAnalyze ? (
          <Button size="sm" color="primary" variant="flat" onPress={onAnalyze} isDisabled={analyzing}>
            {analyzing ? '分析中' : '重新运行 AI 分析'}
          </Button>
        ) : null}
      </div>
      {analysisMessage ? <p className="sa-note" style={{ margin: 0 }}>{analysisMessage}</p> : null}

      {!analysis.problem && !analysis.method && !analysis.tldr && !methodHighlights.length ? (
        <EmptyState
          title="当前还没有可展示的 AI 分析"
          description={
            analysisStatus === 'queued'
              ? '分析任务已经排队，完成后会自动刷新结果。'
              : analysisStatus === 'running'
                ? '分析任务正在执行，完成后会自动刷新结果。'
                : analysisStatus === 'failed'
                  ? `上一次分析失败：${paper?.analysis_error || '请稍后重试。'}`
                  : '可以先手动触发 AI 分析，再回来查看结构化结果。'
          }
        />
      ) : (
        <>
          <div className="sa-detail-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
            <AnalysisBlock title="研究问题" value={analysis.problem} />
            <AnalysisBlock title="核心方法" value={analysis.method} />
            <AnalysisBlock title="主要结论" value={analysis.tldr || analysis.conclusion || analysis.conclusion_zh} />
          </div>

          <section style={panelStyle()}>
            <strong style={{ fontSize: 15 }}>方法重点</strong>
            {methodHighlights.length ? (
              <div style={{ display: 'grid', gap: 8 }}>
                {methodHighlights.map((item) => (
                  <div key={item} style={{ fontSize: 13, lineHeight: 1.8, color: 'var(--sa-text-soft)' }}>
                    {item}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                当前还没有明确的方法重点，可先回到总览补齐研究问题和方法概述。
              </p>
            )}
          </section>

          <div className="sa-detail-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
            <AnalysisBlock title="主要贡献" value={analysis.contribution} />
            <AnalysisBlock title="局限性" value={analysis.limitation} />
          </div>

          <section style={panelStyle()}>
            <strong style={{ fontSize: 15 }}>参考文献</strong>
            {references.length ? (
              <div style={{ display: 'grid', gap: 8 }}>
                {references.map((item) => (
                  <div key={item} style={{ fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                    {item}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                当前还没有结构化参考文献列表，后续会和解析材料一起补进来。
              </p>
            )}
          </section>
        </>
      )}
    </section>
  )
}
