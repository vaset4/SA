'use client'

import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import { downloadPaperSource } from '../../lib/api/papers'
import { EmptyState } from '../shared/EmptyState'

/** @param {object} props 已授权论文 @returns {JSX.Element} 原文件下载操作 */
function SourceDownload({ paper }) {
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const url = useRef('')
  useEffect(() => () => { if (url.current) URL.revokeObjectURL(url.current) }, [])
  const download = async () => {
    setBusy(true); setError('')
    try {
      const blob = await downloadPaperSource({ paperId: paper.paper_id })
      if (url.current) URL.revokeObjectURL(url.current)
      url.current = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url.current
      const extension = /\.[a-z0-9]+$/i.exec(paper.pdf_path || '')?.[0] || ''
      const filename = paper.title || paper.paper_id
      link.download = filename.toLowerCase().endsWith(extension) ? filename : filename + extension
      link.click()
    } catch (error) { setError(error.message) }
    finally { setBusy(false) }
  }
  return <div><button type="button" className="sa-button" disabled={busy} onClick={download}>{busy ? '读取原文中...' : '下载原文件'}</button>
    {error ? <p style={{ color: 'var(--sa-danger)' }}>{error}</p> : null}</div>
}

function cardStyle() {
  return {
    padding: 16,
    borderRadius: 16,
    border: '1px solid var(--sa-border)',
    background: 'rgba(255,255,255,0.92)',
    display: 'grid',
    gap: 8,
  }
}

function AssetRow({ label, value }) {
  if (!value) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 12, alignItems: 'start' }}>
      <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>{label}</span>
      <span style={{ fontSize: 13, lineHeight: 1.7, wordBreak: 'break-all' }}>{value}</span>
    </div>
  )
}

export function PaperOriginalTab({ paper }) {
  if (!paper) {
    return <EmptyState title="原始材料暂不可用" description="请先完成论文详情加载，再查看原文材料。" />
  }

  const references = Array.isArray(paper.analysis?.references)
    ? paper.analysis.references
    : Array.isArray(paper.analysis?.related_work)
      ? paper.analysis.related_work
      : []
  const sourceMarkdownPath = paper.analysis?.source_markdown_path || paper.source_markdown_path || ''
  const sectionSummaries = paper.analysis?.section_summaries && typeof paper.analysis.section_summaries === 'object'
    ? Object.entries(paper.analysis.section_summaries)
    : []
  const figures = Array.isArray(paper.analysis?.figures) ? paper.analysis.figures : []

  return (
    <section className="sa-section-card">
      <div className="sa-section-card__header">
        <p className="sa-section-card__eyebrow">Source Material</p>
        <h3 className="sa-section-card__title">原始材料</h3>
        <p className="sa-section-card__description">
          这里主要保留原始 PDF、Markdown 解析产物和回源信息，不再重复展示总览页与 AI 分析页已经讲过的内容。
        </p>
      </div>

      <section style={cardStyle()}>
        <strong style={{ fontSize: 14 }}>原始 PDF / 解析文件</strong>
        {paper.source === 'upload' ? <SourceDownload paper={paper} /> : null}
        {paper.pdf_url ? (
          <Link href={paper.pdf_url} target="_blank" rel="noreferrer" className="sa-button sa-button--ghost">
            打开原始 PDF
          </Link>
        ) : null}
        <AssetRow label="PDF 地址" value={paper.pdf_url} />
        <AssetRow label="本地 PDF 路径" value={paper.pdf_path} />
        <AssetRow label="Markdown 解析文件" value={sourceMarkdownPath} />
      </section>

      <section style={cardStyle()}>
        <strong style={{ fontSize: 14 }}>回源说明</strong>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.8, color: 'var(--sa-text-soft)' }}>
          如果需要核对 AI 分析、回看公式图表，优先打开原始 PDF；如果是上传论文，也可以顺着 Markdown 解析文件继续回查正文结构。
        </p>
        {!paper.pdf_url && !paper.pdf_path && !sourceMarkdownPath ? (
          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.8, color: 'var(--sa-text-soft)' }}>
            当前还没有可直接打开的原始材料入口，后续可以补 PDF 访问地址或解析产物。
          </p>
        ) : null}
      </section>

      {(sectionSummaries.length || figures.length) ? (
        <section style={cardStyle()}>
          <strong style={{ fontSize: 14 }}>解析线索</strong>
          {sectionSummaries.length ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {sectionSummaries.slice(0, 6).map(([key, value]) => (
                <div key={key} style={{ fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                  {value?.heading || key}
                  {Array.isArray(value?.formula_blocks) && value.formula_blocks.length ? ` · 公式 ${value.formula_blocks.length}` : ''}
                </div>
              ))}
            </div>
          ) : null}
          {figures.length ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {figures.slice(0, 4).map((item, index) => (
                <div key={`${item?.caption || 'figure'}-${index}`} style={{ fontSize: 13, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                  {item?.caption || `图表 ${index + 1}`}
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      <section style={cardStyle()}>
        <strong style={{ fontSize: 14 }}>参考文献</strong>
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
            当前还没有可直接展示的参考文献列表，后续会随着论文材料包一起补全。
          </p>
        )}
      </section>
    </section>
  )
}
