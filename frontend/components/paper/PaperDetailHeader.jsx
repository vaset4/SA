'use client'

import Link from 'next/link'
import { Button } from '@heroui/react'
import { PageHeader } from '../shared/PageHeader'
import { StatusBadge } from '../shared/StatusBadge'
import { getPaperAnalysisStatus } from '../../lib/api/papers'

function statusText(paper, loading) {
  if (loading) return '加载中'
  const status = getPaperAnalysisStatus(paper)
  switch (status) {
    case 'analysis_ready':
    case 'succeeded':
      return '分析已就绪'
    case 'running':
    case 'queued':
      return '分析处理中'
    default:
      return '待补分析'
  }
}

function statusTone(paper, loading) {
  if (loading) return 'warning'
  return ['analysis_ready', 'succeeded'].includes(getPaperAnalysisStatus(paper)) ? 'success' : 'neutral'
}

export function PaperDetailHeader({
  paper,
  loading,
  projectName = '',
  onAddToProject,
  onStartChat,
  onStartResearch,
}) {
  const title = loading ? '论文详情加载中...' : paper?.title || '论文详情'

  return (
    <PageHeader
      eyebrow="Paper Material Desk"
      variant="paper"
      title={title}
      description={paper?.abstract ? paper.abstract.slice(0, 240) : '这里先看总览、AI 分析和原始材料，再决定要不要把它纳入项目主线。'}
      status={(
        <>
          {paper?.is_vectorized ? <StatusBadge tone="success">已可检索</StatusBadge> : null}
          {paper?.metadata_missing?.includes('title') ? <StatusBadge tone="warning">标题待识别，当前显示文件名</StatusBadge> : null}
          <StatusBadge tone={statusTone(paper, loading)}>{statusText(paper, loading)}</StatusBadge>
          {paper?.year ? <span className="sa-chip">年份 {paper.year}</span> : null}
          {paper?.category ? <span className="sa-chip">{paper.category}</span> : null}
        </>
      )}
      actions={(
        <>
          <Link href="/papers" prefetch={false}>
            <Button size="sm" variant="light">返回知识库</Button>
          </Link>
          {paper?.paper_id ? (
            <Button size="sm" color="secondary" variant="flat" onPress={() => onAddToProject(paper.paper_id)}>
              {projectName ? `加入项目 · ${projectName}` : '加入项目'}
            </Button>
          ) : null}
          <Button size="sm" variant="light" onPress={onStartChat}>
            围绕此文提问
          </Button>
          <Button size="sm" color="primary" variant="flat" onPress={onStartResearch}>
            发起内部研究
          </Button>
          {paper?.pdf_url ? (
            <Link href={paper.pdf_url} target="_blank" rel="noreferrer">
              <Button size="sm" variant="light">查看原始 PDF</Button>
            </Link>
          ) : null}
        </>
      )}
    />
  )
}
