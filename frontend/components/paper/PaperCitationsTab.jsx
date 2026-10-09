'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Button } from '@heroui/react'
import { getCitations } from '../../lib/api/papers'
import { EmptyState } from '../shared/EmptyState'
import { PaperEvidencePathPanel } from './PaperEvidencePathPanel'

export function PaperCitationsTab({ paperId, projectId = '' }) {
  const [depth, setDepth] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [tree, setTree] = useState(null)

  const runFetch = async () => {
    if (!paperId || busy) return
    setBusy(true)
    setError('')
    try {
      const data = await getCitations({ paperId, depth })
      setTree(data)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '引用链加载失败')
      setTree(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section style={{ border: '1px solid var(--sa-border)', borderRadius: 12, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
        <h3 style={{ margin: 0, fontSize: 15 }}>引用链分析</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {paperId ? (
            <Link href="/graph" prefetch={false}>
              <Button size="sm" variant="light">查看论文图</Button>
            </Link>
          ) : null}
          <label style={{ color: 'var(--sa-text-soft)', fontSize: 12 }}>
            深度
            <select
              value={depth}
              onChange={(event) => setDepth(Number(event.target.value))}
              style={{
                marginLeft: 6,
                height: 30,
                border: '1px solid var(--sa-border)',
                borderRadius: 8,
                padding: '0 8px',
                background: 'var(--sa-bg)',
              }}
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </label>
          <Button size="sm" variant="flat" onPress={runFetch} isDisabled={busy || !paperId}>
            {busy ? '加载中' : '加载引用链'}
          </Button>
        </div>
      </div>

      {error ? <p style={{ margin: '10px 0 0', color: 'var(--sa-danger)', fontSize: 12 }}>{error}</p> : null}

      <div style={{ marginTop: 12 }}>
        {!tree ? (
          <EmptyState
            title="尚未加载引用链"
            description="点击“加载引用链”后，将在这里显示引用树和最小图证据路径。"
          />
        ) : (
          <div className="sa-stack">
            <div
              className="sa-tool-card"
              style={{ padding: 12, fontSize: 12, color: 'var(--sa-text-soft)' }}
            >
              <p style={{ margin: 0 }}>引用论文数：{tree.references?.length || 0}</p>
              <p style={{ margin: '6px 0 0' }}>被引论文数：{tree.cited_by?.length || 0}</p>
            </div>
            <PaperEvidencePathPanel
              graphEvidence={tree.graph_evidence || []}
              evidenceSummary={tree.evidence_summary || ''}
            />
          </div>
        )}
      </div>
    </section>
  )
}
