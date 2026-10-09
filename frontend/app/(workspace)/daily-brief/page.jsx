'use client'

import Link from 'next/link'
import { EmptyState } from '../../../components/shared/EmptyState'
import { PageHeader } from '../../../components/shared/PageHeader'

export default function DailyBriefPage() {
  return (
    <div style={{ minHeight: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      <PageHeader
        eyebrow="Daily Brief"
        variant="workspace"
        title="每日速览"
        description="集中查看近期值得推进的项目、论文和研究动态。"
        status={(
          <>
            <span className="sa-chip">今日摘要</span>
            <span className="sa-chip">项目动态</span>
          </>
        )}
        actions={(
          <Link href="/projects" className="sa-button sa-button--ghost">
            回项目页
          </Link>
        )}
      />

      <main className="sa-workbench">
        <EmptyState
          title="今天暂无速览"
          description="当前还没有可汇总的今日动态。项目活动、论文更新和研究记录沉淀后会在这里集中展示。"
        />
      </main>
    </div>
  )
}
