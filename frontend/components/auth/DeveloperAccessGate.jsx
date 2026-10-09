'use client'

import { useAuthStore } from '../../stores/auth-store'
import { WorkspaceLayout } from '../layout/WorkspaceLayout'
import { EmptyState } from '../shared/EmptyState'
import { PageHeader } from '../shared/PageHeader'
import { Skeleton } from '../shared/Skeleton'

const DEVELOPER_ROLES = new Set(['developer', 'admin'])

export function isDeveloperRole(role) {
  return DEVELOPER_ROLES.has(role)
}

function AccessLoading() {
  return (
    <main className="sa-workbench">
      <section className="sa-section-card">
        <Skeleton rows={4} height={18} />
      </section>
    </main>
  )
}

function AccessDenied({ title, description }) {
  return (
    <WorkspaceLayout
      focusMode
      header={(
        <PageHeader
          eyebrow="Access"
          variant="workspace"
          title={title}
          description={description}
        />
      )}
    >
      <div className="sa-workbench">
        <EmptyState
          title="当前账号没有访问权限"
          description="该页面用于系统诊断与排障。普通研究流程仍可在项目、知识库和研究页面完成。"
          action={<a className="sa-button sa-button--ghost" href="/projects">返回项目</a>}
        />
      </div>
    </WorkspaceLayout>
  )
}

export function DeveloperAccessGate({ children, title, description }) {
  const auth = useAuthStore()
  if (!auth.ready) return <AccessLoading />
  if (!isDeveloperRole(auth.user?.role)) {
    return <AccessDenied title={title} description={description} />
  }
  return children
}
