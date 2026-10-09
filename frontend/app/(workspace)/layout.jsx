import { Suspense } from 'react'
import { WorkspaceRouteLayoutClient } from '../../components/layout/WorkspaceRouteLayoutClient'

function WorkspaceRouteLayoutFallback() {
  return (
    <div className="sa-app-shell" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
      <div className="sa-panel-card" style={{ width: 'min(420px, calc(100vw - 32px))', padding: 24 }}>
        正在加载工作台...
      </div>
    </div>
  )
}

export default function WorkspaceRouteLayout({ children }) {
  return (
    <Suspense fallback={<WorkspaceRouteLayoutFallback />}>
      <WorkspaceRouteLayoutClient>{children}</WorkspaceRouteLayoutClient>
    </Suspense>
  )
}
