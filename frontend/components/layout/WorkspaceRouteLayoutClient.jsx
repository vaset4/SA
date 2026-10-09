'use client'

import { useEffect } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { AppShell } from './AppShell'
import { StudyAssistantStoreProvider } from '../../stores/study-assistant-store-provider'
import { useAuthStore } from '../../stores/auth-store'
import { buildWorkspaceRedirectTarget } from '../../lib/workspace-routes'

export function WorkspaceRouteLayoutClient({ children }) {
  const auth = useAuthStore()
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  useEffect(() => {
    if (!auth.ready || auth.isAuthenticated) return
    const query = searchParams?.toString()
    const redirect = encodeURIComponent(buildWorkspaceRedirectTarget(pathname, query ? `?${query}` : ''))
    router.replace(`/login?redirect=${redirect}`)
  }, [auth.isAuthenticated, auth.ready, pathname, router, searchParams])

  if (!auth.ready || !auth.isAuthenticated) {
    return (
      <div className="sa-app-shell" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
        <div className="sa-panel-card" style={{ width: 'min(420px, calc(100vw - 32px))', padding: 24 }}>
          正在校验登录状态...
        </div>
      </div>
    )
  }

  return (
    <StudyAssistantStoreProvider>
      <AppShell>{children}</AppShell>
    </StudyAssistantStoreProvider>
  )
}
