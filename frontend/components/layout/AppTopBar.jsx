'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@heroui/react'
import { Icon } from '@iconify/react'
import { useTheme } from 'next-themes'
import { useCallback, useEffect, useState } from 'react'
import { findWorkspaceNavItem } from '../../config/brand'
import { getSystemHealth } from '../../lib/api/system'
import { useAuthStore } from '../../stores/auth-store'
import { useProjectStore } from '../../stores/project-store'
import { useUIStore } from '../../stores/ui-store'

function HealthDot() {
  const [ok, setOk] = useState(null)

  const check = useCallback(async () => {
    try {
      const data = await getSystemHealth()
      setOk(data?.status === 'healthy')
    } catch (error) {
      console.warn('topbar health check failed', error)
      setOk(false)
    }
  }, [])

  useEffect(() => { check(); const t = setInterval(check, 30000); return () => clearInterval(t) }, [check])

  if (ok === null) return null
  return (
    <span
      title={ok ? '系统正常' : '系统异常'}
      style={{ width: 8, height: 8, borderRadius: '50%', background: ok ? 'var(--sa-success)' : 'var(--sa-danger)', flexShrink: 0 }}
    />
  )
}

export function AppTopBar({ navCollapsed = false, onToggleNav }) {
  const pathname = usePathname()
  const { theme, resolvedTheme, setTheme } = useTheme()
  const auth = useAuthStore()
  const projectStore = useProjectStore()
  const ui = useUIStore()
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const toggleTheme = () => {
    if (theme === 'system') { setTheme(resolvedTheme === 'dark' ? 'light' : 'dark'); return }
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }

  const currentNav = findWorkspaceNavItem(pathname)
  const unreadCount = (ui.notifications || []).filter((item) => !ui.readIds.includes(item.log_id)).length
  const username = auth.user?.username || 'researcher'
  const userMark = username.slice(0, 1).toUpperCase()
  const currentProjectLabel = projectStore.activeProject?.name || ''

  return (
    <header className="sa-shell-topbar">
      <div className="sa-topbar-leading">
        <Button isIconOnly variant="light" onPress={onToggleNav} aria-label={navCollapsed ? '展开导航' : '收起导航'}>
          <Icon icon={navCollapsed ? 'solar:hamburger-menu-linear' : 'solar:sidebar-minimalistic-linear'} width={18} />
        </Button>

        <div className="sa-topbar-title">
          <div className="sa-topbar-title__name">{currentNav?.label || '工作台'}</div>
          {currentProjectLabel ? <div className="sa-topbar-context">{currentProjectLabel}</div> : null}
        </div>
      </div>

      <div className="sa-topbar-actions">
        <Link href="/credits" style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>第三方声明</Link>
        <HealthDot />

        <button type="button" className="sa-topbar-notice" aria-label="通知">
          <Icon icon="solar:bell-linear" width={18} />
          {unreadCount ? <span className="sa-topbar-notice__badge">{unreadCount > 9 ? '9+' : unreadCount}</span> : null}
        </button>

        <Button isIconOnly variant="light" onPress={toggleTheme} isDisabled={!mounted} aria-label="切换主题">
          <Icon icon={mounted && resolvedTheme === 'dark' ? 'solar:sun-bold' : 'solar:moon-bold'} width={18} />
        </Button>

        <button type="button" className="sa-topbar-avatar" aria-label="用户中心" onClick={() => auth.logout()}>
          <span>{userMark}</span>
        </button>
      </div>
    </header>
  )
}
