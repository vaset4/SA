'use client'

import { useUIStore } from '../../stores/ui-store'
import { AppTopBar } from './AppTopBar'
import { SideNav } from './SideNav'

export function AppShell({ children }) {
  const ui = useUIStore()
  const collapsed = Boolean(ui.railCollapsed)

  const toggleNav = () => {
    ui.setRailCollapsed(!collapsed)
  }

  return (
    <div className={`sa-app-shell sa-workspace-shell${collapsed ? ' sa-workspace-shell--collapsed' : ''}`}>
      <SideNav collapsed={collapsed} onToggle={toggleNav} />
      <div className="sa-workspace-frame">
        <AppTopBar navCollapsed={collapsed} onToggleNav={toggleNav} />
        <main className="sa-workspace-content">{children}</main>
      </div>
    </div>
  )
}
