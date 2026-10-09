'use client'

export function WorkspaceLayout({
  sidebar = null,
  header = null,
  children,
  rail = null,
  focusMode = false,
  shellClassName = '',
  sidebarClassName = '',
  mainClassName = '',
  railClassName = '',
  contentClassName = '',
}) {
  const resolvedShellClassName = [
    'sa-shell-main',
    focusMode ? 'sa-shell-main--focus' : '',
    !focusMode && rail ? 'sa-shell-main--with-rail' : '',
    shellClassName,
  ].filter(Boolean).join(' ')

  return (
    <div className={resolvedShellClassName}>
      {!focusMode && (
        <aside className={['sa-panel-card sa-sidebar-card', sidebarClassName].filter(Boolean).join(' ')}>
          {sidebar}
        </aside>
      )}
      <section className={['sa-panel-card sa-main-card', mainClassName].filter(Boolean).join(' ')}>
        {header}
        <div className={['sa-main-scroll', contentClassName].filter(Boolean).join(' ')}>{children}</div>
      </section>
      {!focusMode && rail && (
        <aside className={['sa-panel-card sa-rail-card', railClassName].filter(Boolean).join(' ')}>
          {rail}
        </aside>
      )}
    </div>
  )
}
