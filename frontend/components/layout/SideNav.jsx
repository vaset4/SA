'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { BRAND, WORKSPACE_PRIMARY_NAV, WORKSPACE_SECONDARY_NAV } from '../../config/brand'

function WorkspaceIcon({ icon, width = 20 }) {
  const common = {
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
  }

  switch (icon) {
    case 'solar:library-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="M5 5.5h3.5v13H5zM10.25 4.5h3.5v14h-3.5zM15.5 6.5H19v12h-3.5z" />
          <path {...common} d="M4 19.5h16" />
        </svg>
      )
    case 'solar:chat-round-dots-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="M7 18.5 4.5 20v-4.25A7.5 7.5 0 1 1 12 19.5H7Z" />
          <path {...common} d="M9 11.75h.01M12 11.75h.01M15 11.75h.01" />
        </svg>
      )
    case 'solar:microscope-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="m8.5 4 4 4-1.75 1.75-4-4zM9.75 9.25l3.5 3.5M13 12.5a4 4 0 1 1-5.66 5.66" />
          <path {...common} d="M13.5 18.5H20M11 20h9M16 8l3.5 3.5" />
        </svg>
      )
    case 'solar:document-search-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="M7 4.75h7l3.25 3.25V19a1.75 1.75 0 0 1-1.75 1.75H7A1.75 1.75 0 0 1 5.25 19V6.5A1.75 1.75 0 0 1 7 4.75Z" />
          <path {...common} d="M14 4.75V8h3.25" />
          <circle {...common} cx="11" cy="13" r="2.25" />
          <path {...common} d="m12.75 14.75 1.75 1.75" />
        </svg>
      )
    case 'solar:magnifer-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="11" cy="11" r="5.75" />
          <path {...common} d="m15.25 15.25 3.5 3.5" />
        </svg>
      )
    case 'solar:calendar-mark-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <rect {...common} x="4.5" y="5.5" width="15" height="14" rx="2.5" />
          <path {...common} d="M8 3.75v3.5M16 3.75v3.5M4.5 9.5h15" />
          <path {...common} d="m10.25 14 1.25 1.25L14.75 12" />
        </svg>
      )
    case 'solar:compass-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="12" cy="12" r="8" />
          <path {...common} d="m10 10 4-2-2 4-4 2 2-4Z" />
          <path {...common} d="M12 4v2M20 12h-2M12 20v-2M4 12h2" />
        </svg>
      )
    case 'solar:chart-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="M4.5 19.5h15" />
          <path {...common} d="M7.5 17v-5.5M12 17V8M16.5 17v-8.5" />
          <path {...common} d="M6 7.75 10.6 12l3.15-4.25L18 10" />
          <path {...common} d="m16.5 9.5 1.5.5-.5 1.5" />
        </svg>
      )
    case 'solar:graph-new-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="6.5" cy="16.5" r="2.25" />
          <circle {...common} cx="12" cy="7" r="2.25" />
          <circle {...common} cx="17.5" cy="14" r="2.25" />
          <path {...common} d="M8.25 15.1 10.3 8.4M13.9 8.25l2.1 3.5M8.4 17l6.7-2" />
        </svg>
      )
    case 'solar:folder-with-files-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <path {...common} d="M4.5 8.25h15a1.5 1.5 0 0 1 1.5 1.5v7.75a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-8a1.25 1.25 0 0 1 1.25-1.25Z" />
          <path {...common} d="M4 8V6.75A1.75 1.75 0 0 1 5.75 5h4.1l1.65 1.75H18" />
          <path {...common} d="M9 11.5h6M9 14.5h6" />
        </svg>
      )
    case 'solar:settings-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="12" cy="12" r="3.25" />
          <path {...common} d="M12 4.5v2M12 17.5v2M19.5 12h-2M6.5 12h-2M17.3 6.7l-1.4 1.4M8.1 15.9l-1.4 1.4M17.3 17.3l-1.4-1.4M8.1 8.1 6.7 6.7" />
        </svg>
      )
    case 'solar:radar-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="12" cy="12" r="8" />
          <circle {...common} cx="12" cy="12" r="4" />
          <circle {...common} cx="12" cy="12" r="1" />
          <path {...common} d="M12 4v8" />
        </svg>
      )
    case 'solar:square-alt-arrow-right-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <rect {...common} x="4.5" y="4.5" width="15" height="15" rx="3" />
          <path {...common} d="m10 8 4 4-4 4M14 12H9" />
        </svg>
      )
    case 'solar:square-alt-arrow-left-linear':
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <rect {...common} x="4.5" y="4.5" width="15" height="15" rx="3" />
          <path {...common} d="m14 8-4 4 4 4M10 12h5" />
        </svg>
      )
    default:
      return (
        <svg viewBox="0 0 24 24" width={width} height={width} aria-hidden="true">
          <circle {...common} cx="12" cy="12" r="7" />
        </svg>
      )
  }
}

function SideNavLink({ item, collapsed, active }) {
  return (
    <Link
      href={item.href}
      className="sa-side-nav__link"
      data-active={active ? 'true' : 'false'}
      aria-current={active ? 'page' : undefined}
      title={collapsed ? item.label : undefined}
    >
      <span className="sa-side-nav__icon">
        <WorkspaceIcon icon={item.icon} width={20} />
      </span>
      {!collapsed ? <span className="sa-side-nav__label">{item.label}</span> : null}
    </Link>
  )
}

function SideNavGroup({ item, collapsed, pathname }) {
  const children = Array.isArray(item.children) ? item.children : []
  const groupActive = children.some((entry) => entry.match.test(pathname))

  return (
    <div className="sa-side-nav__group" data-active={groupActive ? 'true' : 'false'}>
      <div className="sa-side-nav__group-head" title={collapsed ? item.label : undefined}>
        <span className="sa-side-nav__icon">
          <WorkspaceIcon icon={item.icon} width={20} />
        </span>
        {!collapsed ? <span className="sa-side-nav__label">{item.label}</span> : null}
      </div>
      <div className={`sa-side-nav__sublist${collapsed ? ' sa-side-nav__sublist--collapsed' : ''}`}>
        {children.map((entry) => (
          <SideNavLink
            key={entry.href}
            item={entry}
            collapsed={collapsed}
            active={entry.match.test(pathname)}
          />
        ))}
      </div>
    </div>
  )
}

export function SideNav({ collapsed = false, onToggle }) {
  const pathname = usePathname()

  return (
    <aside className={`sa-side-nav${collapsed ? ' sa-side-nav--collapsed' : ''}`}>
      <div className="sa-side-nav__header">
        <Link href="/" className="sa-side-nav__brand" title={BRAND.name}>
          <span className="sa-brand-mark">{BRAND.shortName}</span>
          {!collapsed ? (
            <span className="sa-side-nav__brand-copy">
              <strong>{BRAND.name}</strong>
              <span>{BRAND.tagline}</span>
            </span>
          ) : null}
        </Link>
        <button type="button" className="sa-side-nav__toggle" onClick={onToggle} aria-label={collapsed ? '展开导航' : '收起导航'}>
          <WorkspaceIcon
            icon={collapsed ? 'solar:square-alt-arrow-right-linear' : 'solar:square-alt-arrow-left-linear'}
            width={18}
          />
        </button>
      </div>

      <div className="sa-side-nav__section">
        <nav className="sa-side-nav__list" aria-label="Workspace Navigation">
          {WORKSPACE_PRIMARY_NAV.map((item) => (
            item.children ? (
              <SideNavGroup key={item.id || item.label} item={item} collapsed={collapsed} pathname={pathname} />
            ) : (
              <SideNavLink
                key={item.href}
                item={item}
                collapsed={collapsed}
                active={item.match.test(pathname)}
              />
            )
          ))}
        </nav>
      </div>

      <div className="sa-side-nav__footer">
        {!collapsed ? <p className="sa-side-nav__caption">系统</p> : null}
        <nav className="sa-side-nav__list" aria-label="System Navigation">
          {WORKSPACE_SECONDARY_NAV.map((item) => (
            <SideNavLink
              key={item.href}
              item={item}
              collapsed={collapsed}
              active={item.match.test(pathname)}
            />
          ))}
        </nav>
      </div>
    </aside>
  )
}
