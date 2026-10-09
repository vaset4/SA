'use client'

export function PageHeader({
  eyebrow = '',
  title,
  description = '',
  actions = null,
  status = null,
  variant = 'workspace',
}) {
  return (
    <header className={`sa-page-header sa-page-header--${variant}`}>
      <div className="sa-page-header__meta">
        {eyebrow ? <p className="sa-page-header__eyebrow">{eyebrow}</p> : null}
        <h1 className="sa-page-header__title">{title}</h1>
        {description ? (
          <p className="sa-page-header__description">{description}</p>
        ) : null}
      </div>
      <div className="sa-page-header__side">
        {status ? <div className="sa-page-header__status-row">{status}</div> : null}
        {actions ? <div className="sa-page-header__actions">{actions}</div> : null}
      </div>
    </header>
  )
}
