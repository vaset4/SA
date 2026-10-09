'use client'

const DEFAULT_ROWS = 4
const DEFAULT_HEIGHT = 18

export function Skeleton({ rows = DEFAULT_ROWS, height = DEFAULT_HEIGHT, variant = 'block' } = {}) {
  const items = Array.from({ length: rows }, (_, index) => index)
  return (
    <div className={`sa-skeleton sa-skeleton--${variant}`} aria-hidden="true">
      {items.map((item) => (
        <span key={item} className="sa-skeleton__row" style={{ minHeight: height }} />
      ))}
    </div>
  )
}
