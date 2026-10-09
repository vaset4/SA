'use client'

const COLOR_MAP = {
  neutral: {
    bg: 'var(--sa-surface)',
    text: 'var(--sa-text-soft)',
    border: 'var(--sa-border)',
  },
  success: {
    bg: 'color-mix(in srgb, var(--sa-success) 16%, var(--sa-surface-soft))',
    text: 'var(--sa-success)',
    border: 'color-mix(in srgb, var(--sa-success) 28%, var(--sa-border))',
  },
  warning: {
    bg: 'color-mix(in srgb, #f59e0b 12%, var(--sa-surface-soft))',
    text: '#b45309',
    border: 'color-mix(in srgb, #f59e0b 24%, var(--sa-border))',
  },
  danger: {
    bg: 'color-mix(in srgb, var(--sa-danger) 12%, var(--sa-surface-soft))',
    text: 'var(--sa-danger)',
    border: 'color-mix(in srgb, var(--sa-danger) 24%, var(--sa-border))',
  },
}

export function StatusBadge({ tone = 'neutral', children }) {
  const palette = COLOR_MAP[tone] || COLOR_MAP.neutral
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        height: 26,
        padding: '0 10px',
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 600,
        border: `1px solid ${palette.border}`,
        color: palette.text,
        background: palette.bg,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  )
}
