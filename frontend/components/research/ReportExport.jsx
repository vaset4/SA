'use client'

import { Button } from '@heroui/react'

function buildFilename(filename, defaultExtension = 'md') {
  const trimmed = String(filename || '').trim()
  if (!trimmed) return `research-report.${defaultExtension}`
  return /\.[a-z0-9]+$/i.test(trimmed) ? trimmed : `${trimmed}.${defaultExtension}`
}

export function ReportExport({
  content = '',
  filename = 'research-report.md',
  label = '导出 Markdown',
  variant = 'flat',
  color = 'primary',
  size = 'sm',
  mimeType = 'text/markdown;charset=utf-8',
  defaultExtension = 'md',
}) {
  const disabled = !String(content || '').trim()

  const handleExport = () => {
    if (disabled) return
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = buildFilename(filename, defaultExtension)
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Button
      size={size}
      color={color}
      variant={variant}
      isDisabled={disabled}
      onPress={handleExport}
    >
      {label}
    </Button>
  )
}
