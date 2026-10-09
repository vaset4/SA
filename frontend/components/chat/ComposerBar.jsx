'use client'

import { Icon } from '@iconify/react'

export function ComposerBar({
  busy,
  restoring,
  draft,
  onDraftChange,
  onSubmit,
  onStop,
  projectName = '',
  paperCount = 0,
}) {
  const disabled = busy || restoring

  return (
    <div className="sa-composer">
      <div className="sa-composer__header">
        <div className="sa-stack" style={{ gap: 8 }}>
          <div className="sa-chip-list">
            {projectName ? <span className="sa-chip">项目 {projectName}</span> : null}
            <span className="sa-chip">论文范围 {paperCount}</span>
          </div>
        </div>
      </div>

      <textarea
        className="sa-textarea"
        value={draft}
        disabled={disabled}
        onChange={(event) => onDraftChange(event.target.value)}
        placeholder="例如：结合当前项目里的论文，先比较方法差异、证据强弱和还需要补哪些材料。"
        onKeyDown={(event) => {
          if (!(event.metaKey || event.ctrlKey) || event.key !== 'Enter') return
          event.preventDefault()
          onSubmit()
        }}
      />

      <div className="sa-composer__actions">
        <div className="sa-note">{disabled ? '处理中' : 'Ctrl/Cmd + Enter'}</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {busy ? (
            <button
              type="button"
              className="sa-button sa-button--ghost"
              onClick={onStop}
              aria-label="停止当前研究"
            >
              <Icon icon="lucide:square" width={15} aria-hidden="true" />
              停止
            </button>
          ) : (
            <button
              type="button"
              className="sa-button sa-button--primary"
              onClick={onSubmit}
              disabled={disabled}
            >
              <Icon icon="lucide:send" width={15} aria-hidden="true" />
              发送问题
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
