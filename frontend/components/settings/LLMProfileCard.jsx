'use client'

import { StatusBadge } from '../shared/StatusBadge'

// 单个 LLM 预设卡片：展示连接信息与重试规则，提供激活/编辑/删除操作
export function LLMProfileCard({ profile, busy = false, onActivate, onEdit, onDelete }) {
  const rule = profile.retry_rule || {}
  return (
    <div
      className="sa-list-card"
      style={{
        display: 'grid',
        gap: 8,
        borderColor: profile.is_active ? 'var(--sa-success)' : undefined,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <strong style={{ fontSize: 15 }}>{profile.name}</strong>
        {profile.is_active ? <StatusBadge tone="success">使用中</StatusBadge> : null}
      </div>

      <div className="sa-note" style={{ display: 'grid', gap: 2 }}>
        <span>模型：{profile.model_name}</span>
        <span>地址：{profile.base_url}</span>
        <span>密钥：{profile.api_key_masked || '（未设置）'}</span>
      </div>

      <div className="sa-note" style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <span>RPM {rule.requests_per_minute ?? 0}</span>
        <span>重试 {rule.max_retries ?? 0} 次</span>
        <span>等待 {rule.retry_backoff_seconds ?? 0}s</span>
        <span>超时 {rule.timeout_seconds ?? 0}s</span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
        {!profile.is_active ? (
          <button type="button" className="sa-button sa-button--primary" onClick={onActivate} disabled={busy}>
            启用
          </button>
        ) : null}
        <button type="button" className="sa-button sa-button--ghost" onClick={onEdit} disabled={busy}>
          编辑
        </button>
        <button
          type="button"
          className="sa-button sa-button--ghost"
          onClick={onDelete}
          disabled={busy || profile.is_active}
          title={profile.is_active ? '正在使用的预设不能删除' : undefined}
        >
          删除
        </button>
      </div>
    </div>
  )
}
