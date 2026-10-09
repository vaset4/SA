'use client'

import { useState } from 'react'

import { testLLMConnection } from '../../lib/api/llm-config'
import { getError } from '../../lib/study-assistant-workspace.mjs'

// 预设默认值：新建时的初始表单，重试规则默认贴合"不稳定免费上游"的经验值
const EMPTY_FORM = {
  name: '',
  base_url: '',
  model_name: '',
  api_key: '',
  temperature: 0,
  retry_rule: {
    requests_per_minute: 30,
    max_retries: 8,
    retry_backoff_seconds: 1.5,
    timeout_seconds: 240,
  },
}

// 把已有预设记录展开成可编辑表单结构；api_key 一律留空，由占位符提示"不改则保留"
function toFormState(profile) {
  if (!profile) return { ...EMPTY_FORM, retry_rule: { ...EMPTY_FORM.retry_rule } }
  return {
    name: profile.name || '',
    base_url: profile.base_url || '',
    model_name: profile.model_name || '',
    api_key: '',
    temperature: profile.temperature ?? 0,
    retry_rule: {
      requests_per_minute: profile.retry_rule?.requests_per_minute ?? 30,
      max_retries: profile.retry_rule?.max_retries ?? 8,
      retry_backoff_seconds: profile.retry_rule?.retry_backoff_seconds ?? 1.5,
      timeout_seconds: profile.retry_rule?.timeout_seconds ?? 240,
    },
  }
}

// LLM 预设新增/编辑表单：连接三要素 + 每供应商重试规则 + 即时测试连接
export function LLMProfileForm({ profile = null, onSubmit, onCancel }) {
  const isEdit = Boolean(profile)
  const [form, setForm] = useState(() => toFormState(profile))
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [testState, setTestState] = useState(null)

  const updateField = (field) => (event) =>
    setForm((current) => ({ ...current, [field]: event.target.value }))

  const updateRule = (field) => (event) =>
    setForm((current) => ({
      ...current,
      retry_rule: { ...current.retry_rule, [field]: event.target.value },
    }))

  // 组装提交载荷：数值字段转数字，避免表单字符串污染后端校验
  const buildPayload = () => ({
    name: form.name.trim(),
    base_url: form.base_url.trim(),
    model_name: form.model_name.trim(),
    api_key: form.api_key,
    temperature: Number(form.temperature) || 0,
    retry_rule: {
      requests_per_minute: Number(form.retry_rule.requests_per_minute) || 0,
      max_retries: Number(form.retry_rule.max_retries) || 0,
      retry_backoff_seconds: Number(form.retry_rule.retry_backoff_seconds) || 0,
      timeout_seconds: Number(form.retry_rule.timeout_seconds) || 240,
    },
  })

  const submit = async () => {
    if (!form.name.trim() || !form.base_url.trim() || !form.model_name.trim()) {
      setError('请填写名称、Base URL 和模型名')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      await onSubmit(buildPayload())
    } catch (reason) {
      setError(getError(reason, '保存失败'))
    } finally {
      setSubmitting(false)
    }
  }

  // 测试连接：body 必须用后端 schema 的 snake_case
  // 编辑且未改 Key 时传 profile_id 取库内真实密钥；新建或已填 Key 时传显式三要素
  const runTest = async () => {
    setTestState({ status: 'running', message: '正在测试连接…' })
    try {
      const result = await testLLMConnection(
        isEdit && !form.api_key.trim()
          ? {
              profile_id: profile.id,
              base_url: form.base_url,
              model_name: form.model_name,
            }
          : {
              base_url: form.base_url,
              model_name: form.model_name,
              api_key: form.api_key,
            },
      )
      setTestState({
        status: result.ok ? 'ok' : 'fail',
        message: `${result.message}${result.latency_ms ? `（${result.latency_ms}ms）` : ''}`,
      })
    } catch (reason) {
      setTestState({ status: 'fail', message: getError(reason, '测试失败') })
    }
  }

  return (
    <div className="sa-list-card" style={{ display: 'grid', gap: 12 }}>
      <h3 style={{ margin: 0, fontSize: 15 }}>{isEdit ? '编辑预设' : '新增预设'}</h3>

      <label style={{ display: 'grid', gap: 6 }}>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>预设名称</span>
        <input className="sa-input" placeholder="如 NVIDIA 免费" value={form.name} onChange={updateField('name')} />
      </label>

      <label style={{ display: 'grid', gap: 6 }}>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>Base URL</span>
        <input className="sa-input" placeholder="https://…/v1" value={form.base_url} onChange={updateField('base_url')} />
      </label>

      <label style={{ display: 'grid', gap: 6 }}>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>模型名</span>
        <input className="sa-input" placeholder="deepseek-ai/deepseek-v4-flash" value={form.model_name} onChange={updateField('model_name')} />
      </label>

      <label style={{ display: 'grid', gap: 6 }}>
        <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
          API Key{isEdit ? '（留空则保留原值）' : ''}
        </span>
        <input
          className="sa-input"
          type="password"
          placeholder={isEdit ? '••••••（不修改可留空）' : 'sk-…'}
          value={form.api_key}
          onChange={updateField('api_key')}
        />
      </label>

      <fieldset style={{ border: '1px solid var(--sa-border)', borderRadius: 8, padding: 12, margin: 0 }}>
        <legend style={{ padding: '0 6px', fontSize: 12, color: 'var(--sa-text-soft)' }}>重试与限流规则（按供应商脾气调）</legend>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
          <label style={{ display: 'grid', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>每分钟请求数 (RPM)</span>
            <input className="sa-input" type="number" min="0" value={form.retry_rule.requests_per_minute} onChange={updateRule('requests_per_minute')} />
          </label>
          <label style={{ display: 'grid', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>最大重试次数</span>
            <input className="sa-input" type="number" min="0" value={form.retry_rule.max_retries} onChange={updateRule('max_retries')} />
          </label>
          <label style={{ display: 'grid', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>重试等待 (秒)</span>
            <input className="sa-input" type="number" min="0" step="0.5" value={form.retry_rule.retry_backoff_seconds} onChange={updateRule('retry_backoff_seconds')} />
          </label>
          <label style={{ display: 'grid', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>超时 (秒)</span>
            <input className="sa-input" type="number" min="5" value={form.retry_rule.timeout_seconds} onChange={updateRule('timeout_seconds')} />
          </label>
        </div>
        <p className="sa-note" style={{ margin: '8px 0 0' }}>
          不稳定的免费上游：RPM 贴合额度、重试次数调高(如 8)、等待调短(如 1.5s)，靠多试扛抖动。
        </p>
      </fieldset>

      {testState ? (
        <p
          className="sa-note"
          style={{
            margin: 0,
            color: testState.status === 'ok' ? 'var(--sa-success)' : testState.status === 'fail' ? 'var(--sa-danger)' : 'var(--sa-text-soft)',
          }}
        >
          {testState.message}
        </p>
      ) : null}

      {error ? <p className="sa-note" style={{ margin: 0, color: 'var(--sa-danger)' }}>{error}</p> : null}

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
        <button type="button" className="sa-button sa-button--ghost" onClick={runTest} disabled={submitting}>
          测试连接
        </button>
        <button type="button" className="sa-button sa-button--ghost" onClick={onCancel} disabled={submitting}>
          取消
        </button>
        <button type="button" className="sa-button sa-button--primary" onClick={submit} disabled={submitting}>
          {submitting ? '保存中…' : '保存'}
        </button>
      </div>
    </div>
  )
}
