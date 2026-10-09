'use client'

import { useEffect, useState } from 'react'

import {
  listLLMProfiles,
  createLLMProfile,
  updateLLMProfile,
  deleteLLMProfile,
  activateLLMProfile,
} from '../../lib/api/llm-config'
import { getError } from '../../lib/study-assistant-workspace.mjs'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { EmptyState } from '../shared/EmptyState'
import { ErrorState } from '../shared/ErrorState'
import { LLMProfileCard } from './LLMProfileCard'
import { LLMProfileForm } from './LLMProfileForm'

// LLM 配置中心：多套上游预设的增删改查 + 一键热切换，切换后无需重启即时生效
export function LLMConfigCenter() {
  const [profiles, setProfiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState('')
  // editing: null=未打开表单；'new'=新增；其它=正在编辑的预设对象
  const [editing, setEditing] = useState(null)
  const [pendingDelete, setPendingDelete] = useState(null)

  // 拉取预设列表，作为页面首屏与每次增删改后的统一刷新入口
  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await listLLMProfiles()
      setProfiles(Array.isArray(data) ? data : [])
    } catch (reason) {
      setError(getError(reason, '加载预设失败'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  // 保存表单：按新增/编辑分派，成功后关闭表单并刷新列表
  // API 约定为位置参数：updateLLMProfile(profileId, payload)
  const handleSubmit = async (payload) => {
    if (editing === 'new') {
      await createLLMProfile(payload)
    } else {
      await updateLLMProfile(editing.id, payload)
    }
    setEditing(null)
    await refresh()
  }

  // 激活预设：触发后端写回 .env + 热重载，成功后刷新以更新"使用中"标记
  const handleActivate = async (profile) => {
    setBusyId(profile.id)
    setError('')
    try {
      await activateLLMProfile(profile.id)
      await refresh()
    } catch (reason) {
      setError(getError(reason, '切换失败'))
    } finally {
      setBusyId('')
    }
  }

  // 确认删除：正在使用的预设已在卡片层禁用，这里兜底处理后端返回的冲突提示
  const handleDelete = async () => {
    const target = pendingDelete
    setPendingDelete(null)
    if (!target) return
    setBusyId(target.id)
    setError('')
    try {
      await deleteLLMProfile(target.id)
      await refresh()
    } catch (reason) {
      setError(getError(reason, '删除失败'))
    } finally {
      setBusyId('')
    }
  }

  if (editing) {
    return (
      <LLMProfileForm
        profile={editing === 'new' ? null : editing}
        onSubmit={handleSubmit}
        onCancel={() => setEditing(null)}
      />
    )
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button type="button" className="sa-button sa-button--primary" onClick={() => setEditing('new')}>
          新增预设
        </button>
      </div>

      {error ? <ErrorState title="操作失败" description={error} onRetry={refresh} /> : null}

      {loading ? (
        <p className="sa-note">加载中…</p>
      ) : profiles.length === 0 ? (
        <EmptyState
          title="还没有 LLM 预设"
          description="新增一套上游供应商配置（地址、密钥、模型与重试规则），即可一键切换。"
        />
      ) : (
        profiles.map((profile) => (
          <LLMProfileCard
            key={profile.id}
            profile={profile}
            busy={busyId === profile.id}
            onActivate={() => handleActivate(profile)}
            onEdit={() => setEditing(profile)}
            onDelete={() => setPendingDelete(profile)}
          />
        ))
      )}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="删除预设"
        description={pendingDelete ? `确认删除「${pendingDelete.name}」？此操作不可撤销。` : ''}
        confirmLabel="删除"
        onConfirm={handleDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
