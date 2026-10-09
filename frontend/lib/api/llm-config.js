import { requestJson } from './core'

// LLM 供应商预设：列出全部（Key 已脱敏），当前激活项带 is_active
export const listLLMProfiles = () =>
  requestJson({ path: '/api/v1/llm/profiles' })

// 新增一套预设
export const createLLMProfile = (payload) =>
  requestJson({
    path: '/api/v1/llm/profiles',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  })

// 编辑预设；api_key 留空表示保留原值
export const updateLLMProfile = (profileId, payload) =>
  requestJson({
    path: `/api/v1/llm/profiles/${encodeURIComponent(profileId)}`,
    init: {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  })

// 删除预设；不能删当前激活项
export const deleteLLMProfile = (profileId) =>
  requestJson({
    path: `/api/v1/llm/profiles/${encodeURIComponent(profileId)}`,
    init: { method: 'DELETE' },
  })

// 激活预设：后端写回 .env + 热重载，运行时立即切换上游
export const activateLLMProfile = (profileId) =>
  requestJson({
    path: `/api/v1/llm/profiles/${encodeURIComponent(profileId)}/activate`,
    init: { method: 'POST' },
  })

// 测试连接：显式传 base_url/model/key 测未保存的草稿，或传 profile_id 用已存 Key 测试
export const testLLMConnection = (payload) =>
  requestJson({
    path: '/api/v1/llm/profiles/test',
    timeoutMs: 25_000,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  })
