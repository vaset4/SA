import { requestJson } from './core'

export const loginWithPassword = ({ username, password }) =>
  requestJson({
    path: '/api/v1/auth/login',
    skipAuth: true,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    },
  })

export const registerWithPassword = ({ username, password }) =>
  requestJson({
    path: '/api/v1/auth/register',
    skipAuth: true,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    },
  })

/**
 * 【认证】【恢复会话】读取当前账号，页面卸载时允许取消读取
 * @param {{signal?: AbortSignal}} options 本次读取的取消信号
 * @returns {Promise<object>} 服务端当前用户
 */
export const getCurrentUser = ({ signal } = {}) =>
  requestJson({ path: '/api/v1/auth/me', init: { signal } })

/**
 * 吊销当前账号的全部 refresh token。
 * @returns {Promise<{revoked: boolean}>} 服务端吊销确认结果
 */
export const logoutCurrentSession = () =>
  requestJson({
    path: '/api/v1/auth/logout',
    init: { method: 'POST' },
  })
