/**
 * 认证会话边界行为测试。
 *
 * @author xxx
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'
import { jsonResponse, loadAuthRequestModules } from './auth-test-support.mjs'
import './auth-session-isolation.test.mjs'

import { normalizeRefreshSession } from './auth-session.mjs'

test('refresh 响应完整覆盖旧 token', () => {
  const session = normalizeRefreshSession({
    access_token: 'new-access',
    refresh_token: 'new-refresh',
    token_type: 'bearer',
    expires_in: 3600,
  })

  assert.equal(session.accessToken, 'new-access')
  assert.equal(session.refreshToken, 'new-refresh')
})

test('refresh 响应缺 token 时显式失败', () => {
  assert.throws(
    () => normalizeRefreshSession({ access_token: 'new-access' }),
    /缺少 token/,
  )
})

test('auth client 不再把旧 session 浅合并回刷新响应', () => {
  const source = fs.readFileSync(new URL('./auth-client.js', import.meta.url), 'utf8')

  assert.match(source, /setAuthSession\(normalizeRefreshSession\(payload\),/)
  assert.doesNotMatch(
    source,
    /refreshToken: payload\?\.refresh_token \|\| sessionCache\.refreshToken/,
  )
})

test('并发 401 只刷新一次，并使用新 access token 重试原请求', async () => {
  const calls = []
  let refreshCalls = 0
  const fetchImplementation = async (url, init = {}) => {
    const authorization = new Headers(init.headers).get('Authorization')
    calls.push({ url, authorization, body: init.body })
    if (url.endsWith('/api/v1/auth/refresh')) {
      refreshCalls += 1
      return jsonResponse({
        access_token: 'new-access',
        refresh_token: 'new-refresh',
        token_type: 'bearer',
        expires_in: 3600,
        user: { user_id: 'user-1' },
      })
    }
    if (authorization === 'Bearer old-access') {
      return jsonResponse({ detail: 'expired' }, 401)
    }
    return jsonResponse({ ok: true })
  }
  const { authClient, apiCore } = await loadAuthRequestModules(fetchImplementation)
  authClient.setAuthSession({
    accessToken: 'old-access',
    refreshToken: 'old-refresh',
    user: { user_id: 'user-1' },
  })

  const results = await Promise.all([
    apiCore.requestJson({ path: '/protected/a' }),
    apiCore.requestJson({ path: '/protected/b' }),
  ])

  assert.deepEqual(results.map(({ ok }) => ok), [true, true])
  assert.equal(refreshCalls, 1)
  assert.equal(authClient.getAccessToken(), 'new-access')
  const resourceCalls = calls.filter(({ url }) => !url.endsWith('/api/v1/auth/refresh'))
  for (const path of ['/protected/a', '/protected/b']) {
    const headers = resourceCalls
      .filter(({ url }) => url.endsWith(path))
      .map(({ authorization }) => authorization)
    assert.deepEqual(headers, ['Bearer old-access', 'Bearer new-access'])
  }
  const refreshCall = calls.find(({ url }) => url.endsWith('/api/v1/auth/refresh'))
  assert.equal(JSON.parse(refreshCall.body).refresh_token, 'old-refresh')
})

test('远端 logout 失败仍清本地 session，并把错误留给登录页', () => {
  const storeSource = fs.readFileSync(new URL('../stores/auth-store.js', import.meta.url), 'utf8')
  const loginSource = fs.readFileSync(
    new URL('../components/auth/LoginPageClient.jsx', import.meta.url),
    'utf8',
  )

  assert.match(storeSource, /clearAuthSession\(\{ expectedContext: context, redirect: remoteError \? false : redirect \}\)/)
  assert.match(storeSource, /error: remoteError/)
  assert.match(loginSource, /localError \|\| auth\.error/)
})

import './auth-stream-isolation.test.mjs'
