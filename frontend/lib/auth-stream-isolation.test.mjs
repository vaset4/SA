import test from 'node:test'
import assert from 'node:assert/strict'
import { deferred, jsonResponse, loadAuthRequestModules, loadClientModule, sessionPayload } from './auth-test-support.mjs'

test('旧账号 SSE 的迟到 401 不能刷新或重放到新账号', async () => {
  const response = deferred()
  let calls = 0
  const modules = await loadAuthRequestModules(() => { calls += 1; return response.promise })
  modules.authClient.setAuthSession(sessionPayload('a'))
  const stream = await loadClientModule(modules, './study-assistant-stream.js')
  const result = stream.runCrossPaperStream({ query: 'q', idempotencyKey: 'original' })
  const rejection = assert.rejects(result, { code: 'AUTH_SESSION_CHANGED' })
  modules.authClient.setAuthSession(sessionPayload('b'))
  response.resolve(jsonResponse({ detail: 'expired' }, 401))
  await rejection
  assert.equal(calls, 1)
  assert.equal(modules.authClient.getAuthSession().user.user_id, 'b')
})

test('账号切换会停止读取旧流，晚到 token 不能送回页面', async () => {
  let control
  const body = new ReadableStream({ start(controller) { control = controller } })
  const modules = await loadAuthRequestModules(async () => new Response(body, { headers: { 'X-RAG-Run-ID': 'run-a' } }),
    { globals: { TextDecoder, Uint8Array } })
  modules.authClient.setAuthSession(sessionPayload('a'))
  const stream = await loadClientModule(modules, './study-assistant-stream.js')
  const opened = deferred()
  const tokens = []
  const result = stream.runCrossPaperStream({ query: 'q', onRun: () => opened.resolve(), onToken: (text) => tokens.push(text) })
  const rejection = assert.rejects(result, { code: 'AUTH_SESSION_CHANGED' })
  await opened.promise
  modules.authClient.setAuthSession(sessionPayload('b'))
  control.enqueue(new TextEncoder().encode('data: {"type":"token","content":"private a"}\n\n'))
  control.close()
  await rejection
  assert.deepEqual(tokens, [])
})
