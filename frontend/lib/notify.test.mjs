import test from 'node:test'
import assert from 'node:assert/strict'

import {
  NOTIFY_DURATION_MS,
  emitNotify,
  setNotifySink,
} from './notify-core.mjs'

test('notify helpers send stable HeroUI toast payloads through one sink', () => {
  const calls = []
  setNotifySink((payload) => calls.push(payload))

  emitNotify({ title: 'Saved', description: 'Project updated', color: 'success' })
  emitNotify({ title: 'Failed', description: 'Network error', color: 'danger' })
  emitNotify({ title: 'Queued', color: 'primary' })
  setNotifySink(null)

  assert.deepEqual(calls, [
    { title: 'Saved', description: 'Project updated', color: 'success', timeout: NOTIFY_DURATION_MS },
    { title: 'Failed', description: 'Network error', color: 'danger', timeout: NOTIFY_DURATION_MS },
    { title: 'Queued', description: '', color: 'primary', timeout: NOTIFY_DURATION_MS },
  ])
})
