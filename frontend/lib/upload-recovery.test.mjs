import test from 'node:test'
import assert from 'node:assert/strict'
import { acceptOrRecoverUpload, observeUploadTask, uploadStage } from './upload-task-flow.mjs'

test('上传响应丢失时查回同一意图，不盲目重发原文件', async () => {
  const requests = []
  const request = { requestId: 'stable-intent', signal: new AbortController().signal }
  const task = { task_id: 'original-task', status: 'queued' }
  const api = { createUploadTask: async (value) => { requests.push(value); throw new Error('lost response') },
    getUploadReceipt: async (value) => { requests.push(value); return task } }
  assert.equal(await acceptOrRecoverUpload({ api, request }), task)
  assert.equal(requests.length, 2)
  assert.equal(requests[0].requestId, requests[1].requestId)
})

test('暂未查到受理不能当作失败后新建上传', async () => {
  const api = { createUploadTask: async () => { throw new Error('lost response') },
    getUploadReceipt: async () => { throw Object.assign(new Error('not yet accepted'), { status: 404 }) } }
  await assert.rejects(acceptOrRecoverUpload({ api, request: { requestId: 'same' } }), { code: 'UPLOAD_RECEIPT_PENDING' })
})

test('暂停是可恢复终态，页面观察不会触发继续或无限轮询', async () => {
  const task = { task_id: 'original', status: 'paused', error: 'quota exhausted' }
  const seen = []
  const api = { getTask: () => assert.fail('paused task must not be polled') }
  assert.equal(await observeUploadTask({ task, api, signal: new AbortController().signal, onTask: (value) => seen.push(value) }), task)
  assert.deepEqual(seen, [task])
})

test('后来删除的论文不能因历史任务曾成功而显示可检索', () => {
  assert.equal(uploadStage({ status: 'completed', result: { searchable: true }, upload: { searchable: false } }), 'completed')
})
