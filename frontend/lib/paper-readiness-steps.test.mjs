import test from 'node:test'
import assert from 'node:assert/strict'

import { buildPaperReadinessSteps } from './paper-readiness-steps.mjs'

test('paper readiness distinguishes ready, pending, failed and unknown states', () => {
  assert.deepEqual(
    buildPaperReadinessSteps({ abstract: 'a', analysis_status: 'failed', is_vectorized: false }).map((item) => item.status),
    ['ready', 'unknown', 'failed', 'pending'],
  )
  assert.deepEqual(
    buildPaperReadinessSteps({ pdf_url: 'x', analysis_status: 'running', is_vectorized: true, index_batch_id: 'verified-batch' }).map((item) => item.status),
    ['ready', 'ready', 'running', 'ready'],
  )
  assert.deepEqual(
    buildPaperReadinessSteps({ analysis: { tldr: 'done' } }).map((item) => item.status),
    ['pending', 'unknown', 'ready', 'unknown'],
  )
})

test('verified publication is usable while optional analysis has not started', () => {
  const statuses = buildPaperReadinessSteps({ pdf_path: 'source.pdf', index_batch_id: 'verified-batch',
    is_vectorized: true, analysis_status: 'not_started' }).map((item) => item.status)
  assert.deepEqual(statuses, ['ready', 'ready', 'pending', 'ready'])
})

test('unverified legacy flags and deleting papers do not advertise retrieval readiness', () => {
  for (const paper of [
    { is_vectorized: true, analysis_status: 'completed' },
    { is_vectorized: true, index_batch_id: 'old-batch', deletion_status: 'deleting' },
  ]) {
    const steps = buildPaperReadinessSteps(paper)
    assert.equal(steps.find((step) => step.key === 'retrieval').status, 'pending')
    assert.equal(steps.find((step) => step.key === 'parse').status, 'unknown')
  }
})
