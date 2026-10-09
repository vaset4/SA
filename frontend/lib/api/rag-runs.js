import { requestJson } from './core'
import { createId } from '../study-assistant-workspace.mjs'

const FINAL_RETRY_TIMEOUT_MS = 600_000

export const getRagRun = ({ runId, authContext, signal }) =>
  requestJson({ path: `/api/v1/rag-runs/${encodeURIComponent(runId)}`, authContext, init: { signal } })

export const cancelRagRun = ({ runId }) =>
  requestJson({
    path: `/api/v1/rag-runs/${encodeURIComponent(runId)}/cancel`,
    init: { method: 'POST' },
  })

export const getLatestConversationRagRun = ({ conversationId }) =>
  requestJson({
    path: `/api/v1/conversations/${encodeURIComponent(conversationId)}/rag-runs/latest`,
  }).then((data) => data?.run || null)

/** @param {object} request 原回答目标及同次请求标识 @returns {Promise<object>} 最终重试真实交付 */
export const retryRagRunFinalAnswer = ({ conversationId, runId, idempotencyKey = createId('final') }) =>
  requestJson({
    path: `/api/v1/conversations/${encodeURIComponent(conversationId)}/rag-runs/${encodeURIComponent(runId)}/retry-final-answer`,
    timeoutMs: FINAL_RETRY_TIMEOUT_MS,
    init: {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
    },
  })

/** @param {object} request 已暂存结果及同次请求标识 @returns {Promise<object>} 不重新生成的补存结果 */
export const retryRagRunSave = ({ runId, idempotencyKey = createId('save') }) =>
  requestJson({
    path: `/api/v1/rag-runs/${encodeURIComponent(runId)}/retry-save`,
    init: { method: 'POST', headers: { 'Idempotency-Key': idempotencyKey } },
  })

/** @param {object} request 已提交答案目标 @returns {Promise<object>} 同版本候选处理事实 */
export const retryRagRunMemoryWriteback = ({ runId }) =>
  requestJson({ path: `/api/v1/rag-runs/${encodeURIComponent(runId)}/retry-memory-writeback`, init: { method: 'POST' } })
