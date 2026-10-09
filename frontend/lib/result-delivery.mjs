// 【结果交付】【当前版本】研究页面与聊天共用已提交结果规则，作者：xxx
import { MESSAGE_EXTRA_KEYS } from './study-assistant-workspace.mjs'

const DELIVERY_LABELS = {
  complete: { tone: 'success', label: '回答完整' },
  partial: { tone: 'warning', label: '部分回答' },
  insufficient: { tone: 'warning', label: '资料不足' },
  failed: { tone: 'danger', label: '交付失败' },
}

/** @param {object|null} report 后端交付报告 @returns {object|null} 交付标签，历史缺报告不补造 */
export function describeDelivery(report) {
  if (!report) return null
  return DELIVERY_LABELS[report.status] || { tone: 'danger', label: '交付状态异常' }
}

/** @param {object} result 后端结果 @returns {object|null} 独立保存状态及真实恢复能力 */
export function describePersistence(result = {}) {
  if (!result.persistence_status) return null
  if (result.persistence_status === 'saved') return { tone: 'neutral', label: '已保存', detail: '', canRetry: false }
  const pending = result.persistence_status === 'pending'
  return {
    tone: 'danger', label: pending ? '答案尚未保存' : '结果未持久化',
    detail: result.persistence_error || (pending ? '可重试保存同一份答案' : '关闭页面后可能无法恢复本次正文'),
    canRetry: pending && result.can_retry_save === true,
  }
}

/** @param {object} session 会话查询结果 @returns {object} 已提交结果或明确的历史投影 */
export function sessionDeliveryValues(session) {
  if (session?.current_result) return session.current_result
  if (session?.current_run_id || session?.result_run_id) {
    if (session.status === 'awaiting_clarification') {
      return { clarification_question: session.values?.clarification_question || '', awaiting_clarification: true }
    }
    return {}
  }
  return session?.values || session || {}
}

/** @param {object} options 目标运行及补丁 @returns {void} 错误目标或缺版本明确拒绝 */
export function assertDeliveryPatch({ runId, patch }) {
  if (!patch || (patch.target_run_id || patch.run_id) !== runId) throw new Error('保存结果与当前回答不属于同一次研究')
  const unsaved = ['pending', 'not_staged'].includes(patch.persistence_status)
  if (!Number.isInteger(patch.delivery_revision) || patch.delivery_revision < (unsaved ? 0 : 1)) throw new Error('保存结果缺少有效版本')
  if (typeof patch.content !== 'string') throw new Error('保存结果缺少正文')
}

/** @param {object[]} messages 当前消息 @param {object} options 目标消息和完整补丁 @returns {object[]} 不回退版本的消息列表 */
export function applyDeliveryPatch(messages, { messageId, runId, patch }) {
  assertDeliveryPatch({ runId, patch })
  return messages.map((message) => {
    if (message.id !== messageId || message.run_id !== runId) return message
    const unsaved = ['pending', 'not_staged'].includes(patch.persistence_status)
    if (Number.isInteger(message.delivery_revision)
      && (patch.delivery_revision < message.delivery_revision || (!unsaved && patch.delivery_revision === message.delivery_revision))) return message
    // 1. 新产物整体替换，旧回合的引用、报告和候选不能靠缺字段残留
    const stable = Object.fromEntries(Object.entries(message).filter(([key]) => !MESSAGE_EXTRA_KEYS.includes(key)))
    return { ...stable, ...patch, id: message.id, role: 'assistant', pending: false, typewriter: false,
      renderMode: patch.render_mode || patch.renderMode || message.renderMode }
  })
}

/** @param {object} result 操作回执 @returns {object} 已提交消息或明确未保存的本地正文预览 */
export function deliveryResponsePatch(result) {
  if (result.message_patch) return result.message_patch
  const response = result.response
  if (['pending', 'not_staged'].includes(response?.persistence_status) && typeof response.answer === 'string') {
    return { ...response, content: response.answer, render_mode: 'markdown' }
  }
  throw new Error(result.operation?.error || '该操作尚未提供可应用的结果，请刷新运行状态')
}

/** @param {object} result 当前提交结果 @returns {boolean} 是否仍需跟进候选 */
export function hasPendingCandidates(result) {
  if (['pending', 'not_staged'].includes(result?.persistence_status)) return false
  return result?.memory_writeback?.status === 'pending'
    || result?.memory_writeback?.reason_codes?.includes('awaiting_result_commit') === true
}

/** @param {object[]} messages 当前消息 @param {object} result 权威当前结果 @returns {object[]} 按版本更新的消息 */
export function applyCurrentResult(messages, result) {
  return messages.map((message) => {
    if (message.run_id !== result.run_id || result.delivery_revision < message.delivery_revision) return message
    if (result.delivery_revision === message.delivery_revision) {
      return { ...message, memory_writeback: result.memory_writeback, execution_stopped_at: result.execution_stopped_at }
    }
    return applyDeliveryPatch([message], { messageId: message.id, runId: result.run_id,
      patch: { ...result, content: result.answer, render_mode: 'markdown' } })[0]
  })
}

/** @param {object} result 当前结果 @returns {string} 停止后仍未定性的可恢复异常 */
export function candidateRecoveryError(result) {
  const candidate = result?.memory_writeback
  const operation = result?.operation_run_id || result?.producer_run_id || result?.run_id
  return hasPendingCandidates(result) && candidate.processing_source === 'delivery_run'
    && candidate.processing_run_id === operation && candidate.processing_run_id && result.execution_stopped_at
    ? '候选状态恢复异常：执行已停止，但候选处理尚无结果，可重新查询' : ''
}
