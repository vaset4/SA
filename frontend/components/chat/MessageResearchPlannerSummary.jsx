'use client'

import { useState } from 'react'
import { Icon } from '@iconify/react'

const PLANNER_SUCCESS_STATUSES = new Set(['evidence_found', 'inferred'])
const PLANNER_BLOCKED_STATUSES = new Set(['blocked'])
const PLANNER_INSUFFICIENT_STATUSES = new Set(['insufficient_evidence', 'external_needed'])

const PLANNER_STATUS_LABELS = {
  ok: '完成',
  complete: '完成',
  partial: '部分完成',
  failed: '失败',
  insufficient: '证据不足',
  skipped: '已跳过',
  analyzed: '已规划',
  validated: '已校验',
  executed: '已执行',
}

function hasObjectFields(raw) {
  return raw && typeof raw === 'object' && Object.keys(raw).length > 0
}

function textValue(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return `${value}`
  return ''
}

function numericMetric(value, fallback = 0) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

function formatPlannerStatus(status) {
  const normalized = `${status || ''}`.trim().toLowerCase()
  return PLANNER_STATUS_LABELS[normalized] || normalized || '未标注'
}

function countPlannerClaims(results) {
  return results.reduce((total, result) => (
    total + (Array.isArray(result?.claims) ? result.claims.length : 0)
  ), 0)
}

function countPlannerStatuses(results, allowed) {
  return results.filter((result) => allowed.has(`${result?.status || ''}`)).length
}

export function hasPlannerSummary(message, plan) {
  const runtimeChain = `${message?.runtime_chain || message?.trace_summary?.runtime_chain || ''}`.trim()
  return runtimeChain === 'planner' || Boolean(plan)
}

/** @param {object} message 消息 @param {object} plan 计划 @param {Array} results 单元结果 @returns {object|null} 取证与交付分别表达的摘要 */
function buildPlannerSummary(message, plan, results) {
  if (!hasPlannerSummary(message, plan)) return null
  const sourceStatus = hasObjectFields(message?.source_status)
    ? message.source_status
    : (message?.trace_summary?.source_status || {})
  const trace = hasObjectFields(message?.trace_summary) ? message.trace_summary : {}
  const units = Array.isArray(plan?.units) ? plan.units : []
  const evidenceRecords = Array.isArray(message?.evidence_records) ? message.evidence_records : []
  const errors = Array.isArray(message?.planner_final_validation_errors) ? message.planner_final_validation_errors : []
  return {
    total: numericMetric(trace.planner_units_total, units.length),
    success: numericMetric(trace.planner_units_success, countPlannerStatuses(results, PLANNER_SUCCESS_STATUSES)),
    blocked: numericMetric(trace.planner_units_blocked, countPlannerStatuses(results, PLANNER_BLOCKED_STATUSES)),
    insufficient: numericMetric(trace.planner_units_insufficient, countPlannerStatuses(results, PLANNER_INSUFFICIENT_STATUSES)),
    evidenceCount: numericMetric(trace.planner_evidence_count, evidenceRecords.length),
    claimCount: numericMetric(trace.planner_claim_count, countPlannerClaims(results)),
    plannerStatus: sourceStatus.planner || trace.planner_status || '',
    finalStatus: message?.final_delivery_report?.status || sourceStatus.planner_final || trace.planner_final_status || '',
    finalStatusLabel: message?.final_delivery_report?.status === 'insufficient' ? '内容不足' : '',
    acceptedBlocks: message?.final_delivery_report
      ? message.final_delivery_report.valid_business_block_ids?.length || 0
      : numericMetric(trace.planner_final_accepted_block_count, message?.planner_final_blocks?.length || 0),
    rejectedBlocks: numericMetric(trace.planner_final_rejected_block_count, errors.length),
    finalReason: textValue(message?.source_status_reasons?.planner_final || trace?.source_status_reasons?.planner_final),
    errorCodes: [...new Set(errors.map((error) => textValue(error?.code)).filter(Boolean))],
  }
}

/** @param {object} message 当前消息 @returns {boolean} 当前结果是否支持最终回答重试 */
export function canRetryPlannerFinalAnswer(message) {
  // 取证 ok/partial，且最终未完全成功（failed/skipped/partial）时可只重试最终回答
  const sourceStatus = hasObjectFields(message?.source_status)
    ? message.source_status
    : (message?.trace_summary?.source_status || {})
  const planner = `${sourceStatus.planner || ''}`.trim()
  const plannerFinal = `${sourceStatus.planner_final || ''}`.trim()
  const plannerOk = ['ok', 'partial', 'insufficient', 'no_evidence'].includes(planner)
  const finalNeedsRetry = !plannerFinal || plannerFinal === 'failed' || plannerFinal === 'skipped' || plannerFinal === 'partial'
  return plannerOk && finalNeedsRetry && plannerFinal !== 'ok' && Boolean(message?.run_id)
}

export function MessageResearchPlannerSummary({
  message,
  plan,
  unitResults = [],
  hasTrailingContent = false,
  onRetryFinalAnswer = null,
  retryingFinalAnswer = false,
}) {
  const plannerSummary = buildPlannerSummary(message, plan, unitResults)
  // 诊断默认折叠，避免校验码/状态压过答案；重试按钮始终在摘要行可见
  const [open, setOpen] = useState(false)
  if (!plannerSummary) return null
  const showRetry = canRetryPlannerFinalAnswer(message) && typeof onRetryFinalAnswer === 'function'

  return (
    <div className="sa-stack" style={{ marginBottom: hasTrailingContent ? 12 : 0 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>研究摘要</p>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          取证 {formatPlannerStatus(plannerSummary.plannerStatus)}
        </span>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          最终回答 {plannerSummary.finalStatusLabel || formatPlannerStatus(plannerSummary.finalStatus)}
        </span>
        <span className="sa-chip" style={{ fontSize: 12 }}>
          {plannerSummary.success}/{plannerSummary.total} 单元 · 证据 {plannerSummary.evidenceCount}
        </span>
        {showRetry ? (
          <button
            type="button"
            className="sa-button sa-button--ghost"
            title="重试最终回答"
            onClick={() => onRetryFinalAnswer(message)}
            disabled={retryingFinalAnswer}
          >
            <Icon icon="solar:refresh-linear" width={16} aria-hidden="true" />
            {retryingFinalAnswer ? '重试中' : '重试最终回答'}
          </button>
        ) : null}
        <button
          type="button"
          className="sa-button sa-button--ghost"
          style={{ fontSize: 12, minHeight: 28 }}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? '收起诊断' : '查看诊断'}
        </button>
      </div>
      {open ? (
        <>
          <p style={{ margin: 0, color: 'var(--sa-text-soft)' }}>
            已完成 {plannerSummary.success}/{plannerSummary.total} 个单元；
            证据 {plannerSummary.evidenceCount} 条；claims {plannerSummary.claimCount} 条。
          </p>
          {plannerSummary.blocked || plannerSummary.insufficient ? (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
              受阻 {plannerSummary.blocked} 个；证据不足 {plannerSummary.insufficient} 个。
            </p>
          ) : null}
          {plannerSummary.acceptedBlocks || plannerSummary.rejectedBlocks ? (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
              最终答案块已接受 {plannerSummary.acceptedBlocks} 个；校验错误 {plannerSummary.rejectedBlocks} 个。
            </p>
          ) : null}
          {plannerSummary.finalReason ? (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
              最终回答状态：{plannerSummary.finalReason}
            </p>
          ) : null}
          {plannerSummary.errorCodes.length ? (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>
              校验代码：{plannerSummary.errorCodes.join('、')}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
