import { externalResearchModeLabel, normalizeExternalResearchMode } from './external-research-mode'

const TRACE_STAGE_LABELS = Object.freeze({
  queued: '请求已入队',
  provider_queued: '等待模型服务额度',
  provider_cooldown: '等待模型服务额度恢复',
  provider_admitted: '已获得模型服务额度',
  provider_cancelled: '已取消等待模型服务',
  provider_queue_failed: '模型服务额度等待失败',
  session: '研究会话已建立',
  route: '研究路由完成',
  run_header: '研究运行配置',
  paper_set_resolved: '论文池解析完成',
  comparison_matrix_ready: '比较矩阵已生成',
  evidence_accumulated: '证据卡片已装配',
  reasoning_summary: '推理摘要',
  source_call: '来源调用',
  tool: '调用工具',
  analyze: '分析阶段',
  done: '研究完成',
  clarify_with_user: '澄清判断',
  clarify_fallback: '澄清阶段降级',
  write_research_brief: '生成研究简报',
  write_research_brief_fallback: '研究简报降级',
  supervisor_iteration: '规划研究子任务',
  supervisor_fallback: '研究规划降级',
  search_topic_start: '开始检索主题',
  search_topic_done: '主题检索完成',
  researcher: '执行研究子任务',
  researcher_fallback: '研究子任务降级',
  researcher_tools: '执行外部搜索',
  compress_research: '压缩研究发现',
  compress_research_fallback: '压缩阶段降级',
  supervisor_tools: '汇总并发研究结果',
  supervisor_tools_fallback: '子任务自动降级',
  final_report_generation: '生成最终报告',
  final_report_prompt_chars: '准备最终报告提示词',
  final_report_fallback: '最终报告降级',
  llm_structured_start: '结构化推理开始',
  llm_structured_done: '结构化推理完成',
  llm_text_start: '文本生成开始',
  llm_text_done: '文本生成完成',
  run_deep_research_graph_start: '启动研究图',
  run_deep_research_graph_paused_for_clarification: '等待用户澄清',
  run_deep_research_graph_done: '研究图完成',
  error: '请求失败',
})

function normalizeStage(stage = '') {
  return String(stage || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
}

function fallbackTitle(stage = '', type = '') {
  const raw = String(stage || type || 'event').trim()
  if (!raw) return '事件'
  return raw.replace(/[_-]+/g, ' ')
}

export function formatResearchTraceTitle(event = {}) {
  const key = normalizeStage(event.stage || event.type)
  return TRACE_STAGE_LABELS[key] || fallbackTitle(event.stage, event.type)
}

/** @param {object[]} events 本次事件流 @param {string} runId 活动运行 @returns {object|null} 仍在等待的真实请求 */
export function currentProviderWait(events, runId) {
  if (!runId) return null
  const waiting = new Map()
  for (const event of Array.isArray(events) ? events : []) {
    if (event?.run_id !== runId || !event.request_id) continue
    // 1. 【模型服务】【等待展示】每次HTTP尝试独立退出，其他并发请求仍保留等待
    if (event.stage === 'provider_queued') waiting.set(event.request_id, event)
    if (['provider_admitted', 'provider_cancelled', 'provider_queue_failed'].includes(event.stage)) {
      waiting.delete(event.request_id)
    }
  }
  return waiting.values().next().value || null
}

export function formatResearchTraceDetail(event = {}) {
  if (event.stage === 'provider_queued' || event.stage === 'provider_cooldown') {
    const seconds = event.estimated_wait_seconds
    return Number.isFinite(seconds)
      ? `预计等待约 ${Math.ceil(seconds)} 秒，可能随服务额度变化；可取消本次研究`
      : '正在等待模型服务额度，可取消本次研究'
  }
  if (event.detail) return String(event.detail)
  if (event.stage === 'run_header') {
    const mode = externalResearchModeLabel(event.research_mode)
    const paperCount = Number(event.paper_count || 0)
    return `${mode} · 背景论文 ${paperCount} 篇`
  }
  if (event.stage === 'paper_set_resolved') {
    return `已解析 ${event.result_count || 0} 篇背景论文`
  }
  if (event.stage === 'comparison_matrix_ready') {
    return `比较矩阵 ${event.row_count || 0} 行`
  }
  if (event.stage === 'evidence_accumulated') {
    return `已装配 ${event.evidence_count || 0} 张证据卡`
  }
  if (event.stage === 'reasoning_summary') {
    return event.next_action ? `${event.summary || event.detail || '已生成摘要'}；下一步：${event.next_action}` : (event.summary || '已生成推理摘要')
  }
  if (event.stage === 'source_call') {
    const sourceType = event.source_type || 'source'
    const resultCount = Number(event.result_count || 0)
    return `${sourceType} · ${resultCount} 条结果`
  }
  if (event.intent) return `已选择执行路径：${event.intent}`
  if (event.tool_name) return `工具调用：${event.tool_name}`
  if (event.prompt) return `课题：${event.prompt}`
  if (event.research_mode) return `模式：${externalResearchModeLabel(normalizeExternalResearchMode(event.research_mode))}`
  if (event.session_id) return `session_id：${event.session_id}`
  if (event.node) return `节点：${event.node}`
  return '状态已更新'
}
