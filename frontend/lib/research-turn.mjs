// 项目科研回合的前端展示语义：本轮深度、记忆读取终态、候选写回终态。
// 与后端受控枚举 research_contracts.py 的 TurnDepth / CapabilityStatus 对齐，改枚举时同步这一处。
//
// 一条硬规则贯穿本文件：只描述真实发生过的动作。
// 候选生成出来是「待你审核」，不是「已纳入记忆」；深度评估过不等于「计划完成」；
// 读到记忆不等于「已引用」—— 记忆是项目历史判断，本来就不得作为引用来源。

export const TURN_DEPTH_MINIMAL = 'minimal'
export const TURN_DEPTH_LOOKUP = 'lookup'
export const TURN_DEPTH_RESEARCH = 'research'

export const TURN_DEPTH_META = {
  [TURN_DEPTH_MINIMAL]: { label: '直接协助', hint: '本轮没有检索论文，只用对话上下文与项目记忆作答' },
  [TURN_DEPTH_LOOKUP]: { label: '轻量查证', hint: '本轮查了论文原文，可展开看依据' },
  [TURN_DEPTH_RESEARCH]: { label: '复杂研究', hint: '本轮拆了子任务，逐篇取证并核对冲突与缺口' },
}

export function formatTurnDepth(value) {
  return TURN_DEPTH_META[value]?.label || ''
}

/**
 * 这条消息是不是项目科研回合。
 * 判据是后端有没有下发 turn_depth：普通 RAG 问答不产这个字段，
 * 缺了就按老样子渲染，不给它编一个深度。
 * @param {object} message 助手消息
 * @returns {boolean}
 */
export function isProjectResearchTurn(message = {}) {
  return Boolean(TURN_DEPTH_META[message.turn_depth])
}

/**
 * 按本轮深度决定回答卡露出哪些入口。
 *
 * 直接协助档只露回答与项目记忆标识：这一档一次论文检索都没发，
 * 给个点开就是空的依据入口，等于摆出一个没发生过的动作；阶段进度条同理不摆。
 * 「过程」入口三档都留 —— 失败与终态要能被看见，这是排查的唯一入口，
 * 直接协助档的阶段信息仍可从这里点开，只是不默认占版面。
 *
 * @param {string} depth 本轮深度，缺失或未知按老样子全露出
 * @returns {{showEvidenceToggle: boolean, showProcessToggle: boolean, showProgressSteps: boolean}}
 */
export function turnDepthView(depth) {
  if (!TURN_DEPTH_META[depth]) {
    return { showEvidenceToggle: true, showProcessToggle: true, showProgressSteps: true }
  }
  const beyondMinimal = depth !== TURN_DEPTH_MINIMAL
  return {
    showEvidenceToggle: beyondMinimal,
    showProcessToggle: true,
    showProgressSteps: beyondMinimal,
  }
}

// 四态到展示色调的映射：empty 与 failed 不共用色调，用户要一眼分清
// 「项目里没有相关判断」和「记忆存储出问题了」
const TONE_OK = 'ok'
const TONE_NEUTRAL = 'neutral'
const TONE_DANGER = 'danger'

/**
 * 记忆读取终态的展示文案。
 * @param {object|null} result MemoryReadResult
 * @returns {{tone: string, label: string, detail: string, itemCount: number}|null}
 *   null 表示后端没下发这个字段，调用方不渲染这一项，而不是显示「未使用记忆」
 */
export function describeMemoryRead(result) {
  if (!result || typeof result !== 'object' || !result.status) return null
  const items = Array.isArray(result.items) ? result.items : []
  if (result.status === 'success') {
    return {
      tone: TONE_OK,
      label: `使用了 ${items.length} 条项目历史判断`,
      detail: '这些是项目内已确认的判断，不是论文原文，回答里不会把它们当引用来源',
      itemCount: items.length,
    }
  }
  if (result.status === 'empty') {
    return { tone: TONE_NEUTRAL, label: '未使用记忆', detail: '项目内没有符合条件的已确认判断', itemCount: 0 }
  }
  if (result.status === 'skipped') {
    return { tone: TONE_NEUTRAL, label: '未读取记忆', detail: '本轮判断不需要项目历史判断', itemCount: 0 }
  }
  return {
    tone: TONE_DANGER,
    label: '记忆读取失败',
    detail: result.error_kind ? `失败类型：${result.error_kind}` : '记忆存储暂时不可用，本轮回答没带上项目历史判断',
    itemCount: 0,
  }
}

/**
 * 候选处理进度与写回终态的展示文案。
 * skipped 与 empty 分开说：一个是本档不评估，一个是评估过但没有够格的判断。
 * @param {object|null} result MemoryWritebackResult
 * @param {string|undefined} persistenceStatus 同份答案的保存状态，历史缺失时不推断已保存
 * @returns {{tone: string, label: string, detail: string, candidateIds: string[], retryable: boolean}|null}
 */
export function describeMemoryWriteback(result, persistenceStatus) {
  if (!result || typeof result !== 'object' || !result.status) return null
  // 1. 【项目记忆】【提交后候选】保存失败保留候选准备事实，但不能显示处理已开始或已保存
  if (['pending', 'not_staged'].includes(persistenceStatus)) {
    return { tone: TONE_NEUTRAL, label: '记忆候选尚未处理', detail: '需先保存答案，再处理本轮候选',
      candidateIds: [], retryable: false }
  }
  const candidateIds = Array.isArray(result.candidate_ids) ? result.candidate_ids : []
  const reasons = Array.isArray(result.reason_codes) ? result.reason_codes.filter(Boolean) : []
  if (result.status === 'pending' || reasons.includes('awaiting_result_commit')) {
    return {
      tone: TONE_NEUTRAL,
      label: '记忆候选处理中',
      detail: persistenceStatus === 'saved' ? '答案已保存，正在查询候选处理状态' : '正在查询候选处理状态',
      candidateIds: [],
      retryable: false,
    }
  }
  if (result.status === 'success') {
    return {
      tone: TONE_OK,
      // 候选还在待审位，措辞不能说成「已纳入记忆」
      label: `生成 ${candidateIds.length} 条待审候选`,
      detail: '需要你确认后才会进入记忆库并被 Agent 读到',
      candidateIds,
      retryable: false,
    }
  }
  if (result.status === 'empty') {
    return {
      tone: TONE_NEUTRAL,
      label: '本轮没有值得留下的判断',
      detail: reasons.length ? `原因：${reasons.join('、')}` : '评估过了，本轮结论不适合形成长期判断',
      candidateIds: [],
      retryable: false,
    }
  }
  if (result.status === 'skipped') {
    return {
      tone: TONE_NEUTRAL,
      label: '本轮不评估候选',
      detail: reasons.length ? `原因：${reasons.join('、')}` : '只有复杂研究档且终态成功的回合才评估候选',
      candidateIds: [],
      retryable: false,
    }
  }
  const retryable = result.retryable === true
  return {
    tone: TONE_DANGER,
    label: retryable ? '候选写回失败（可重试）' : '候选写回失败',
    detail: [
      result.error_kind ? `失败类型：${result.error_kind}` : '',
      retryable ? '可重试这份已保存答案的候选处理' : '这次失败重试也不会变，请查看轨迹定位原因',
    ].filter(Boolean).join('；'),
    candidateIds,
    retryable,
  }
}

/**
 * 汇总本轮各项能力的真实结果，供回答卡一行行摆出来。
 * 只放后端真下发过的项：拿不到的能力不出现，也不补一个「正常」冒充。
 * @param {object} message 助手消息
 * @returns {Array<{key: string, tone: string, label: string, detail: string}>}
 */
export function summarizeTurnCapabilities(message = {}) {
  const rows = []
  const depth = message.turn_depth
  if (TURN_DEPTH_META[depth]) {
    rows.push({ key: 'depth', tone: TONE_NEUTRAL, label: TURN_DEPTH_META[depth].label, detail: TURN_DEPTH_META[depth].hint })
  }
  const read = describeMemoryRead(message.memory_read_result)
  if (read) rows.push({ key: 'memory_read', tone: read.tone, label: read.label, detail: read.detail })

  // 无检索结果要说出来：需要论文证据的深度拿不到引用时，用户得知道是没检索到而不是没检索
  if (depth === TURN_DEPTH_LOOKUP || depth === TURN_DEPTH_RESEARCH) {
    const citationCount = Array.isArray(message.citations) ? message.citations.length : 0
    rows.push(citationCount
      ? { key: 'retrieval', tone: TONE_OK, label: `命中 ${citationCount} 条论文依据`, detail: '可在「依据」里逐条核对原文' }
      : { key: 'retrieval', tone: TONE_DANGER, label: '无检索结果', detail: '本轮没有命中可引用的论文原文，可缩小范围或补材料' })
  }

  const gapCount = Array.isArray(message.missing_evidence) ? message.missing_evidence.length : 0
  if (gapCount) {
    rows.push({ key: 'evidence_gap', tone: TONE_DANGER, label: `${gapCount} 处证据缺口`, detail: '这些判断还缺原文支撑，展开「依据」可看缺口清单' })
  }

  const writeback = describeMemoryWriteback(message.memory_writeback, message.persistence_status)
  if (writeback) rows.push({ key: 'memory_writeback', tone: writeback.tone, label: writeback.label, detail: writeback.detail })
  return rows
}
