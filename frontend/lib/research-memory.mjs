// 研究记忆层前端展示常量：maturity/类型/状态/关系的中文标签与选项
// 与后端受控枚举 research_memory_contracts.py 对齐，改枚举时同步这一处

// 认知成熟度阶梯：直觉 → 倾向 → 已确立（台账主轴）
export const MATURITY_ORDER = ['intuition', 'leaning', 'established']

export const MATURITY_META = {
  intuition: { label: '直觉', level: 1, hint: '凭直觉或来自对话，尚无文献背书' },
  leaning: { label: '倾向', level: 2, hint: '已有证据支撑，但仍存冲突或待验证' },
  established: { label: '已确立', level: 3, hint: '证据充分、无未决冲突，可作为论证基石' },
}

export function formatMaturity(value) {
  return MATURITY_META[value]?.label || value || '直觉'
}

export function maturityLevel(value) {
  return MATURITY_META[value]?.level || 1
}

// 记忆类型标签
export const MEMORY_TYPE_META = {
  research_purpose: '研究目的',
  paper_note: '论文笔记',
  method_note: '方法笔记',
  dataset_note: '数据集笔记',
  metric_note: '指标笔记',
  comparison_note: '对比笔记',
  evidence_clip: '证据摘录',
  question_note: '问题笔记',
  gap_note: '空白笔记',
  decision_note: '决策笔记',
  synthesis_note: '推断综合',
}

export const MEMORY_TYPE_OPTIONS = Object.entries(MEMORY_TYPE_META).map(([value, label]) => ({ value, label }))

export function formatMemoryType(value) {
  return MEMORY_TYPE_META[value] || value || ''
}

// 生命周期状态
export const STATUS_META = {
  draft: '草稿',
  confirmed: '已确认',
  archived: '已归档',
}

export function formatStatus(value) {
  return STATUS_META[value] || value || ''
}

// 新鲜度状态
export const STALE_META = {
  fresh: '证据有效',
  stale: '证据存疑',
  needs_rebind: '待重绑',
}

export function formatStale(value) {
  return STALE_META[value] || value || ''
}

// 证据模式
export const EVIDENCE_MODE_META = {
  evidence_backed: '文献背书',
  user_note_only: '纯笔记',
  mixed: '混合',
}

export const EVIDENCE_MODE_OPTIONS = Object.entries(EVIDENCE_MODE_META).map(([value, label]) => ({ value, label }))

export function formatEvidenceMode(value) {
  return EVIDENCE_MODE_META[value] || value || ''
}

// 记忆关系类型
export const LINK_TYPE_META = {
  based_on: '基于',
  conflicts_with: '冲突',
  supersedes: '修正',
}

export const LINK_TYPE_OPTIONS = Object.entries(LINK_TYPE_META).map(([value, label]) => ({ value, label }))

export function formatLinkType(value) {
  return LINK_TYPE_META[value] || value || ''
}

// ─── 记忆状态三分档 ───
// 记忆页要能分别查看已确认、待审核、待重新核对三档。这三档是审核与新鲜度的轴，
// 与台账的成熟度轴（直觉→倾向→已确立）互不替代：同一条记忆两边都会出现。

export const MEMORY_STATE_CONFIRMED = 'confirmed'
export const MEMORY_STATE_PENDING = 'pending'
export const MEMORY_STATE_REBIND = 'rebind'
export const MEMORY_STATE_INACTIVE = 'inactive'

export const MEMORY_STATE_META = {
  [MEMORY_STATE_CONFIRMED]: {
    label: '已确认',
    hint: 'Agent 每轮可读到的就是这一档：已接纳、已确认、证据仍有效',
  },
  [MEMORY_STATE_PENDING]: {
    label: '待审核',
    hint: '等你决定的一档：AI 草稿等接纳，手写的文献背书判断等挂证据',
  },
  [MEMORY_STATE_REBIND]: {
    label: '待重新核对',
    hint: '证据存疑或待重绑，Agent 读不到，需要你重新核对来源',
  },
  [MEMORY_STATE_INACTIVE]: {
    label: '未生效',
    hint: '已否决或已归档，不参与 Agent 上下文，也不在上面三档里',
  },
}

export function formatMemoryState(value) {
  return MEMORY_STATE_META[value]?.label || value || ''
}

// AI 产出的草稿来源，判断待审档里该给哪套操作时用
const AI_DRAFT_SOURCE_KINDS = new Set(['agent_draft', 'system_draft'])

export function isAiDraftSource(sourceKind) {
  return AI_DRAFT_SOURCE_KINDS.has(String(sourceKind || ''))
}

/**
 * 把一条记忆归入状态三分档（互斥且完备）。
 * 判定次序即优先级：失效最要紧，其次是等人动手，最后才是已生效。
 * @param {object} item 记忆条目（需含 review_state / status / stale_state）
 * @returns {string} MEMORY_STATE_* 之一
 */
export function classifyMemoryState(item = {}) {
  const reviewState = String(item.review_state || '')
  const status = String(item.status || '')
  const staleState = String(item.stale_state || '')

  // 1. 证据不再可靠的排最前：这一档 Agent 读不到，摆在后面会让用户以为记忆仍在生效
  if (staleState === 'stale' || staleState === 'needs_rebind') return MEMORY_STATE_REBIND
  // 2. 否决与归档不进前三档，也不许混进待审骗用户再审一次
  if (reviewState === 'dismissed' || status === 'archived') return MEMORY_STATE_INACTIVE
  // 3. AI 草稿等接纳；手写的文献背书判断落 draft 等挂证据，两者都在等用户动手
  if (reviewState === 'pending_review' || status === 'draft') return MEMORY_STATE_PENDING
  // 4. 剩下的才是四道资格门认的那一档：accepted + confirmed + fresh
  if (reviewState === 'accepted' && status === 'confirmed' && staleState === 'fresh') {
    return MEMORY_STATE_CONFIRMED
  }
  return MEMORY_STATE_INACTIVE
}

/**
 * 按状态三分档把记忆列表分组，供三状态视图一次渲染。
 * @param {Array<object>} items 记忆条目列表
 * @returns {{confirmed: Array, pending: Array, rebind: Array, inactive: Array}}
 */
export function groupMemoryByState(items = []) {
  const groups = {
    [MEMORY_STATE_CONFIRMED]: [],
    [MEMORY_STATE_PENDING]: [],
    [MEMORY_STATE_REBIND]: [],
    [MEMORY_STATE_INACTIVE]: [],
  }
  for (const item of items) {
    groups[classifyMemoryState(item)].push(item)
  }
  return groups
}

/**
 * 待审档里这条记忆该走哪套操作。
 * AI 草稿走接纳/忽略；手写的文献背书草稿必须先挂证据 —— 直接确认会绕过
 * 「事实型记忆确认前至少绑一条证据」这道既有检查。
 * @param {object} item 记忆条目
 * @returns {'accept_draft'|'bind_evidence'|'confirm'} 操作类别
 */
export function pendingActionKind(item = {}) {
  if (String(item.review_state || '') === 'pending_review') return 'accept_draft'
  if (String(item.evidence_mode || '') === 'evidence_backed' && !isAiDraftSource(item.source_kind)) {
    return 'bind_evidence'
  }
  return 'confirm'
}
