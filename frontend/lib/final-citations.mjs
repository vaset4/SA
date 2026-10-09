/** 同次最终 Markdown 的引用位置投影；作者：xxx */
import { decodeString } from 'micromark-util-decode-string'

export const FINAL_DELIVERY_VERSION = 'research-planner-final-v3'
const SAFE_ID = /^[A-Za-z0-9_.:-]+$/
const LITERAL_NODES = new Set(['code', 'inlineCode', 'html', 'link', 'linkReference', 'definition'])

/** @param {object} message 消息 @returns {boolean} 是否声明新版交付 */
export function hasFinalDelivery(message = {}) {
  return message.final_delivery_report != null || message.final_delivery_applicability === 'research'
    || message.final_generation_attempt?.contract_version === FINAL_DELIVERY_VERSION
}

/** @param {string} messageId 消息身份 @param {string} evidenceId 真实来源身份 @returns {string} 来源锚点 */
export function finalEvidenceAnchor(messageId, evidenceId) {
  const message = String(messageId || 'message').replace(/[^A-Za-z0-9_.:-]/g, '_')
  return `#evidence-${message}-${evidenceId}`
}

/** @param {object} message 同次消息 @param {string} content 原始 Markdown @returns {object} UTF-16 位置与位置错误 */
export function prepareFinalCitations(message, content = '') {
  const report = message?.final_delivery_report
  if (!hasFinalDelivery(message)) return { bindings: [], errors: [] }
  if (report?.contract_version !== FINAL_DELIVERY_VERSION) {
    return { bindings: [], errors: ['当前交付缺少可识别的引用合同'] }
  }
  // 1. 【研究交付】【引用坐标】服务端码点下标显式映射到 JavaScript UTF-16，保留组合字符
  const offsets = [0]
  let units = 0
  for (const point of content) {
    units += point.length
    offsets.push(units)
  }
  const bindings = []
  const errors = []
  const raw = message.final_citation_bindings
  const attempt = message.final_generation_attempt?.attempt_id
  if (!Array.isArray(raw) || !Array.isArray(report.citation_bindings) || !attempt || attempt !== report.attempt_id) {
    return { bindings, errors: ['引用与正文的尝试身份不一致'] }
  }
  for (const item of raw) {
    const start = offsets[item?.start]
    const end = offsets[item?.end]
    const declared = report.citation_bindings?.some((entry) => (
      ['attempt_id', 'block_id', 'alias', 'evidence_id', 'start', 'end'].every((key) => entry[key] === item[key])
    ))
    const valid = item?.attempt_id === attempt && declared && typeof item.evidence_id === 'string'
      && SAFE_ID.test(item.evidence_id) && typeof item.block_id === 'string' && item.block_id.length > 0
      && /^E\d+$/.test(item.alias) && Number.isInteger(item.start) && Number.isInteger(item.end)
      && start !== undefined && end !== undefined && end > start
      && content.slice(start, end) === `[${item.alias}]`
    if (!valid || bindings.some((entry) => start < entry.end && end > entry.start)) {
      errors.push('部分引用的位置或来源关系无法核对')
      continue
    }
    bindings.push({ ...item, start, end, href: finalEvidenceAnchor(message.id, item.evidence_id) })
  }
  return { bindings: bindings.sort((left, right) => left.start - right.start), errors: [...new Set(errors)] }
}

/** @param {object} node 原始文本节点 @param {object} options 正文与绑定 @returns {Array} 仅替换指定位置的文本节点 */
function splitCitationText(node, options) {
  const start = node.position?.start?.offset
  const end = node.position?.end?.offset
  if (!Number.isInteger(start) || !Number.isInteger(end)) return [node]
  const source = options.content.slice(start, end)
  // 1. 【研究交付】【原文对应】用渲染器相同的 Markdown 解码保留转义和实体；不用同名标记搜索
  if (decodeString(source) !== node.value) return [node]
  const selected = options.bindings.filter((entry) => entry.start >= start && entry.end <= end)
  if (!selected.length) return [node]
  const children = []
  let cursor = start
  for (const entry of selected) {
    if (entry.start > cursor) children.push({ type: 'text', value: decodeString(options.content.slice(cursor, entry.start)) })
    children.push({ type: 'link', url: entry.href,
      data: { hProperties: { 'data-evidence-binding': entry.href } },
      children: [{ type: 'text', value: `[${entry.alias}]` }] })
    cursor = entry.end
  }
  if (cursor < end) children.push({ type: 'text', value: decodeString(options.content.slice(cursor, end)) })
  return children
}

/** @param {object} node Markdown 节点 @param {object} options 同次正文与绑定 @returns {Array} 保留代码和原生链接的节点 */
function projectCitationNode(node, options) {
  if (LITERAL_NODES.has(node.type)) return [node]
  if (node.type === 'text') return splitCitationText(node, options)
  if (!Array.isArray(node.children)) return [node]
  return [{ ...node, children: node.children.flatMap((child) => projectCitationNode(child, options)) }]
}

/** @param {object} options 同次正文与 UTF-16 绑定 @returns {Function} Markdown AST 插件 */
export function remarkFinalCitations(options) {
  return function transform(tree) {
    return projectCitationNode(tree, options)[0]
  }
}

/** @param {object} message 完整消息 @returns {Array} 同次合法引用对应的完整来源卡片 */
export function finalCitationRecords(message) {
  const { bindings } = prepareFinalCitations(message, message.content || '')
  const sourceRecords = Array.isArray(message.evidence_records) ? message.evidence_records : []
  const records = new Map(sourceRecords.map((record) => [record.evidence_id, record]))
  const selected = new Map()
  for (const binding of bindings) {
    const record = records.get(binding.evidence_id)
    if (record && !selected.has(binding.evidence_id)) {
      selected.set(binding.evidence_id, { ...record, citation_alias: binding.alias })
    }
  }
  return [...selected.values()]
}
