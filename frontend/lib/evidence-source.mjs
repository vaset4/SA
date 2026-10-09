/** 证据来源标签；作者：xxx */

/**
 * 用已保存的全部来源显示页码，历史块锚点不代表新证据的全部页码范围
 * @param {object} record 证据或引用记录
 * @returns {string} 章节、实际页码与定位精度说明
 */
export function evidenceSourceLabel(record = {}) {
  const section = record.section_path || record.section || ''
  // 1. 【科研取证】【来源展示】历史记录沿用原有块定位，不补造新版范围
  if (record.read_contract_version == null) {
    return [section, record.page_num ? `p.${record.page_num}` : ''].filter(Boolean).join(' · ')
  }
  if (record.source_location_status === 'legacy_chunk') {
    return [section, '来源块定位，未保存完整页码范围'].filter(Boolean).join(' · ')
  }
  // 2. 【科研取证】【来源展示】页码只来自实际片段，单个兼容锚点不参与合并
  const spans = Array.isArray(record.source_spans) ? record.source_spans : []
  const fragments = spans.flatMap((span) => Array.isArray(span.source_fragments) ? span.source_fragments : [])
  const located = fragments.filter((fragment) => Number.isInteger(fragment.page_num) && fragment.page_num > 0)
  const pages = [...new Set(located.map((fragment) => fragment.page_num))].sort((a, b) => a - b)
  const pageLabel = pages.length ? `${pages.length > 1 ? 'pp.' : 'p.'}${pages.join('、')}` : ''
  const allLocated = fragments.length > 0 && located.length === fragments.length
    && spans.every((span) => Array.isArray(span.source_fragments) && span.source_fragments.length > 0)
  const limitation = allLocated ? '' : pages.length ? '部分来源无页码' : '未提供页码'
  return [section, pageLabel, limitation].filter(Boolean).join(' · ')
}
