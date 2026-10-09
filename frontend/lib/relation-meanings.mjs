export const RELATION_MEANINGS = {
  USES_METHOD: '使用相近方法',
  SHARES_METHOD: '方法路径相近',
  USES_DATASET: '使用相同数据集',
  SHARES_DATASET: '数据基础相近',
  ADDRESSES_TASK: '研究任务相关',
  SHARES_TASK: '研究目标相近',
  SAME_VENUE: '发表场景相近',
  CITES: '引用关系',
  EXTENDS: '延展关系',
}

export function formatRelationLabel(value) {
  const text = String(value || '').trim()
  if (!text) return '未知关系'
  return RELATION_MEANINGS[text] || text.toLowerCase().split('_').filter(Boolean).join(' ')
}

export function formatRelationMeaning(value) {
  const text = String(value || '').trim()
  return RELATION_MEANINGS[text] || '尚未配置业务解释的关系类型'
}
