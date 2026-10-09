const READY_STATUSES = new Set(['analysis_ready', 'succeeded', 'ready', 'completed'])
const RUNNING_STATUSES = new Set(['queued', 'running'])

/** @param {object} paper 论文记录 @returns {string} 可选 AI 分析状态 */
function getAnalysisStatus(paper) {
  // 1. 【上传】【就绪显示】分析状态不代替正文和索引核验
  const status = String(paper?.analysis_status || '').trim()
  if (status) return status
  return paper?.analysis?.problem || paper?.analysis?.method || paper?.analysis?.tldr ? 'analysis_ready' : 'not_started'
}

function hasMaterial(paper) {
  return Boolean(paper?.pdf_path || paper?.pdf_url || paper?.abstract)
}

function analysisStep(paper) {
  const status = getAnalysisStatus(paper)
  if (READY_STATUSES.has(status)) return { key: 'analysis', label: '分析', status: 'ready' }
  if (RUNNING_STATUSES.has(status)) return { key: 'analysis', label: '分析', status: 'running' }
  if (status === 'failed') return { key: 'analysis', label: '分析', status: 'failed', hint: '可重试分析' }
  return { key: 'analysis', label: '分析', status: 'pending' }
}

/** @param {object} paper 论文发布事实 @returns {Array<object>} 原文、解析、分析和检索的真实状态 */
export function buildPaperReadinessSteps(paper = {}) {
  const analysis = analysisStep(paper)
  const published = Boolean(paper.index_batch_id) && paper.is_vectorized === true
    && (!paper.deletion_status || paper.deletion_status === 'active')
  return [
    {
      key: 'material',
      label: '材料',
      status: hasMaterial(paper) ? 'ready' : 'pending',
      hint: hasMaterial(paper) ? '' : '补充 PDF 或摘要',
    },
    { key: 'parse', label: '解析', status: published ? 'ready' : 'unknown' },
    analysis,
    {
      key: 'retrieval',
      label: '检索',
      status: published ? 'ready' : paper.is_vectorized === undefined ? 'unknown' : 'pending',
    },
  ]
}
