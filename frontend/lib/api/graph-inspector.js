import { requestJson } from './core'

// 把对象拼成查询串，跳过空值；统一走 lib/api/core，不直接 fetch
function buildQuery(params) {
  const sp = new URLSearchParams()
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    sp.set(key, String(value))
  })
  const qs = sp.toString()
  return qs ? `?${qs}` : ''
}

// LightRAG 图概览：总数 / 类型分布 / 可疑·孤儿·重名计数
export async function getLightragOverview(libraryId = '') {
  return requestJson({
    path: `/api/v1/graph-inspector/lightrag/overview${buildQuery({ library_id: libraryId })}`,
  })
}

// LightRAG 实体明细：支持仅可疑 / 类型 / 关键词过滤 + 分页
export async function listLightragEntities({ libraryId = '', suspect, entityType, q, limit = 500, offset = 0 } = {}) {
  const path = `/api/v1/graph-inspector/lightrag/entities${buildQuery({
    library_id: libraryId,
    suspect,
    entity_type: entityType,
    q,
    limit,
    offset,
  })}`
  return requestJson({ path })
}

// LightRAG 关系明细：源 / 目标 / 描述 / 关键词 / 权重
export async function listLightragRelations({ libraryId = '', limit = 500, offset = 0 } = {}) {
  return requestJson({
    path: `/api/v1/graph-inspector/lightrag/relations${buildQuery({ library_id: libraryId, limit, offset })}`,
  })
}

// LightRAG 文档摄入状态：processed / failed 计数 + 每篇标题/状态/错误（诊断"图里几篇论文"）
export async function listLightragDocuments(libraryId = '') {
  return requestJson({
    path: `/api/v1/graph-inspector/lightrag/documents${buildQuery({ library_id: libraryId })}`,
  })
}

// LightRAG 文本 chunk：原文内容 + 归属论文 + 顺序（诊断"实际摄入了什么内容"）
export async function listLightragChunks({ libraryId = '', fullDocId = '', limit = 50, offset = 0 } = {}) {
  return requestJson({
    path: `/api/v1/graph-inspector/lightrag/chunks${buildQuery({ library_id: libraryId, full_doc_id: fullDocId, limit, offset })}`,
  })
}

// Neo4j 研究对象图概览：节点 / 边 / 论文计数（按 paper_ids 圈范围）
export async function getNeo4jOverview({ paperIds = '' } = {}) {
  return requestJson({
    path: `/api/v1/graph-inspector/neo4j/overview${buildQuery({ paper_ids: paperIds })}`,
  })
}
