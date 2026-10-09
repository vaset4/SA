import { requestJson } from './core'

export async function listRagTraces({ limit = 50, requestId, sessionId, q } = {}) {
  const params = new URLSearchParams()
  if (limit) params.set('limit', String(limit))
  if (requestId) params.set('request_id', requestId)
  if (sessionId) params.set('session_id', sessionId)
  if (q) params.set('q', q)
  const qs = params.toString()
  return requestJson({ path: `/api/v1/observability/rag-traces${qs ? `?${qs}` : ''}` })
}

export async function getRagTrace(requestId) {
  return requestJson({ path: `/api/v1/observability/rag-traces/${encodeURIComponent(requestId)}` })
}
