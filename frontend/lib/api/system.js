import { requestJson } from './core'

export const getSystemHealth = () =>
  requestJson({ path: '/api/v1/health' })
