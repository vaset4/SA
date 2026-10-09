import { requestJson } from './core'

export const listNotifications = () =>
  requestJson({ path: '/api/v1/notifications' })

export const markNotificationRead = ({ logId, read = true }) =>
  requestJson({
    path: `/api/v1/notifications/${logId}`,
    init: {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ read }),
    },
  })
