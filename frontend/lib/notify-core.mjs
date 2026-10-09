export const NOTIFY_DURATION_MS = 4200

let notifySink = null

export function setNotifySink(nextSink) {
  notifySink = typeof nextSink === 'function' ? nextSink : null
}

export function emitNotify({ title, description = '', color = 'default' }) {
  if (!notifySink) {
    throw new Error('notify sink is not configured')
  }
  notifySink({
    title,
    description,
    color,
    timeout: NOTIFY_DURATION_MS,
  })
}
