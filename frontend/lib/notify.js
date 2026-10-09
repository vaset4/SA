'use client'

import { addToast } from '@heroui/react'
import {
  NOTIFY_DURATION_MS,
  emitNotify,
  setNotifySink as setCoreNotifySink,
} from './notify-core.mjs'

setCoreNotifySink(addToast)

export function setNotifySink(nextSink) {
  setCoreNotifySink(nextSink || addToast)
}

function notify({ title, description = '', color = 'default' }) {
  emitNotify({ title, description, color })
}

export function notifySuccess(title, description = '') {
  notify({ title, description, color: 'success' })
}

export function notifyError(title, description = '') {
  notify({ title, description, color: 'danger' })
}

export function notifyInfo(title, description = '') {
  notify({ title, description, color: 'primary' })
}

export { NOTIFY_DURATION_MS }
