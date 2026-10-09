import { MESSAGE_EXTRA_KEYS } from '../lib/study-assistant-workspace.mjs'

function sameExtra(left, right) {
  if (left === right) return true
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null)
}

export function sameMessages(left = [], right = []) {
  if (left === right) return true
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
  return left.every((item, index) => {
    const next = right[index]
    return item?.id === next?.id
      && item?.role === next?.role
      && item?.title === next?.title
      && item?.content === next?.content
      && item?.tone === next?.tone
      && item?.renderMode === next?.renderMode
      && MESSAGE_EXTRA_KEYS.every((key) => sameExtra(item?.[key], next?.[key]))
  })
}
