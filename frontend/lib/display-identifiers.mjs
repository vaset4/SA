const SHORT_ID_LENGTH = 8

export function shortIdentifier(value) {
  const text = String(value || '').trim()
  if (!text) return ''
  return text.length > SHORT_ID_LENGTH ? `${text.slice(0, SHORT_ID_LENGTH)}...` : text
}

export function fallbackTitle(value, fallback = '未命名') {
  const text = String(value || '').trim()
  return text || fallback
}

export function formatNoteResourceLabels(note) {
  const labels = []
  if (note?.paper_id) labels.push(`关联论文 ${shortIdentifier(note.paper_id)}`)
  if (note?.session_id) labels.push(`关联研究 ${shortIdentifier(note.session_id)}`)
  if (note?.conversation_id) labels.push(`关联对话 ${shortIdentifier(note.conversation_id)}`)
  return labels
}
