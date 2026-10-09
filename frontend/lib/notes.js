function normalizeKey(value = '') {
  return `${value || ''}`.trim().toLowerCase().replace(/[-\s]+/g, '_')
}

const NOTE_TYPE_LABELS = {
  reading_note: 'Reading Note',
  evidence_note: 'Evidence Note',
  comparison_note: 'Comparison Note',
  question_note: 'Question Note',
  idea_note: 'Idea Note',
  experiment_note: 'Experiment Note',
}

const SOURCE_TYPE_LABELS = {
  manual: 'Manual',
  local_rag: 'Local RAG',
  deep_research: 'Deep Research',
}

export const NOTE_TYPE_OPTIONS = Object.entries(NOTE_TYPE_LABELS).map(([value, label]) => ({ value, label }))
export const NOTE_SOURCE_OPTIONS = Object.entries(SOURCE_TYPE_LABELS).map(([value, label]) => ({ value, label }))

export function normalizeNoteType(value = '') {
  const key = normalizeKey(value)
  return NOTE_TYPE_LABELS[key] ? key : 'reading_note'
}

export function normalizeNoteSource(value = '') {
  const key = normalizeKey(value)
  return SOURCE_TYPE_LABELS[key] ? key : 'manual'
}

export function formatNoteType(value = '') {
  return NOTE_TYPE_LABELS[normalizeNoteType(value)]
}

export function formatNoteSource(value = '') {
  return SOURCE_TYPE_LABELS[normalizeNoteSource(value)]
}

export function normalizeNote(note) {
  if (!note || typeof note !== 'object') return null
  return {
    ...note,
    note_type: normalizeNoteType(note.note_type),
    source_type: normalizeNoteSource(note.source_type),
    evidence_reference: note.evidence_reference && typeof note.evidence_reference === 'object'
      ? note.evidence_reference
      : {},
  }
}

export function normalizeNoteCollection(notes) {
  return Array.isArray(notes) ? notes.map((item) => normalizeNote(item)).filter(Boolean) : []
}
