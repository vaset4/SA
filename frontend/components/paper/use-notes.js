'use client'

import { useEffect, useState } from 'react'
import { addNote, listNotes } from '../../lib/api/papers'
import { getError } from '../../lib/study-assistant-workspace.mjs'

export function useNotes(selectedPaperId) {
  const [notes, setNotes] = useState([])
  const [noteDraft, setNoteDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const refresh = async (paperId = selectedPaperId) => {
    if (!paperId) return setNotes([])
    const data = await listNotes({ paperId })
    setNotes(data?.notes ?? [])
  }

  useEffect(() => { refresh().catch((reason) => setError(getError(reason, '笔记加载失败'))) }, [selectedPaperId])

  const save = async () => {
    if (!selectedPaperId || !noteDraft.trim() || busy) return
    setBusy(true)
    setError('')
    try {
      await addNote({
        paperId: selectedPaperId,
        content: noteDraft.trim(),
        noteType: 'reading_note',
        sourceType: 'manual',
      })
      setNoteDraft('')
      await refresh(selectedPaperId)
    } catch (reason) {
      setError(getError(reason, '笔记保存失败'))
    } finally {
      setBusy(false)
    }
  }

  return { notes, noteDraft, setNoteDraft, busy, error, save }
}
