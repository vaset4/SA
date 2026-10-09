'use client'

import { createContext, useContext, useEffect, useMemo, useReducer } from 'react'
import { addNote, getPaperDetail, getCitations, listNotes, listPapers } from '../lib/api/papers'
import { LIMITS, getError, selectNextPaperId } from '../lib/study-assistant-workspace.mjs'

const initialState = {
  papers: [],
  selectedPaperId: '',
  paper: null,
  loading: true,
  detailLoading: false,
  error: '',
  detailError: '',
  notes: [],
  noteDraft: '',
  notesBusy: false,
  noteError: '',
  citations: null,
  citationsBusy: false,
  citationsError: '',
}

function reducer(state, action) {
  switch (action.type) {
    case 'setState':
      return { ...state, ...action.patch }
    default:
      return state
  }
}

const LibraryStoreContext = createContext(null)

export function LibraryStoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    let active = true

    listPapers({ limit: LIMITS.list })
      .then((data) => {
        if (!active) return
        const items = data?.papers ?? []
        const nextId = selectNextPaperId(items, '')
        dispatch({
          type: 'setState',
          patch: { papers: items, selectedPaperId: nextId, error: '', loading: false },
        })
      })
      .catch((reason) => {
        if (!active) return
        dispatch({
          type: 'setState',
          patch: { loading: false, error: getError(reason, '论文列表加载失败') },
        })
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!state.selectedPaperId) {
      dispatch({
        type: 'setState',
        patch: {
          paper: null,
          detailLoading: false,
          detailError: '',
          notes: [],
          citations: null,
          citationsError: '',
        },
      })
      return
    }

    let active = true
    dispatch({ type: 'setState', patch: { detailLoading: true, detailError: '' } })

    Promise.allSettled([
      getPaperDetail({ paperId: state.selectedPaperId }),
      listNotes({ paperId: state.selectedPaperId }),
    ]).then(([paperResult, notesResult]) => {
      if (!active) return
      dispatch({
        type: 'setState',
        patch: {
          detailLoading: false,
          paper: paperResult.status === 'fulfilled' ? paperResult.value : null,
          detailError: paperResult.status === 'rejected'
            ? getError(paperResult.reason, '论文详情加载失败')
            : '',
          notes: notesResult.status === 'fulfilled' ? notesResult.value?.notes ?? [] : [],
          noteError: notesResult.status === 'rejected'
            ? getError(notesResult.reason, '笔记加载失败')
            : '',
        },
      })
    })

    return () => {
      active = false
    }
  }, [state.selectedPaperId])

  const actions = useMemo(() => ({
    async refresh(preferred = state.selectedPaperId) {
      const data = await listPapers({ limit: LIMITS.list })
      const items = data?.papers ?? []
      const nextId = selectNextPaperId(items, preferred)
      dispatch({
        type: 'setState',
        patch: { papers: items, selectedPaperId: nextId, error: '' },
      })
      return { items, nextId }
    },
    setSelectedPaperId(selectedPaperId) {
      dispatch({ type: 'setState', patch: { selectedPaperId } })
    },
    upsertPaper(nextPaper) {
      if (!nextPaper?.paper_id) return
      const papers = state.papers.some((paper) => paper.paper_id === nextPaper.paper_id)
        ? state.papers.map((paper) => (paper.paper_id === nextPaper.paper_id ? nextPaper : paper))
        : [nextPaper, ...state.papers]
      dispatch({
        type: 'setState',
        patch: {
          papers,
          paper: state.selectedPaperId === nextPaper.paper_id ? nextPaper : state.paper,
        },
      })
    },
    setNoteDraft(noteDraft) {
      dispatch({ type: 'setState', patch: { noteDraft } })
    },
    async refreshNotes(paperId = state.selectedPaperId) {
      if (!paperId) {
        dispatch({ type: 'setState', patch: { notes: [] } })
        return
      }
      const data = await listNotes({ paperId })
      dispatch({ type: 'setState', patch: { notes: data?.notes ?? [], noteError: '' } })
    },
    async saveNote() {
      if (!state.selectedPaperId || !state.noteDraft.trim() || state.notesBusy) return
      dispatch({ type: 'setState', patch: { notesBusy: true, noteError: '' } })
      try {
        await addNote({ paperId: state.selectedPaperId, content: state.noteDraft.trim() })
        const data = await listNotes({ paperId: state.selectedPaperId })
        dispatch({
          type: 'setState',
          patch: {
            notesBusy: false,
            noteDraft: '',
            notes: data?.notes ?? [],
          },
        })
      } catch (reason) {
        dispatch({
          type: 'setState',
          patch: {
            notesBusy: false,
            noteError: getError(reason, '笔记保存失败'),
          },
        })
      }
    },
    async loadCitations(depth = 1) {
      if (!state.selectedPaperId || state.citationsBusy) return
      dispatch({
        type: 'setState',
        patch: {
          citationsBusy: true,
          citationsError: '',
        },
      })
      try {
        const citations = await getCitations({ paperId: state.selectedPaperId, depth })
        dispatch({
          type: 'setState',
          patch: {
            citationsBusy: false,
            citations,
          },
        })
      } catch (reason) {
        dispatch({
          type: 'setState',
          patch: {
            citationsBusy: false,
            citationsError: getError(reason, '引用链加载失败'),
          },
        })
      }
    },
  }), [state.citationsBusy, state.noteDraft, state.notesBusy, state.paper, state.papers, state.selectedPaperId])

  const value = useMemo(() => ({
    ...state,
    ...actions,
  }), [actions, state])

  return (
    <LibraryStoreContext.Provider value={value}>
      {children}
    </LibraryStoreContext.Provider>
  )
}

export function useLibraryStore() {
  const context = useContext(LibraryStoreContext)
  if (!context) {
    throw new Error('useLibraryStore must be used within LibraryStoreProvider')
  }
  return context
}
