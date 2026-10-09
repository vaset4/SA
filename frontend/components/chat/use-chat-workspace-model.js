'use client'

import { useCallback, useEffect, useMemo, useRef } from 'react'
import { buildActiveOutputItems } from '../../lib/chat-research-scope'
import { sameMessages } from '../../utils/message-utils'

const EMPTY_MESSAGES = []
const EMPTY_PAPER_IDS = []

function persistedConversationMessages(messages) {
  return messages.filter((item) => !['welcome', 'restored'].includes(String(item?.id || '')))
}

function papersForConversation({ conversation, libraryPapers }) {
  const ids = conversation?.associatedPapers || EMPTY_PAPER_IDS
  return libraryPapers.filter((paper) => ids.includes(paper.paper_id))
}

function scopedConversationPapers({ associatedPapers, libraryPapers, paperIdParam }) {
  const items = associatedPapers.map((paper) => ({ ...paper, removable: true }))
  if (!paperIdParam || items.some((paper) => paper?.paper_id === paperIdParam)) return items
  const matchedPaper = libraryPapers.find((paper) => paper.paper_id === paperIdParam)
  return [
    ...items,
    {
      ...(matchedPaper || { paper_id: paperIdParam, title: paperIdParam }),
      removable: false,
    },
  ]
}

function projectScopedPapers(papers) {
  const items = Array.isArray(papers) ? papers : []
  return items.map((paper) => ({ ...paper, removable: false }))
}

function useConversationSelection({ conversationId, conversations }) {
  useEffect(() => {
    if (!conversations.isLoading && conversationId && conversationId !== conversations.activeId) {
      conversations.setActiveId(conversationId)
    }
  }, [conversationId, conversations.activeId, conversations.isLoading])
}

function useWorkspaceBinding({ workspace, activeConversation, activeProjectId, effectiveResearchPapers }) {
  const bindConversationRef = useRef(workspace.bindConversation)

  useEffect(() => {
    bindConversationRef.current = workspace.bindConversation
  }, [workspace.bindConversation])

  useEffect(() => {
    bindConversationRef.current({
      conversationId: activeConversation?.id || '',
      sessionId: activeConversation?.sessionId || '',
      projectId: activeProjectId,
      papers: effectiveResearchPapers,
      messages: activeConversation?.id ? activeConversation.messages || EMPTY_MESSAGES : EMPTY_MESSAGES,
    })
  }, [activeConversation?.id, activeConversation?.messages, activeConversation?.sessionId, activeProjectId, effectiveResearchPapers])
}

function useChatMode(workspace) {
  useEffect(() => {
    if (workspace.mode !== 'chat') workspace.setMode('chat')
  }, [workspace, workspace.mode])
}

function useSelectedPaperSync({ scopedPapers, library }) {
  useEffect(() => {
    const paperId = scopedPapers[0]?.paper_id
    if (paperId && library.selectedPaperId !== paperId) library.setSelectedPaperId(paperId)
  }, [library, scopedPapers])
}

function useConversationMessageSync({ activeConversation, conversations, persistedMessages, workspace }) {
  useEffect(() => {
    if (!activeConversation?.id) return
    if (workspace.conversationId !== activeConversation.id) return
    if (workspace.restoring) return
    if (sameMessages(persistedMessages, activeConversation.messages || EMPTY_MESSAGES)) return
    conversations.setMessages(activeConversation.id, persistedMessages, { bumpUpdatedAt: false })
  }, [activeConversation?.id, activeConversation?.messages, conversations, persistedMessages, workspace.conversationId, workspace.restoring])
}

function useSearchParamSync({ activeConversationId, library, paperIdParam, promptParam, workspace }) {
  const handledRef = useRef({ paperKey: '', promptKey: '' })

  useEffect(() => {
    if (paperIdParam) {
      const nextPaperKey = `${activeConversationId || 'no-conversation'}:${paperIdParam}`
      if (handledRef.current.paperKey !== nextPaperKey) {
        handledRef.current.paperKey = nextPaperKey
        library.setSelectedPaperId(paperIdParam)
      }
    }
    if (promptParam) {
      const nextPromptKey = `${activeConversationId || 'no-conversation'}:${promptParam}`
      if (handledRef.current.promptKey !== nextPromptKey) {
        handledRef.current.promptKey = nextPromptKey
        workspace.seedDraft(promptParam)
      }
    }
  }, [activeConversationId, library, paperIdParam, promptParam, workspace])
}

function useFeedAutoScroll({ feedRef, activeConversationId, messageCount }) {
  const scrollStateRef = useRef({ conversationId: '', messageCount: 0 })

  useEffect(() => {
    const node = feedRef.current
    if (!node) return
    const previous = scrollStateRef.current
    const switchedConversation = previous.conversationId !== activeConversationId
    const appendedMessages = messageCount > previous.messageCount
    if (switchedConversation || appendedMessages) {
      node.scrollTo({ top: node.scrollHeight, behavior: switchedConversation ? 'auto' : 'smooth' })
    }
    scrollStateRef.current = { conversationId: activeConversationId, messageCount }
  }, [activeConversationId, feedRef, messageCount])
}

function usePaperActions({ activeConversation, conversations }) {
  const addPaper = useCallback(async (paperId) => {
    if (!paperId) return
    if (activeConversation?.id) {
      await conversations.addPaper(activeConversation.id, paperId)
      return
    }
    const conversationId = await conversations.createNew(`论文研读 · ${paperId}`)
    await conversations.addPaper(conversationId, paperId)
  }, [activeConversation?.id, conversations])

  const removePaper = useCallback(async (paperId) => {
    if (!activeConversation?.id || !paperId) return
    await conversations.removePaper(activeConversation.id, paperId)
  }, [activeConversation?.id, conversations])

  return { addPaper, removePaper }
}

export function useChatWorkspaceModel({ conversationId, searchParams, conversations, library, projectStore, workspace, feedRef }) {
  const paperIdParam = searchParams.get('paperId') || ''
  const promptParam = searchParams.get('prompt') || ''
  const activeProjectId = projectStore.activeProjectId || ''
  const activeConversation = conversations.activeConversation
  const persistedMessages = useMemo(() => persistedConversationMessages(workspace.messages), [workspace.messages])
  const associatedPapers = useMemo(
    () => papersForConversation({ conversation: activeConversation, libraryPapers: library.papers }),
    [activeConversation?.associatedPapers, library.papers],
  )
  const scopedPapers = useMemo(() => {
    return scopedConversationPapers({ associatedPapers, libraryPapers: library.papers, paperIdParam })
  }, [associatedPapers, library.papers, paperIdParam])
  const projectPapers = useMemo(() => projectScopedPapers(projectStore.papers), [projectStore.papers])
  const effectiveResearchPapers = scopedPapers.length ? scopedPapers : projectPapers
  const outputItems = useMemo(
    () => buildActiveOutputItems({ conversation: activeConversation, workspace, projectStore }),
    [activeConversation, projectStore, workspace],
  )
  const paperActions = usePaperActions({ activeConversation, conversations })

  useConversationSelection({ conversationId, conversations })
  useWorkspaceBinding({ workspace, activeConversation, activeProjectId, effectiveResearchPapers })
  useChatMode(workspace)
  useSelectedPaperSync({ scopedPapers, library })
  useConversationMessageSync({ activeConversation, conversations, persistedMessages, workspace })
  useSearchParamSync({ activeConversationId: activeConversation?.id || '', library, paperIdParam, promptParam, workspace })
  useFeedAutoScroll({ feedRef, activeConversationId: activeConversation?.id || '', messageCount: persistedMessages.length })

  return { activeConversation, activeProjectId, effectiveResearchPapers, outputItems, paperActions, scopedPapers }
}
