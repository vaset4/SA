'use client'

import { useCallback, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { LibraryPageShell } from './LibraryPageShell'
import { useLibraryTools } from './use-library-tools'
import { useConversationStore } from '../../stores/conversation-store'
import { useLibraryStore } from '../../stores/library-store'
import { useProjectStore } from '../../stores/project-store'

export function PapersPageClient() {
  const conversations = useConversationStore()
  const library = useLibraryStore()
  const projectStore = useProjectStore()
  const router = useRouter()
  const searchParams = useSearchParams()
  const pushFeed = useCallback(() => {}, [])
  const tools = useLibraryTools(pushFeed)
  const projectIdParam = searchParams.get('projectId') || ''
  const queryParam = searchParams.get('q') || ''

  useEffect(() => {
    if (!projectIdParam || projectIdParam === projectStore.activeProjectId) return
    projectStore.setActiveProjectId(projectIdParam)
  }, [projectIdParam, projectStore])

  const handleAddToConversation = async (paperId) => {
    if (!paperId) return
    if (!conversations.activeId) {
      const conversationId = await conversations.createNew(`论文研读 · ${paperId}`)
      conversations.addPaper(conversationId, paperId)
      router.push(`/chat/${conversationId}`)
      return
    }
    conversations.addPaper(conversations.activeId, paperId)
  }

  const handleAddToProject = async (paperId) => {
    if (!projectStore.activeProjectId || !paperId) return
    await projectStore.addPaper(paperId)
  }

  return (
    <LibraryPageShell
      library={library}
      tools={tools}
      projectId={projectStore.activeProjectId}
      initialQuery={queryParam}
      onAddToConversation={handleAddToConversation}
      onAddToProject={handleAddToProject}
    />
  )
}
