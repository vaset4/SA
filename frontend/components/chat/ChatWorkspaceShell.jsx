'use client'

import { useRef } from 'react'
import { useSearchParams } from 'next/navigation'
import { WorkspaceLayout } from '../layout/WorkspaceLayout'
import { useConversationStore } from '../../stores/conversation-store'
import { useLibraryStore } from '../../stores/library-store'
import { useProjectStore } from '../../stores/project-store'
import { useWorkspaceStore } from '../../stores/workspace-store'
import { ChatHeader } from './ChatHeader'
import { ChatResearchRail } from './ChatResearchRail'
import { ComposerBar } from './ComposerBar'
import { ConversationSidebar } from './ConversationSidebar'
import { MessageViewport } from './MessageViewport'
import { useChatWorkspaceModel } from './use-chat-workspace-model'

function ChatWorkspaceRail({ library, model, onAddPaper, onRemovePaper }) {
  return (
    <ChatResearchRail
      library={library}
      activePaperIds={model.activeConversation?.associatedPapers || []}
      onAddPaper={onAddPaper}
      onRemovePaper={onRemovePaper}
    />
  )
}

function ChatWorkspaceHeader({ projectStore, workspace, model }) {
  return (
    <ChatHeader
      conversation={model.activeConversation}
      currentSessionId={workspace.sessionId}
      papers={model.scopedPapers}
      projectName={projectStore.activeProject?.name || ''}
      projectId={model.activeProjectId}
      busy={workspace.busy}
    />
  )
}

function ChatWorkspaceMain({ feedRef, workspace, projectStore, paperCount }) {
  return (
    <div className="sa-chat-shell sa-research-room-reading">
      <div ref={feedRef} className="sa-chat-shell__feed">
        <MessageViewport messages={workspace.messages} />
      </div>
      <ComposerBar
        busy={workspace.busy}
        restoring={workspace.restoring}
        draft={workspace.draft}
        onDraftChange={workspace.setDraft}
        onSubmit={workspace.submit}
        onStop={workspace.stop}
        projectName={projectStore.activeProject?.name || ''}
        paperCount={paperCount}
      />
    </div>
  )
}

export function ChatWorkspaceShell({ conversationId = null }) {
  const searchParams = useSearchParams()
  const conversations = useConversationStore()
  const library = useLibraryStore()
  const projectStore = useProjectStore()
  const workspace = useWorkspaceStore()
  const feedRef = useRef(null)
  const model = useChatWorkspaceModel({
    conversationId,
    searchParams,
    conversations,
    library,
    projectStore,
    workspace,
    feedRef,
  })

  return (
    <WorkspaceLayout
      shellClassName="sa-research-room-shell sa-research-chat-workspace"
      contentClassName="sa-research-room-main-scroll"
      sidebar={<ConversationSidebar />}
      rail={(
        <ChatWorkspaceRail
          library={library}
          model={model}
          onAddPaper={model.paperActions.addPaper}
          onRemovePaper={model.paperActions.removePaper}
        />
      )}
      header={(
        <ChatWorkspaceHeader
          projectStore={projectStore}
          workspace={workspace}
          model={model}
        />
      )}
    >
      <ChatWorkspaceMain
        feedRef={feedRef}
        workspace={workspace}
        projectStore={projectStore}
        paperCount={model.effectiveResearchPapers.length}
      />
    </WorkspaceLayout>
  )
}
