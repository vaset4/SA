'use client'

import { ConversationStoreProvider } from './conversation-store'
import { LibraryStoreProvider } from './library-store'
import { ProjectStoreProvider } from './project-store'
import { WorkspaceStoreProvider } from './workspace-store'
import { UIStoreProvider } from './ui-store'

export function StudyAssistantStoreProvider({ children }) {
  return (
    <ConversationStoreProvider>
      <ProjectStoreProvider>
        <LibraryStoreProvider>
          <WorkspaceStoreProvider>
            <UIStoreProvider>
              {children}
            </UIStoreProvider>
          </WorkspaceStoreProvider>
        </LibraryStoreProvider>
      </ProjectStoreProvider>
    </ConversationStoreProvider>
  )
}
