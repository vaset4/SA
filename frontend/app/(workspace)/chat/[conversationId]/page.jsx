import { Suspense } from 'react'
import { ChatWorkspaceShell } from '../../../../components/chat/ChatWorkspaceShell'

export default function ConversationChatPage({ params }) {
  return (
    <Suspense fallback={null}>
      <ChatWorkspaceShell conversationId={params.conversationId} />
    </Suspense>
  )
}
