import { Suspense } from 'react'
import { ChatWorkspaceShell } from '../../../components/chat/ChatWorkspaceShell'

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatWorkspaceShell />
    </Suspense>
  )
}
