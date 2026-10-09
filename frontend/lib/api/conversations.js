import { requestJson } from './core'
import { MESSAGE_EXTRA_KEYS } from '../study-assistant-workspace.mjs'

function messageMetadata(message) {
  const metadata = {}
  for (const key of MESSAGE_EXTRA_KEYS) {
    if (message?.[key] !== undefined) {
      metadata[key] = message[key]
    }
  }
  return metadata
}

export const listConversations = ({ limit = 50, offset = 0, sort = 'updated_at' } = {}) =>
  requestJson({ path: `/api/v1/conversations?limit=${limit}&offset=${offset}&sort=${sort}` })

export const createConversationRecord = ({ title = '新对话', associatedPapers = [], sessionId = '' } = {}) =>
  requestJson({
    path: '/api/v1/conversations',
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        associated_papers: associatedPapers,
        session_id: sessionId || null,
      }),
    },
  })

export const getConversation = ({ conversationId }) =>
  requestJson({ path: `/api/v1/conversations/${conversationId}` })

export const updateConversationRecord = ({
  conversationId,
  title,
  isPinned,
  tags,
  associatedPapers,
  sessionId,
}) =>
  requestJson({
    path: `/api/v1/conversations/${conversationId}`,
    init: {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        is_pinned: isPinned,
        tags,
        associated_papers: associatedPapers,
        session_id: sessionId || null,
      }),
    },
  })

export const deleteConversationRecord = ({ conversationId }) =>
  requestJson({
    path: `/api/v1/conversations/${conversationId}`,
    init: { method: 'DELETE' },
  })

export const setConversationMessages = ({ conversationId, messages, sessionId = '' }) =>
  requestJson({
    path: `/api/v1/conversations/${conversationId}/messages`,
    init: {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({
            role: m.role,
            content: m.content,
            title: m.title || null,
            tone: m.tone || null,
            render_mode: m.renderMode || m.render_mode || null,
            metadata: messageMetadata(m),
          })),
        session_id: sessionId || null,
      }),
    },
  })

export const getConversationMessages = ({ conversationId }) =>
  requestJson({ path: `/api/v1/conversations/${conversationId}` })
    .then((data) => data?.messages || [])
