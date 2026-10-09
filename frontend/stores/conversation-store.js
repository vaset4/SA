'use client'

import { createContext, useContext, useEffect, useMemo, useReducer } from 'react'
import {
  createConversationRecord,
  deleteConversationRecord,
  getConversation,
  listConversations,
  updateConversationRecord,
} from '../lib/api/conversations'
import { createSession } from '../lib/api/research'
import { StorageService } from '../lib/storage-service'
import { assertAuthContext, isCurrentAuthContext } from '../lib/auth-client'
import { useAuthStore } from './auth-store'
import { sameMessages } from '../utils/message-utils'
import { pickMessageExtras } from '../lib/study-assistant-workspace.mjs'

function normalizeConversation(conversation = {}) {
  return {
    id: conversation.id || '',
    sessionId: StorageService.normalizeSessionId(conversation.session_id || conversation.sessionId),
    title: conversation.title || '新对话',
    createdAt: conversation.created_at || conversation.createdAt || new Date().toISOString(),
    updatedAt: conversation.updated_at || conversation.updatedAt || new Date().toISOString(),
    isPinned: Boolean(conversation.is_pinned ?? conversation.isPinned),
    tags: conversation.tags || [],
    associatedPapers: conversation.associated_papers || conversation.associatedPapers || [],
    messageCount: Number(conversation.message_count ?? conversation.messageCount ?? 0),
  }
}

function normalizeMessages(messages = []) {
  return messages.map((message, index) => {
    const extras = message.role === 'assistant' ? pickMessageExtras(message) : {}
    return {
      id: message.id || `msg-${index}`,
      role: message.role,
      content: message.content,
      title: message.title || undefined,
      tone: message.tone || undefined,
      renderMode: message.render_mode || message.renderMode || undefined,
      ...extras,
    }
  })
}

function sortConversations(conversations) {
  return [...conversations].sort((left, right) => {
    if (left.isPinned && !right.isPinned) return -1
    if (!left.isPinned && right.isPinned) return 1
    return new Date(right.updatedAt) - new Date(left.updatedAt)
  })
}

const initialState = {
  conversations: [],
  activeId: null,
  messagesById: {},
  isLoading: true,
  error: '',
}

function reducer(state, action) {
  switch (action.type) {
    case 'setState':
      return { ...state, ...action.patch }
    default:
      return state
  }
}

const ConversationStoreContext = createContext(null)

export function ConversationStoreProvider({ children }) {
  const auth = useAuthStore()
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    let active = true
    const context = auth.sessionContext

    /** 【对话】【账号恢复】无参数，逐步恢复当前账号的列表和消息；返回完成信号 */
    async function hydrate() {
      if (!auth.ready || !isCurrentAuthContext(context)) return
      if (!auth.isAuthenticated) {
        StorageService.saveActiveConversationId('')
        if (!active || !isCurrentAuthContext(context)) return
        dispatch({
          type: 'setState',
          patch: {
            conversations: [],
            activeId: null,
            messagesById: {},
            isLoading: false,
            error: '',
          },
        })
        return
      }

      dispatch({ type: 'setState', patch: { isLoading: true, error: '' } })

      try {
        let conversations = (await listConversations())?.conversations?.map(normalizeConversation) || []
        if (!active || !isCurrentAuthContext(context)) return

        if (!conversations.length) {
          const sessionPayload = await createSession({ topic: '默认对话' })
          if (!active || !isCurrentAuthContext(context)) return
          const created = await createConversationRecord({
            title: '默认对话',
            sessionId: sessionPayload?.session_id || '',
          })
          if (!active || !isCurrentAuthContext(context)) return
          conversations = [normalizeConversation(created)]
        }

        const savedActiveId = StorageService.getActiveConversationId()
        const activeId = savedActiveId && conversations.some((item) => item.id === savedActiveId)
          ? savedActiveId
          : conversations[0]?.id ?? null

        let messagesById = {}
        if (activeId) {
          const detail = await getConversation({ conversationId: activeId })
          if (!active || !isCurrentAuthContext(context)) return
          messagesById = {
            [activeId]: normalizeMessages(detail?.messages || []),
          }
        }

        if (!active || !isCurrentAuthContext(context)) return
        StorageService.saveActiveConversationId(activeId || '')
        dispatch({
          type: 'setState',
          patch: {
            conversations: sortConversations(conversations),
            activeId,
            messagesById,
            isLoading: false,
            error: '',
          },
        })
      } catch (error) {
        if (!active || !isCurrentAuthContext(context)) return
        dispatch({
          type: 'setState',
          patch: {
            conversations: [],
            activeId: null,
            messagesById: {},
            isLoading: false,
            error: error instanceof Error ? error.message : '对话列表加载失败',
          },
        })
      }
    }

    hydrate()

    return () => {
      active = false
    }
  }, [auth.isAuthenticated, auth.ready, auth.sessionContext])

  useEffect(() => {
    if (!state.activeId || !isCurrentAuthContext(auth.sessionContext)) return
    StorageService.saveActiveConversationId(state.activeId)
  }, [state.activeId, auth.sessionContext])

  const actions = useMemo(() => {
    const context = auth.sessionContext
    /** 【对话】【详情读取】@param {string} conversationId 对话 ID @returns {Promise<object|null>} 当前账号的对话 */
    const loadConversationDetail = async (conversationId) => {
      assertAuthContext(context)
      if (!conversationId) return null
      const detail = await getConversation({ conversationId })
      assertAuthContext(context)
      const normalizedConversation = normalizeConversation(detail)
      const normalizedMessages = normalizeMessages(detail?.messages || [])
      dispatch({
        type: 'setState',
        patch: {
          conversations: sortConversations(
            state.conversations.map((item) => (item.id === conversationId ? { ...item, ...normalizedConversation } : item)),
          ),
          messagesById: {
            ...state.messagesById,
            [conversationId]: normalizedMessages,
          },
        },
      })
      return normalizedConversation
    }

    /** 【对话】【本地更新】@param {string} conversationId 对话 ID @param {object|Function} updater 修改 @returns {object|null} 更新结果 */
    const updateConversationLocally = (conversationId, updater) => {
      if (!isCurrentAuthContext(context)) return null
      const currentConversation = state.conversations.find((item) => item.id === conversationId)
      if (!currentConversation) return null
      const patch = typeof updater === 'function' ? updater(currentConversation) : updater
      if (!patch || patch === currentConversation) return currentConversation
      const nextConversation = {
        ...currentConversation,
        ...patch,
        updatedAt: patch.updatedAt || new Date().toISOString(),
      }
      dispatch({
        type: 'setState',
        patch: {
          conversations: sortConversations(
            state.conversations.map((item) => (item.id === conversationId ? nextConversation : item)),
          ),
        },
      })
      return nextConversation
    }

    return {
      /** 【对话】【账号隔离】@param {string} activeId 选中对话 ID @returns {Promise<void>} 加载选中对话 */
      async setActiveId(activeId) {
        assertAuthContext(context)
        dispatch({ type: 'setState', patch: { activeId } })
        if (activeId && !state.messagesById[activeId]) {
          try {
            await loadConversationDetail(activeId)
          } catch (error) {
            if (!isCurrentAuthContext(context)) return
            dispatch({
              type: 'setState',
              patch: { error: error instanceof Error ? error.message : '对话详情加载失败' },
            })
          }
        }
      },
      /** 【对话】【账号隔离】@param {string} title 标题 @param {object} options 返回选项 @returns {Promise<string|object>} 新对话 */
      async createNew(title = '新对话', options = {}) {
        assertAuthContext(context)
        const sessionPayload = await createSession({ topic: title })
        assertAuthContext(context)
        const created = await createConversationRecord({
          title,
          sessionId: sessionPayload?.session_id || '',
        })
        assertAuthContext(context)
        const newConversation = normalizeConversation(created)
        dispatch({
          type: 'setState',
          patch: {
            conversations: sortConversations([...state.conversations, newConversation]),
            activeId: newConversation.id,
            messagesById: {
              ...state.messagesById,
              [newConversation.id]: [],
            },
            error: '',
          },
        })
        return options?.returnConversation ? newConversation : newConversation.id
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @returns {Promise<boolean>} 删除结果 */
      async deleteConversation(conversationId) {
        assertAuthContext(context)
        await deleteConversationRecord({ conversationId })
        assertAuthContext(context)
        const conversations = state.conversations.filter((item) => item.id !== conversationId)
        const nextActiveId = state.activeId === conversationId ? conversations[0]?.id ?? null : state.activeId
        const nextMessages = { ...state.messagesById }
        delete nextMessages[conversationId]
        dispatch({
          type: 'setState',
          patch: {
            conversations,
            activeId: nextActiveId,
            messagesById: nextMessages,
          },
        })
        return true
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @param {string} title 标题 @returns {Promise<void>} 保存完成 */
      async updateTitle(conversationId, title) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation) return
        const nextTitle = title.trim()
        if (!nextTitle || nextTitle === currentConversation.title) return
        await updateConversationRecord({
          conversationId,
          title: nextTitle,
          isPinned: currentConversation.isPinned,
          tags: currentConversation.tags,
          associatedPapers: currentConversation.associatedPapers,
          sessionId: currentConversation.sessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { title: nextTitle })
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @param {string} sessionId 研究会话 ID @returns {Promise<void>} 关联完成 */
      async setSessionId(conversationId, sessionId) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation) return
        const normalizedSessionId = StorageService.normalizeSessionId(sessionId)
        if (currentConversation.sessionId === normalizedSessionId) return
        await updateConversationRecord({
          conversationId,
          title: currentConversation.title,
          isPinned: currentConversation.isPinned,
          tags: currentConversation.tags,
          associatedPapers: currentConversation.associatedPapers,
          sessionId: normalizedSessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { sessionId: normalizedSessionId })
      },
      /** 【对话】【消息更新】@param {string} conversationId 对话 ID @param {object[]} messages 消息 @param {object} options 时间更新选项 @returns {void} 更新当前账号消息 */
      setMessages(conversationId, messages, options = {}) {
        if (!isCurrentAuthContext(context)) return
        const currentMessages = state.messagesById[conversationId] || []
        if (sameMessages(currentMessages, messages)) return
        const bumpUpdatedAt = options.bumpUpdatedAt !== false
        dispatch({
          type: 'setState',
          patch: {
            messagesById: {
              ...state.messagesById,
              [conversationId]: messages,
            },
            conversations: sortConversations(
              state.conversations.map((item) => (
                item.id === conversationId
                  ? {
                    ...item,
                    messageCount: messages.length,
                    updatedAt: bumpUpdatedAt ? new Date().toISOString() : item.updatedAt,
                  }
                  : item
              )),
            ),
          },
        })
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @param {string} paperId 论文 ID @returns {Promise<void>} 添加关联 */
      async addPaper(conversationId, paperId) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation || currentConversation.associatedPapers.includes(paperId)) return
        const associatedPapers = [...currentConversation.associatedPapers, paperId]
        await updateConversationRecord({
          conversationId,
          title: currentConversation.title,
          isPinned: currentConversation.isPinned,
          tags: currentConversation.tags,
          associatedPapers,
          sessionId: currentConversation.sessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { associatedPapers })
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @param {string} paperId 论文 ID @returns {Promise<void>} 移除关联 */
      async removePaper(conversationId, paperId) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation || !currentConversation.associatedPapers.includes(paperId)) return
        const associatedPapers = currentConversation.associatedPapers.filter((item) => item !== paperId)
        await updateConversationRecord({
          conversationId,
          title: currentConversation.title,
          isPinned: currentConversation.isPinned,
          tags: currentConversation.tags,
          associatedPapers,
          sessionId: currentConversation.sessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { associatedPapers })
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @returns {Promise<void>} 保存置顶状态 */
      async togglePin(conversationId) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation) return
        const isPinned = !currentConversation.isPinned
        await updateConversationRecord({
          conversationId,
          title: currentConversation.title,
          isPinned,
          tags: currentConversation.tags,
          associatedPapers: currentConversation.associatedPapers,
          sessionId: currentConversation.sessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { isPinned, updatedAt: currentConversation.updatedAt })
      },
      /** 【对话】【账号隔离】@param {string} conversationId 对话 ID @param {string[]} tags 标签 @returns {Promise<void>} 保存标签 */
      async updateTags(conversationId, tags) {
        assertAuthContext(context)
        const currentConversation = state.conversations.find((item) => item.id === conversationId)
        if (!currentConversation) return
        await updateConversationRecord({
          conversationId,
          title: currentConversation.title,
          isPinned: currentConversation.isPinned,
          tags,
          associatedPapers: currentConversation.associatedPapers,
          sessionId: currentConversation.sessionId,
        })
        assertAuthContext(context)
        updateConversationLocally(conversationId, { tags })
      },
      /** 【对话】【账号隔离】@returns {Promise<object|null>} 当前账号选中对话的最新内容 */
      async refreshActiveConversation() {
        assertAuthContext(context)
        if (!state.activeId) return null
        return loadConversationDetail(state.activeId)
      },
    }
  }, [state.activeId, state.conversations, state.messagesById, auth.sessionContext])

  const value = useMemo(() => {
    const activeConversation = state.conversations.find((item) => item.id === state.activeId)
    return {
      ...state,
      activeConversation: activeConversation
        ? { ...activeConversation, messages: state.messagesById[activeConversation.id] || [] }
        : null,
      recentConversations: state.conversations.slice(0, 5),
      ...actions,
    }
  }, [actions, state])

  return (
    <ConversationStoreContext.Provider value={value}>
      {children}
    </ConversationStoreContext.Provider>
  )
}

export function useConversationStore() {
  const context = useContext(ConversationStoreContext)
  if (!context) {
    throw new Error('useConversationStore must be used within ConversationStoreProvider')
  }
  return context
}
