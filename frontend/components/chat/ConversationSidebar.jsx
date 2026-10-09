'use client'

import Link from 'next/link'
import { useState } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { buildInternalResearchHref } from '../../lib/workspace-routes'
import {
  buildProjectScopeItems,
  buildProjectSessionItems,
  buildUnassignedConversationItems,
} from '../../lib/chat-research-scope'
import { useConversationStore } from '../../stores/conversation-store'
import { useProjectStore } from '../../stores/project-store'
import { useProjectSessionIndex } from './use-project-session-index'

function ScopeCard({ item, onSelect }) {
  return (
    <button type="button" className="sa-research-room-scope-card" data-active={item.active} onClick={() => onSelect(item)}>
      <strong>{item.title}</strong>
      <span>{item.description}</span>
    </button>
  )
}

function SessionCard({ item, onSelect }) {
  const content = (
    <>
      <strong>{item.title}</strong>
      <span>{item.description}</span>
    </>
  )
  if (!item.conversationId) {
    return <Link href={item.href} className="sa-research-room-session-card">{content}</Link>
  }
  return (
    <button type="button" className="sa-research-room-session-card" data-active={item.active} onClick={() => onSelect(item)}>
      {content}
    </button>
  )
}

function EditableConversation({ item, titleDraft, setTitleDraft, onSave }) {
  return (
    <div className="sa-stack">
      <input
        className="sa-input"
        value={titleDraft}
        onChange={(event) => setTitleDraft(event.target.value)}
      />
      <button type="button" className="sa-button" onClick={() => onSave(item)}>
        保存标题
      </button>
    </div>
  )
}

function ConversationActions({ item, onEdit, onTogglePin, onDelete }) {
  if (!item.conversationId) return null
  return (
    <div className="sa-conversation-item__actions">
      <button type="button" className="sa-button sa-button--ghost" onClick={() => onTogglePin(item)}>
        置顶
      </button>
      <button type="button" className="sa-button sa-button--ghost" onClick={() => onEdit(item)}>
        编辑
      </button>
      <button type="button" className="sa-button sa-button--ghost" onClick={() => onDelete(item)}>
        删除
      </button>
    </div>
  )
}

function SessionList({ label, items, emptyText, editingId, titleDraft, setTitleDraft, onSelect, onEdit, onSave, onTogglePin, onDelete }) {
  return (
    <>
      <p className="sa-research-room-label">{label}</p>
      <div className="sa-research-room-session-list">
        {!items.length ? <div className="sa-research-room-session-card"><span>{emptyText}</span></div> : null}
        {items.map((item) => (
          <div key={item.id} className="sa-stack" style={{ gap: 8 }}>
            {editingId === item.conversationId ? (
              <EditableConversation item={item} titleDraft={titleDraft} setTitleDraft={setTitleDraft} onSave={onSave} />
            ) : (
              <>
                <SessionCard item={item} onSelect={onSelect} />
                <ConversationActions item={item} onEdit={onEdit} onTogglePin={onTogglePin} onDelete={onDelete} />
              </>
            )}
          </div>
        ))}
      </div>
    </>
  )
}

function SidebarMetrics({ projectStore }) {
  return (
    <div className="sa-research-room-metrics">
      <div className="sa-research-room-metric">
        <strong>{projectStore.summary?.totalPapers || 0}</strong>
        <span>项目论文</span>
      </div>
      <div className="sa-research-room-metric">
        <strong>{projectStore.sessions?.length || 0}</strong>
        <span>项目会话</span>
      </div>
    </div>
  )
}

function ScopePicker({ projectStore, onSelect }) {
  return (
    <>
      <p className="sa-research-room-label">项目范围</p>
      <div className="sa-research-room-scope-list">
        {buildProjectScopeItems(projectStore).map((item) => (
          <ScopeCard key={item.id} item={item} onSelect={onSelect} />
        ))}
      </div>
    </>
  )
}

function ContinueProjectLink() {
  return (
    <>
      <p className="sa-research-room-label">范围说明</p>
      <Link href={buildInternalResearchHref({ prompt: '请基于当前项目材料继续推进下一轮内部研究。' })} className="sa-active-output-item">
        <strong>从当前项目继续</strong>
        <span>进入对话后会保留当前项目范围。</span>
      </Link>
    </>
  )
}

function SessionSections({ items, editing, actions }) {
  return (
    <>
      <SessionList
        label="会话"
        items={items.projectSessionItems}
        emptyText="当前项目还没有关联研究会话。"
        editingId={editing.editingId}
        titleDraft={editing.titleDraft}
        setTitleDraft={editing.setTitleDraft}
        {...actions}
      />
      <SessionList
        label="未归属"
        items={items.unassignedItems}
        emptyText="没有未归属会话。"
        editingId={editing.editingId}
        titleDraft={editing.titleDraft}
        setTitleDraft={editing.setTitleDraft}
        {...actions}
      />
    </>
  )
}

function indexNotice(index) {
  if (index.error) return <p className="sa-note" style={{ color: 'var(--sa-danger)' }}>项目会话归属索引加载失败：{index.error}</p>
  if (index.loading) return <p className="sa-note">项目会话归属索引加载中...</p>
  return null
}

export function ConversationSidebar() {
  const router = useRouter()
  const pathname = usePathname()
  const conversations = useConversationStore()
  const projectStore = useProjectStore()
  const [editingId, setEditingId] = useState(null)
  const [titleDraft, setTitleDraft] = useState('')
  const index = useProjectSessionIndex({ projects: projectStore.projects, activeProjectId: projectStore.activeProjectId, activeSessions: projectStore.sessions || [] })
  const items = {
    projectSessionItems: buildProjectSessionItems({ projectSessions: projectStore.sessions || [], conversations: conversations.conversations, activeConversationId: conversations.activeId }),
    unassignedItems: buildUnassignedConversationItems({ conversations: conversations.conversations, linkedSessionIds: index.linkedSessionIds, activeConversationId: conversations.activeId }),
  }
  const actions = useSidebarActions({ router, pathname, conversations, projectStore, setEditingId, setTitleDraft, titleDraft })
  return (
    <div className="sa-conversation-sidebar">
      <SidebarMetrics projectStore={projectStore} />
      <button type="button" className="sa-button sa-button--primary" onClick={actions.onCreate} disabled={conversations.isLoading}>新对话</button>
      <ScopePicker projectStore={projectStore} onSelect={actions.onSelectScope} />
      {indexNotice(index)}
      <SessionSections items={items} editing={{ editingId, titleDraft, setTitleDraft }} actions={actions} />
      <ContinueProjectLink />
    </div>
  )
}

function useSidebarActions({ router, pathname, conversations, projectStore, setEditingId, setTitleDraft, titleDraft }) {
  return {
    async onCreate() {
      const createdConversation = await conversations.createNew('新对话', { returnConversation: true })
      const conversationId = typeof createdConversation === 'string' ? createdConversation : createdConversation?.id || ''
      const sessionId = typeof createdConversation === 'string' ? '' : createdConversation?.sessionId || ''
      if (!conversationId) {
        throw new Error('新对话创建失败：缺少 conversation_id')
      }
      if (projectStore.activeProjectId && !sessionId) {
        throw new Error('新对话无法绑定当前项目：缺少 session_id')
      }
      if (projectStore.activeProjectId) {
        await projectStore.attachSession(sessionId)
      }
      router.push(`/chat/${conversationId}`)
    },
    onSelectScope(item) {
      projectStore.setActiveProjectId(item.type === 'unassigned' ? '' : item.id)
    },
    onSelect(item) {
      if (!item.conversationId || (item.conversationId === conversations.activeId && pathname === `/chat/${item.conversationId}`)) return
      conversations.setActiveId(item.conversationId)
      router.push(`/chat/${item.conversationId}`)
    },
    onEdit(item) {
      setEditingId(item.conversationId)
      setTitleDraft(item.title)
    },
    onSave(item) {
      conversations.updateTitle(item.conversationId, titleDraft.trim() || item.title)
      setEditingId(null)
    },
    onTogglePin(item) {
      conversations.togglePin(item.conversationId)
    },
    onDelete(item) {
      conversations.deleteConversation(item.conversationId)
    },
  }
}
