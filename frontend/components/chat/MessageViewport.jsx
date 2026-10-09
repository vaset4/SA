'use client'

import { MessageAnswerShell } from './MessageAnswerShell'
import { MessageContent } from './MessageContent'
import { EmptyState } from '../shared/EmptyState'

/**
 * 对话消息列表：用户气泡 + 助手答案主卡（次要信息收纳在主卡内）。
 * @param {{ messages: object[] }} props
 */
export function MessageViewport({ messages }) {
  const actualMessages = messages.filter((item) => item.role !== 'system' || item.content !== '欢迎使用学习助手！')

  if (!actualMessages.length) {
    return (
      <EmptyState
        title="开始研究对话"
        description="先写下问题，再从当前项目材料里收束证据。"
      />
    )
  }

  return (
    <div className="sa-message-list">
      {actualMessages.map((message) => {
        const isAssistant = message.role === 'assistant'
        return (
          <div key={message.id} className="sa-message" data-role={message.role}>
            <div className="sa-message__avatar">{message.role === 'user' ? '你' : '研究'}</div>
            <div className="sa-message__body">
              {message.role === 'user' && message.title ? (
                <h3 style={{ marginTop: 0 }}>{message.title}</h3>
              ) : null}
              {isAssistant ? (
                <MessageAnswerShell message={message} />
              ) : (
                <MessageContent message={message} />
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
