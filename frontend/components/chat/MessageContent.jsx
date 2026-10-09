'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import { hasFinalDelivery, prepareFinalCitations, remarkFinalCitations } from '../../lib/final-citations.mjs'

import {
  RENDER_MODE_MARKDOWN,
  resolveMessageRenderMode,
  sanitizeMessageContent,
} from '../../lib/study-assistant-workspace.mjs'

const SAFE_LINK_PATTERN = /^(https?:\/\/|mailto:)/i
const SAFE_EVIDENCE_ANCHOR_PATTERN = /^#evidence-[A-Za-z0-9_.:-]+$/
const FENCE_PATTERN = /^```/
const EVIDENCE_GROUP_PATTERN = /\[((?:E\d+(?:\s*,\s*)?)+)\](?!\()/g

// 打字机：约 16ms/字，整体不超过 2s，避免长文干等
const TYPEWRITER_CHAR_MS = 16
const TYPEWRITER_MAX_MS = 2000

function isSafeHref(href = '') {
  const cleanHref = href.trim()
  return SAFE_LINK_PATTERN.test(cleanHref) || SAFE_EVIDENCE_ANCHOR_PATTERN.test(cleanHref)
}

function normalizeMathDelimiters(content = '') {
  return `${content}`
    .replace(/\r\n/g, '\n')
    .split(/(```[\s\S]*?```)/g)
    .map((part) => {
      if (FENCE_PATTERN.test(part.trim())) return part
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => `\n$$\n${formula.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_, formula) => `$${formula.trim()}$`)
    })
    .join('')
}

function safeAnchorSegment(value = '') {
  return String(value || '').replace(/[^A-Za-z0-9_.:-]/g, '_')
}

function normalizeEvidenceLinks(content = '', messageId = '') {
  const anchorPrefix = `evidence-${safeAnchorSegment(messageId || 'message')}`
  return `${content}`
    .split(/(```[\s\S]*?```)/g)
    .map((part) => {
      if (FENCE_PATTERN.test(part.trim())) return part
      return part.replace(EVIDENCE_GROUP_PATTERN, (_, group) => (
        group
          .split(',')
          .map((marker) => marker.trim())
          .filter(Boolean)
          .map((marker) => `[${marker}](#${anchorPrefix}-${marker})`)
          .join(', ')
      ))
    })
    .join('')
}

function Link({ href = '', children }) {
  if (!isSafeHref(href)) {
    return <span>{children}</span>
  }
  if (href.trim().startsWith('#')) {
    // 证据角标样式：正文里 [E1] 可点击跳转，不打断阅读
    const label = typeof children === 'string' ? children : null
    const isEvidence = typeof href === 'string' && href.includes('#evidence-')
    return (
      <a
        href={href}
        className={isEvidence ? 'sa-evidence-ref' : undefined}
        style={isEvidence ? {
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          minWidth: 18,
          height: 18,
          padding: '0 5px',
          margin: '0 2px',
          borderRadius: 999,
          fontSize: 11,
          fontWeight: 600,
          lineHeight: 1,
          textDecoration: 'none',
          verticalAlign: 'super',
          color: 'var(--sa-accent, #2563eb)',
          background: 'color-mix(in srgb, var(--sa-accent, #2563eb) 12%, transparent)',
          border: '1px solid color-mix(in srgb, var(--sa-accent, #2563eb) 28%, transparent)',
        } : undefined}
      >
        {label || children}
      </a>
    )
  }
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  )
}

/** @param {object} props Markdown 链接节点 @returns {JSX.Element} 只允许本次合法位置生成的内部引用链接 */
function FinalLink({ href = '', node, children }) {
  if (href.startsWith('#') && node?.properties?.['data-evidence-binding'] !== href) {
    return <span>{children}</span>
  }
  return <Link href={href}>{children}</Link>
}

function Code({ className = '', children }) {
  return <code className={className || undefined}>{children}</code>
}

function Pre({ children }) {
  return <pre className="sa-richtext__code-block">{children}</pre>
}

function Table({ children }) {
  return (
    <div className="sa-richtext__table-wrap">
      <table>{children}</table>
    </div>
  )
}

const MARKDOWN_COMPONENTS = {
  a: Link,
  code: Code,
  pre: Pre,
  table: Table,
}
const FINAL_MARKDOWN_COMPONENTS = { ...MARKDOWN_COMPONENTS, a: FinalLink }

/**
 * 计算打字机每步间隔：总时长不超过上限，最短不低于 1ms。
 * @param {number} length 全文长度
 * @returns {number} 每字符毫秒
 */
function resolveTypewriterStepMs(length) {
  if (length <= 0) return TYPEWRITER_CHAR_MS
  const capped = Math.min(TYPEWRITER_CHAR_MS, Math.floor(TYPEWRITER_MAX_MS / length))
  return Math.max(1, capped)
}

/**
 * 前端打字机：定稿文本已在 message.content，仅做展示层逐字动画。
 * 卸载 / 切换消息时直接展示全文，不卡住。
 * @param {{ text: string, enabled: boolean }} props
 * @returns {{ visibleText: string, animating: boolean }}
 */
function useTypewriterText({ text, enabled }) {
  const [visibleLength, setVisibleLength] = useState(enabled ? 0 : text.length)
  const finishedRef = useRef(false)
  const textRef = useRef(text)

  useEffect(() => {
    textRef.current = text
    // 1. 未启用或已播完：直接全文
    if (!enabled || finishedRef.current) {
      setVisibleLength(text.length)
      return undefined
    }
    // 2. 空文本：无需动画
    if (!text) {
      setVisibleLength(0)
      finishedRef.current = true
      return undefined
    }
    // 3. 按步推进 visibleLength，到顶后标记完成
    setVisibleLength(0)
    const stepMs = resolveTypewriterStepMs(text.length)
    let index = 0
    const timer = setInterval(() => {
      index += 1
      if (index >= textRef.current.length) {
        setVisibleLength(textRef.current.length)
        finishedRef.current = true
        clearInterval(timer)
        return
      }
      setVisibleLength(index)
    }, stepMs)
    return () => {
      // 仅清定时器；不在 cleanup 里标记 finished，避免 React Strict Mode 双调用直接跳过动画
      clearInterval(timer)
    }
  }, [text, enabled])

  const safeLength = Math.min(visibleLength, text.length)
  return {
    visibleText: text.slice(0, safeLength),
    animating: enabled && safeLength < text.length,
  }
}

/** @param {object} props 消息正文、身份和等待文案 @returns {JSX.Element} 保留同次原始坐标的安全正文 */
function renderBody({ content, message, fallback, citationProjection }) {
  if (!content) {
    return <div className="sa-message-text">{fallback}</div>
  }

  if (resolveMessageRenderMode(message) !== RENDER_MODE_MARKDOWN) {
    return <div className="sa-message-text">{content}</div>
  }

  const finalDelivery = hasFinalDelivery(message)
  const plugins = finalDelivery
    ? [remarkMath, remarkGfm, [remarkFinalCitations, { content, bindings: citationProjection.bindings }]]
    : [remarkMath, remarkGfm]
  return (
    <div className="sa-richtext">
      <ReactMarkdown
        remarkPlugins={plugins}
        rehypePlugins={[rehypeKatex]}
        components={finalDelivery ? FINAL_MARKDOWN_COMPONENTS : MARKDOWN_COMPONENTS}
      >
        {finalDelivery ? content : normalizeMathDelimiters(normalizeEvidenceLinks(content, message?.id))}
      </ReactMarkdown>
      {citationProjection.errors.length ? <p role="status">{citationProjection.errors.join('；')}</p> : null}
    </div>
  )
}

/** @param {object} props 消息与等待文案 @returns {JSX.Element} 正文及按真实生成状态控制的展示动画 */
export function MessageContent({ message, fallback = '等待回复...' }) {
  // 1. 【研究交付】【原始正文】新版坐标绑定原 Markdown，显示前不删除标记或改写换行
  const fullContent = hasFinalDelivery(message) ? (message?.content || '') : sanitizeMessageContent(message?.content || '')
  // 2. 【研究交付】【位置投影】完整正文只核对一次，打字动画不逐字重建全文码点映射
  const citationProjection = useMemo(() => prepareFinalCitations(message, message?.content || ''), [message])
  // 仅定稿成功且本次落地标记 typewriter 时做逐字；失败态/历史恢复整块显示
  const typewriterEnabled = Boolean(
    message?.typewriter
    && message?.source_status?.planner_final === 'ok'
    && fullContent,
  )
  const { visibleText, animating } = useTypewriterText({
    text: fullContent,
    enabled: typewriterEnabled,
  })

  // pending 占位且尚无正文：显示等待文案
  if (!fullContent && message?.pending) {
    return <div className="sa-message-text">{fallback}</div>
  }

  return (
    <>
      {renderBody({ content: visibleText, message, fallback, citationProjection })}
      {animating ? (
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 6,
            height: 14,
            marginLeft: 2,
            verticalAlign: 'text-bottom',
            background: 'var(--sa-accent, #2563eb)',
            animation: 'sa-typewriter-caret 0.8s steps(1) infinite',
          }}
        />
      ) : null}
    </>
  )
}
