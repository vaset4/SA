'use client'

import { useState } from 'react'

export function MessageAnswerBriefSection({ answerBrief }) {
  // 回答简报默认折叠：intent/focus/禁止表述对普通阅读噪音大
  const [open, setOpen] = useState(false)
  if (!answerBrief || typeof answerBrief !== 'object') return null
  const intent = answerBrief.intent || ''
  const focus = answerBrief.focus || ''
  const uncertainties = Array.isArray(answerBrief.uncertainties)
    ? answerBrief.uncertainties.filter(Boolean)
    : []
  const forbiddenClaims = Array.isArray(answerBrief.forbidden_claims)
    ? answerBrief.forbidden_claims.filter(Boolean)
    : []

  if (!intent && !focus && !uncertainties.length && !forbiddenClaims.length) return null

  return (
    <div className="sa-stack" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <p className="sa-section-title" style={{ margin: 0 }}>回答简报</p>
        {intent ? <span className="sa-chip" style={{ fontSize: 12 }}>{intent}</span> : null}
        <button
          type="button"
          className="sa-button sa-button--ghost"
          style={{ fontSize: 12, minHeight: 28 }}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? '收起' : '查看'}
        </button>
      </div>
      {open ? (
        <>
          {focus ? (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-soft)' }}>
              聚焦：{typeof focus === 'string' ? focus : '已聚焦'}
            </p>
          ) : null}
          {uncertainties.length ? (
            <div style={{ marginTop: 6 }}>
              <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-text-faint)' }}>不确定性:</p>
              {uncertainties.map((item, index) => (
                <p key={index} style={{ margin: '2px 0 0 8px', fontSize: 12, color: 'var(--sa-text-soft)' }}>• {item}</p>
              ))}
            </div>
          ) : null}
          {forbiddenClaims.length ? (
            <div style={{ marginTop: 6, padding: '6px 10px', borderRadius: 6, background: 'color-mix(in srgb, var(--sa-danger, #dc2626) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--sa-danger, #dc2626) 20%, var(--sa-border))' }}>
              <p style={{ margin: 0, fontSize: 12, color: 'var(--sa-danger, #dc2626)' }}>禁止表述:</p>
              {forbiddenClaims.map((item, index) => (
                <p key={index} style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--sa-text-soft)' }}>• {item}</p>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
