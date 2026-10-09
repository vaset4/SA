'use client'

import { isProjectResearchTurn, summarizeTurnCapabilities } from '../../lib/research-turn.mjs'

// 三种色调：正常绿 / 中性灰 / 出问题红。empty 与 failed 不共用色调，
// 「项目里没有相关判断」和「存储不可用」要一眼分得开
const TONE_STYLE = {
  ok: { color: 'var(--sa-accent, #16a34a)' },
  neutral: { color: 'var(--sa-text-soft)' },
  danger: { color: 'var(--sa-danger, #b91c1c)' },
}

/**
 * 回答卡里的「本轮结果」一栏：把本轮各项能力的真实终态逐行摆出来。
 *
 * 只渲染后端真下发过的能力项 —— 拿不到的不出现，也不补一个「正常」冒充；
 * 未使用记忆、无检索结果、证据缺口、写回失败都当成正常展示内容，不藏。
 *
 * @param {{ message: object }} props
 */
export function MessageTurnSummary({ message }) {
  if (!isProjectResearchTurn(message)) return null
  const rows = summarizeTurnCapabilities(message)
  if (!rows.length) return null

  return (
    <div className="sa-stack" style={{ marginTop: 8, display: 'grid', gap: 4 }}>
      {rows.map((row) => (
        <div key={row.key} style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'baseline', fontSize: 12 }}>
          <span style={{ fontWeight: 500, ...TONE_STYLE[row.tone] }}>{row.label}</span>
          <span style={{ color: 'var(--sa-text-faint)' }}>{row.detail}</span>
        </div>
      ))}
    </div>
  )
}
