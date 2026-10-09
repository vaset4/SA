'use client'

import { useEffect, useMemo, useState } from 'react'
import { Button } from '@heroui/react'
import { StatusBadge } from '../shared/StatusBadge'
import { formatResearchTraceDetail, formatResearchTraceTitle } from '../../lib/research-trace'

function formatTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function asList(values) {
  return Array.isArray(values) ? values.filter(Boolean) : []
}

export function TraceReplayPanel({ events = [] }) {
  const orderedEvents = useMemo(() => asList(events), [events])
  const signature = useMemo(
    () => orderedEvents.map((event) => `${event.stage || event.type || 'event'}-${event.at || ''}`).join('|'),
    [orderedEvents],
  )
  const [replayMode, setReplayMode] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [replayIndex, setReplayIndex] = useState(0)
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    setReplayMode(false)
    setPlaying(false)
    setReplayIndex(0)
  }, [signature])

  useEffect(() => {
    if (!replayMode || !playing) return
    if (replayIndex >= orderedEvents.length - 1) {
      setPlaying(false)
      return
    }
    const timeout = window.setTimeout(() => {
      setReplayIndex((current) => Math.min(current + 1, orderedEvents.length - 1))
    }, speed === 2 ? 420 : 900)
    return () => window.clearTimeout(timeout)
  }, [orderedEvents.length, playing, replayIndex, replayMode, speed])

  const startReplay = () => {
    if (!orderedEvents.length) return
    setReplayMode(true)
    setReplayIndex(0)
    setPlaying(true)
  }

  const visibleEvents = replayMode
    ? orderedEvents.slice(0, replayIndex + 1)
    : orderedEvents.slice().reverse()

  const currentEvent = replayMode ? orderedEvents[replayIndex] : null
  const replayComplete = replayMode && replayIndex >= Math.max(orderedEvents.length - 1, 0) && !playing

  if (!orderedEvents.length) {
    return <p style={{ margin: 0, color: 'var(--sa-text-soft)', fontSize: 13 }}>当前没有额外轨迹事件。</p>
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {!replayMode ? (
            <Button size="sm" variant="flat" color="primary" onPress={startReplay}>
              开始回放
            </Button>
          ) : (
            <>
              <Button size="sm" variant="flat" color="primary" onPress={() => setPlaying((current) => !current)}>
                {playing ? '暂停' : '继续'}
              </Button>
              <Button
                size="sm"
                variant="light"
                onPress={() => {
                  setReplayIndex(0)
                  setPlaying(true)
                }}
              >
                重新开始
              </Button>
              <Button
                size="sm"
                variant="light"
                onPress={() => {
                  setReplayMode(false)
                  setPlaying(false)
                  setReplayIndex(0)
                }}
              >
                退出回放
              </Button>
            </>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <StatusBadge tone={replayMode ? (playing ? 'warning' : 'success') : 'neutral'}>
            {replayMode ? `回放 ${Math.min(replayIndex + 1, orderedEvents.length)}/${orderedEvents.length}` : `共 ${orderedEvents.length} 条事件`}
          </StatusBadge>
          {replayMode ? (
            <>
              <Button size="sm" variant={speed === 1 ? 'flat' : 'light'} onPress={() => setSpeed(1)}>
                1x
              </Button>
              <Button size="sm" variant={speed === 2 ? 'flat' : 'light'} onPress={() => setSpeed(2)}>
                2x
              </Button>
            </>
          ) : null}
        </div>
      </div>

      {replayMode ? (
        <div
          style={{
            padding: 12,
            borderRadius: 14,
            border: '1px solid var(--sa-border)',
            background: replayComplete
              ? 'color-mix(in srgb, var(--sa-accent) 10%, var(--sa-surface-soft))'
              : 'var(--sa-surface-soft)',
            display: 'grid',
            gap: 4,
          }}
        >
          <strong style={{ fontSize: 13 }}>
            {replayComplete ? '回放完成' : (currentEvent ? formatResearchTraceTitle(currentEvent) : '准备回放')}
          </strong>
          <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
            {replayComplete
              ? '当前已完成整条研究时间线回放，可以重新开始或退出回放。'
              : (currentEvent ? formatResearchTraceDetail(currentEvent) : '将按时间顺序逐步还原研究轨迹。')}
          </span>
          {currentEvent?.at ? (
            <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>{formatTime(currentEvent.at)}</span>
          ) : null}
        </div>
      ) : null}

      <div style={{ display: 'grid', gap: 8 }}>
        {visibleEvents.map((event, index) => {
          const isActive = replayMode && index === visibleEvents.length - 1
          return (
            <article
              key={`${event.at || event.stage || 'trace'}-${index}`}
              style={{
                padding: 12,
                borderRadius: 12,
                border: '1px solid var(--sa-border)',
                background: isActive
                  ? 'color-mix(in srgb, var(--sa-accent) 9%, white)'
                  : 'rgba(255,255,255,0.82)',
                display: 'grid',
                gap: 4,
              }}
            >
              <strong style={{ fontSize: 13 }}>{formatResearchTraceTitle(event)}</strong>
              <span style={{ fontSize: 12, color: 'var(--sa-text-soft)' }}>
                {formatResearchTraceDetail(event)}
              </span>
              {event.at ? (
                <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>{formatTime(event.at)}</span>
              ) : null}
            </article>
          )
        })}
      </div>
    </div>
  )
}
