'use client'

import { useEffect, useMemo, useState } from 'react'
import { WorkspaceLayout } from '../../../../components/layout/WorkspaceLayout'
import { ResearchSessionActions } from '../../../../components/research-session/ResearchSessionActions'
import { ResearchSessionLayout } from '../../../../components/research-session/ResearchSessionLayout'
import { EmptyState } from '../../../../components/shared/EmptyState'
import { PageHeader } from '../../../../components/shared/PageHeader'
import { StatusBadge } from '../../../../components/shared/StatusBadge'
import { getSessionDetail } from '../../../../lib/api/research'
import {
  DEVELOPER_ROLES,
  buildResearchSessionModel,
  headerStatus,
} from '../../../../lib/research-session-view'
import { useAuthStore } from '../../../../stores/auth-store'
import { useWorkspaceStore } from '../../../../stores/workspace-store'

import { hasPendingCandidates, candidateRecoveryError } from '../../../../lib/result-delivery.mjs'

const ACTIVE_SESSION_STATUSES = new Set(['created', 'running', 'awaiting_clarification'])
const SESSION_POLL_INTERVAL_MS = 2000

function useResearchSession(sessionId) {
  const [state, setState] = useState({ loading: true, error: '', session: null })
  const auth = useAuthStore()
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let active = true
    let timer = null
    setState({ loading: true, error: '', session: null })

    const load = async () => {
      try {
        const session = await getSessionDetail({ sessionId })
        if (!active) return
        const recoveryError = candidateRecoveryError(session?.current_result)
        setState({ loading: false, error: recoveryError, session })
        if (!recoveryError && (ACTIVE_SESSION_STATUSES.has(session?.status) || hasPendingCandidates(session?.current_result))) {
          timer = setTimeout(load, SESSION_POLL_INTERVAL_MS)
        }
      } catch (reason) {
        if (!active) return
        setState((current) => ({
          loading: false,
          error: reason.message || '研究会话加载失败',
          session: current.session,
        }))
      }
    }

    load()
    return () => {
      active = false
      if (timer) clearTimeout(timer)
    }
  }, [reloadToken, sessionId, auth.sessionContext])

  return { ...state, reload: () => setReloadToken((value) => value + 1) }
}

function SessionContent({ state, model, isDeveloper }) {
  if (state.loading) {
    return <EmptyState title="正在加载研究会话" description="稍后会展示报告、证据与研究过程。" />
  }
  if (state.error && !model) {
    return (
      <EmptyState
        title="研究会话不可用"
        description={state.error}
        action={<button type="button" className="sa-button sa-button--ghost" onClick={state.reload}>重新获取</button>}
      />
    )
  }
  if (!model) {
    return <EmptyState title="研究会话为空" description="当前 session 没有可展示的数据。" />
  }
  return <>
    {state.error ? <div role="alert">{state.error}<button type="button" onClick={state.reload}>重新查询</button></div> : null}
    <ResearchSessionLayout model={model} isDeveloper={isDeveloper} />
  </>
}

function ResearchSessionView({ sessionId }) {
  const workspace = useWorkspaceStore()
  const auth = useAuthStore()
  const state = useResearchSession(sessionId)
  const isDeveloper = DEVELOPER_ROLES.has(auth.user?.role)
  const liveTrace = workspace.researchTrace?.sessionId === sessionId ? workspace.researchTrace : null
  const model = useMemo(() => {
    if (!state.session) return null
    return buildResearchSessionModel({ sessionId, session: state.session, liveTrace })
  }, [liveTrace, sessionId, state.session])
  const status = headerStatus(state, model || {})

  return (
    <WorkspaceLayout
      focusMode
      shellClassName="sa-research-room-shell"
      contentClassName="sa-main-scroll--page"
      header={(
        <PageHeader
          title={model?.title || `研究会话 ${sessionId}`}
          description="报告优先，证据和研究过程随后复核。"
          status={<StatusBadge tone={status.tone}>{status.label}</StatusBadge>}
          actions={model ? <ResearchSessionActions model={model} onSessionChange={state.reload} /> : null}
        />
      )}
    >
      <SessionContent state={state} model={model} isDeveloper={isDeveloper} />
    </WorkspaceLayout>
  )
}

export default function ResearchSessionPage({ params }) {
  return <ResearchSessionView sessionId={params.sessionId} />
}
