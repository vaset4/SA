'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../stores/auth-store'
import { DEFAULT_WORKSPACE_ENTRY } from '../../lib/workspace-routes'

export function LoginPageClient({ redirectTarget = DEFAULT_WORKSPACE_ENTRY }) {
  const auth = useAuthStore()
  const router = useRouter()
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [localError, setLocalError] = useState('')

  useEffect(() => {
    if (!auth.ready || !auth.isAuthenticated) return
    router.replace(redirectTarget)
  }, [auth.isAuthenticated, auth.ready, redirectTarget, router])

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLocalError('')
    try {
      if (mode === 'register') {
        await auth.register({ username, password })
      } else {
        await auth.login({ username, password })
      }
      router.replace(redirectTarget)
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : '认证失败')
    }
  }

  return (
    <div className="sa-app-shell" style={{ minHeight: '100vh', padding: 24 }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 420px)',
          gap: 24,
          width: 'min(1120px, 100%)',
          margin: '0 auto',
          alignItems: 'stretch',
        }}
      >
        <section className="sa-panel-card" style={{ padding: 32, display: 'grid', gap: 18 }}>
          <span className="sa-hero__eyebrow">Account Access</span>
          <h1 className="sa-hero__title" style={{ margin: 0 }}>登录后进入项目驱动的研究工作台</h1>
          <p className="sa-hero__description" style={{ margin: 0 }}>
            会话、论文、项目和研究记录都已经切到服务端账号作用域。登录后会先回到项目页，再进入内部研究或外部研究。
          </p>
          <div className="sa-stack" style={{ gap: 10 }}>
            <div className="sa-chip">账号隔离</div>
            <div className="sa-chip">登录状态自动维护</div>
            <div className="sa-chip">会话与项目同步</div>
          </div>
          <Link href="/" className="sa-button sa-button--ghost" style={{ width: 'fit-content' }}>
            返回首页
          </Link>
        </section>

        <section className="sa-panel-card" style={{ padding: 24 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
            <button
              type="button"
              className={`sa-button ${mode === 'login' ? 'sa-button--primary' : 'sa-button--ghost'}`}
              onClick={() => setMode('login')}
            >
              登录
            </button>
            <button
              type="button"
              className={`sa-button ${mode === 'register' ? 'sa-button--primary' : 'sa-button--ghost'}`}
              onClick={() => setMode('register')}
            >
              注册
            </button>
          </div>

          <form onSubmit={handleSubmit} className="sa-stack" style={{ gap: 14 }}>
            <label className="sa-stack" style={{ gap: 6 }}>
              <span className="sa-section-title">用户名</span>
              <input
                className="sa-input"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="至少 3 个字符"
                autoComplete="username"
              />
            </label>

            <label className="sa-stack" style={{ gap: 6 }}>
              <span className="sa-section-title">密码</span>
              <input
                className="sa-input"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="至少 6 个字符"
                autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
              />
            </label>

            {(localError || auth.error) ? (
              <div className="sa-panel-card" style={{ padding: 12, color: 'var(--sa-danger, #b91c1c)' }}>
                {localError || auth.error}
              </div>
            ) : null}

            <button type="submit" className="sa-button sa-button--primary" disabled={auth.busy}>
              {auth.busy ? '处理中...' : mode === 'register' ? '注册并进入项目工作台' : '登录'}
            </button>
          </form>
        </section>
      </div>
    </div>
  )
}
