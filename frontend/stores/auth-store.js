'use client'

import { Fragment, createContext, useContext, useEffect, useMemo, useReducer } from 'react'
import {
  assertAuthContext,
  captureAuthContext,
  clearAuthSession,
  getAuthSession,
  isCurrentAuthContext,
  setAuthSession,
  subscribeAuthSession,
} from '../lib/auth-client'
import {
  getCurrentUser,
  loginWithPassword,
  logoutCurrentSession,
  registerWithPassword,
} from '../lib/api/auth'

/** 【认证】【初始化】无参数，返回本次挂载时的会话及页面状态 */
function createInitialState() {
  return { ready: false, busy: false, error: '', session: getAuthSession(), sessionContext: captureAuthContext() }
}

/** @param {object} state 当前状态 @param {object} action 状态事件 @returns {object} 新状态 */
function reducer(state, action) {
  if (action.type === 'sessionChanged') {
    const boundaryChanged = state.sessionContext.version !== action.context.version
    return {
      ...state,
      session: action.session,
      sessionContext: boundaryChanged ? action.context : state.sessionContext,
      ...(boundaryChanged ? { ready: true, busy: false, error: action.reason?.message || '' } : {}),
    }
  }
  if (action.type === 'setState') return { ...state, ...action.patch }
  return state
}

const AuthStoreContext = createContext(null)

/**
 * 【认证】【恢复会话】只把当前用户资料写回发起恢复的同一会话
 * @param {object} context 发起时的会话上下文
 * @param {{dispatch: Function, signal: AbortSignal}} options 状态更新及卸载信号
 * @returns {Promise<void>} 完成恢复或记录当前会话的真实错误
 */
async function restoreCurrentSession(context, { dispatch, signal }) {
  try {
    const data = await getCurrentUser({ signal })
    if (signal.aborted) return
    assertAuthContext(context)
    setAuthSession(
      { ...getAuthSession(), user: data?.user },
      { expectedContext: context, preserveContext: true },
    )
    dispatch({ type: 'setState', patch: { ready: true, error: '' } })
  } catch (error) {
    if (signal.aborted || !isCurrentAuthContext(context)) return
    console.error('【认证】【恢复会话】当前用户读取失败:', error)
    if (error.code === 'AUTH_SESSION_INVALID' || error.status === 401) {
      clearAuthSession({ expectedContext: context, redirect: false, reason: error })
    }
    dispatch({
      type: 'setState',
      patch: { ready: true, error: error instanceof Error ? error.message : '当前用户读取失败' },
    })
  }
}

/** 【认证】【恢复订阅】@param {Function} dispatch 状态更新入口 @returns {void} 挂载恢复与卸载清理 */
function useAuthBootstrap(dispatch) {
  useEffect(() => {
    const unsubscribe = subscribeAuthSession((session, reason) => {
      dispatch({ type: 'sessionChanged', session, context: captureAuthContext(), reason })
    })
    const controller = new AbortController()
    const context = captureAuthContext()
    if (getAuthSession().accessToken) {
      void restoreCurrentSession(context, { dispatch, signal: controller.signal })
    } else {
      dispatch({ type: 'setState', patch: { ready: true } })
    }
    return () => {
      controller.abort()
      unsubscribe()
    }
  }, [dispatch])
}

/**
 * 【认证】【账号登录】登录或注册的晚到响应不能覆盖期间切换的新会话
 * @param {Function} authenticate 登录或注册接口
 * @param {object} credentials 用户输入的账号密码
 * @param {{context: object, dispatch: Function}} options 原会话及状态更新入口
 * @returns {Promise<object>} 已提交的完整会话
 */
async function submitAuthentication(authenticate, credentials, { context, dispatch }) {
  assertAuthContext(context)
  dispatch({ type: 'setState', patch: { busy: true, error: '' } })
  try {
    const payload = await authenticate(credentials)
    assertAuthContext(context)
    setAuthSession(payload, { expectedContext: context })
    dispatch({ type: 'setState', patch: { busy: false, ready: true, error: '' } })
    return payload
  } catch (error) {
    if (isCurrentAuthContext(context)) {
      dispatch({
        type: 'setState',
        patch: { busy: false, error: error instanceof Error ? error.message : '账号认证失败' },
      })
    }
    throw error
  }
}

/**
 * 【认证】【退出吊销】远端退出完成后只清理发起退出的会话
 * @param {object} context 原会话上下文
 * @param {{dispatch: Function, redirect: boolean}} options 状态更新及跳转策略
 * @returns {Promise<void>} 清理当前会话；远端失败保留可见错误
 */
async function submitLogout(context, { dispatch, redirect }) {
  assertAuthContext(context)
  dispatch({ type: 'setState', patch: { busy: true, error: '' } })
  let remoteError = ''
  try {
    await logoutCurrentSession()
  } catch (error) {
    if (!isCurrentAuthContext(context)) return
    console.error('【认证】【退出吊销】远端账号会话吊销失败:', error)
    remoteError = '已退出本机，但远端会话未确认吊销，请重新登录后重试。'
  }
  // 1. 【认证】【退出吊销】旧账号的退出响应不能清理新登录的账号
  if (!isCurrentAuthContext(context)) return
  clearAuthSession({ expectedContext: context, redirect: remoteError ? false : redirect })
  dispatch({ type: 'setState', patch: { busy: false, error: remoteError } })
}

/** @param {{children: React.ReactNode}} props 账号相关子树 @returns {React.ReactNode} 认证状态提供器 */
export function AuthStoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, undefined, createInitialState)
  useAuthBootstrap(dispatch)

  const actions = useMemo(() => ({
    /** @param {object} credentials 登录信息 @returns {Promise<object>} 新会话 */
    login(credentials) {
      return submitAuthentication(loginWithPassword, credentials, { context: state.sessionContext, dispatch })
    },
    /** @param {object} credentials 注册信息 @returns {Promise<object>} 新会话 */
    register(credentials) {
      return submitAuthentication(registerWithPassword, credentials, { context: state.sessionContext, dispatch })
    },
    /** @param {{redirect?: boolean}} options 退出跳转选项 @returns {Promise<void>} 退出完成 */
    logout({ redirect = true } = {}) {
      return submitLogout(state.sessionContext, { dispatch, redirect })
    },
  }), [state.sessionContext])

  const value = useMemo(() => ({
    ...state,
    ...actions,
    accessToken: state.session.accessToken,
    refreshToken: state.session.refreshToken,
    user: state.session.user,
    isAuthenticated: Boolean(state.session.accessToken),
  }), [actions, state])

  // 1. 【认证】【账号切换】重建账号相关页面和 store；令牌续期保留同一版本
  return (
    <AuthStoreContext.Provider value={value}>
      <Fragment key={state.sessionContext.version}>{children}</Fragment>
    </AuthStoreContext.Provider>
  )
}

/** 【认证】【状态读取】无参数，返回当前认证状态和账号操作 */
export function useAuthStore() {
  const context = useContext(AuthStoreContext)
  if (!context) {
    throw new Error('useAuthStore must be used within AuthStoreProvider')
  }
  return context
}
