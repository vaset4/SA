'use client'

import { createContext, useContext, useEffect, useMemo, useReducer } from 'react'
import { listNotifications, markNotificationRead } from '../lib/api/notifications'
import { getError } from '../lib/study-assistant-workspace.mjs'

const initialState = {
  activeInsightSection: 'explore',
  railCollapsed: false,
  notifications: [],
  readIds: [],
  notificationsBusy: false,
  notificationsError: '',
}

function reducer(state, action) {
  switch (action.type) {
    case 'setState':
      return { ...state, ...action.patch }
    default:
      return state
  }
}

const UIStoreContext = createContext(null)

export function UIStoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    let active = true
    dispatch({ type: 'setState', patch: { notificationsBusy: true } })
    listNotifications()
      .then((data) => {
        if (!active) return
        dispatch({
          type: 'setState',
          patch: {
            notificationsBusy: false,
            notifications: data?.notifications ?? [],
            notificationsError: '',
          },
        })
      })
      .catch((reason) => {
        if (!active) return
        dispatch({
          type: 'setState',
          patch: {
            notificationsBusy: false,
            notificationsError: getError(reason, '通知加载失败'),
          },
        })
      })

    return () => {
      active = false
    }
  }, [])

  const actions = useMemo(() => ({
    setActiveInsightSection(activeInsightSection) {
      dispatch({ type: 'setState', patch: { activeInsightSection } })
    },
    setRailCollapsed(railCollapsed) {
      dispatch({ type: 'setState', patch: { railCollapsed } })
    },
    async refreshNotifications() {
      dispatch({ type: 'setState', patch: { notificationsBusy: true, notificationsError: '' } })
      try {
        const data = await listNotifications()
        dispatch({
          type: 'setState',
          patch: {
            notificationsBusy: false,
            notifications: data?.notifications ?? [],
          },
        })
      } catch (reason) {
        dispatch({
          type: 'setState',
          patch: {
            notificationsBusy: false,
            notificationsError: getError(reason, '通知加载失败'),
          },
        })
      }
    },
    async markRead(logId) {
      try {
        await markNotificationRead({ logId })
        dispatch({
          type: 'setState',
          patch: {
            readIds: state.readIds.includes(logId) ? state.readIds : [...state.readIds, logId],
          },
        })
      } catch (reason) {
        dispatch({
          type: 'setState',
          patch: {
            notificationsError: getError(reason, '通知标记失败'),
          },
        })
      }
    },
  }), [state.readIds])

  const value = useMemo(() => ({
    ...state,
    ...actions,
  }), [actions, state])

  return (
    <UIStoreContext.Provider value={value}>
      {children}
    </UIStoreContext.Provider>
  )
}

export function useUIStore() {
  const context = useContext(UIStoreContext)
  if (!context) {
    throw new Error('useUIStore must be used within UIStoreProvider')
  }
  return context
}
