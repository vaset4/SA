/**
 * Storage Service for Study Assistant
 * Restricts browser persistence to auth and lightweight UI state only.
 */

const AUTH_SESSION_KEY = 'sa-auth-session'
const ACTIVE_CONVERSATION_KEY = 'sa-active-conversation'
const UI_PREFERENCES_KEY = 'sa-ui-preferences'

export class StorageService {
  static normalizeSessionId(sessionId) {
    if (!sessionId || typeof sessionId !== 'string') return ''
    return sessionId.startsWith('session-') ? '' : sessionId
  }

  static getItem(key) {
    if (typeof window === 'undefined') return null
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : null
    } catch (error) {
      console.error(`Error reading from localStorage key "${key}":`, error)
      return null
    }
  }

  static setItem(key, value) {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch (error) {
      console.error(`Error writing to localStorage key "${key}":`, error)
    }
  }

  static removeItem(key) {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.removeItem(key)
    } catch (error) {
      console.error(`Error removing from localStorage key "${key}":`, error)
    }
  }

  static getActiveConversationId() {
    if (typeof window === 'undefined') return null
    return window.localStorage.getItem(ACTIVE_CONVERSATION_KEY)
  }

  static saveActiveConversationId(id) {
    if (typeof window === 'undefined') return
    if (!id) {
      window.localStorage.removeItem(ACTIVE_CONVERSATION_KEY)
      return
    }
    window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, id)
  }

  static getAuthSession() {
    return this.getItem(AUTH_SESSION_KEY)
  }

  static saveAuthSession(session) {
    this.setItem(AUTH_SESSION_KEY, session)
  }

  static clearAuthSession() {
    this.removeItem(AUTH_SESSION_KEY)
  }

  static getUiPreferences() {
    return this.getItem(UI_PREFERENCES_KEY) || {}
  }

  static saveUiPreferences(preferences) {
    this.setItem(UI_PREFERENCES_KEY, preferences || {})
  }
}
