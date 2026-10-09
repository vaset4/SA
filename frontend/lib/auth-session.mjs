/**
 * 认证会话传输字段的唯一归一化边界。
 *
 * @author xxx
 */

/**
 * 把登录或持久化 payload 归一化成前端 session。
 * @param {object|null|undefined} rawSession 原始会话数据
 * @returns {{accessToken:string,refreshToken:string,tokenType:string,expiresIn:number,user:object|null}} 规范会话
 */
export function normalizeAuthSession(rawSession) {
  const payload = rawSession || {}
  return {
    accessToken: payload.accessToken || payload.access_token || '',
    refreshToken: payload.refreshToken || payload.refresh_token || '',
    tokenType: payload.tokenType || payload.token_type || 'bearer',
    expiresIn: Number(payload.expiresIn || payload.expires_in || 0) || 0,
    user: payload.user || null,
  }
}

/**
 * 校验刷新响应是完整 session，禁止缺字段时保留旧 token。
 * @param {object|null|undefined} payload refresh 接口响应
 * @returns {ReturnType<typeof normalizeAuthSession>} 完整刷新会话
 */
export function normalizeRefreshSession(payload) {
  const session = normalizeAuthSession(payload)
  if (!session.accessToken || !session.refreshToken) {
    throw new Error('刷新响应缺少 token，请重新登录')
  }
  return session
}
