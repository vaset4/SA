'use client'

function formatTime(value) {
  if (!value) return '刚刚'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚'
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function toneForActivity(type) {
  switch (type) {
    case 'analysis_completed':
      return 'rgba(242, 249, 255, 0.96)'
    case 'internal_research_started':
      return 'rgba(246, 245, 244, 0.96)'
    default:
      return 'rgba(255,255,255,0.94)'
  }
}

export function ProjectActivityFeed({ activities = [], onRefresh = null, refreshing = false }) {
  const visibleActivities = Array.isArray(activities) ? activities.slice(0, 8) : []

  return (
    <section className="sa-section-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="sa-section-card__header">
          <span className="sa-section-card__eyebrow">
            Activity
          </span>
          <div>
            <h3 className="sa-section-card__title">项目动态</h3>
            <p className="sa-section-card__description">
              右侧先收三类稳定事件：论文加入项目、AI 分析完成、内部研究启动。
            </p>
          </div>
        </div>
        {typeof onRefresh === 'function' ? (
          <button type="button" className="sa-button sa-button--ghost" onClick={() => onRefresh()} disabled={refreshing}>
            {refreshing ? '刷新中' : '刷新动态'}
          </button>
        ) : null}
      </div>

      {!visibleActivities.length ? (
        <p className="sa-note">
          当前还没有可回看的项目动态。先加论文、补分析或发起第一次内部研究，动态流就会开始累积。
        </p>
      ) : (
        <div className="sa-timeline">
          {visibleActivities.map((activity) => (
            <article
              key={activity.id}
              className="sa-timeline__item"
              style={{
                background: toneForActivity(activity.type),
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <strong style={{ fontSize: 13 }}>{activity.label}</strong>
                <span style={{ fontSize: 11, color: 'var(--sa-text-faint)' }}>{formatTime(activity.at)}</span>
              </div>
              <span style={{ fontSize: 14 }}>{activity.title}</span>
              <span style={{ fontSize: 12, lineHeight: 1.75, color: 'var(--sa-text-soft)' }}>
                {activity.description}
              </span>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
