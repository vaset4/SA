'use client'

export function ProjectScopePanel({ scope }) {
  if (!scope) {
    return <p style={{ margin: 0, color: 'var(--sa-text-soft)', fontSize: 13 }}>当前项目还没有配置作用域。</p>
  }

  const visibleLibraries = (scope.library_ids || []).filter((item) => item && item !== 'default')
  const taskFilters = scope.task_filters || []
  const methodFilters = scope.method_filters || []
  const hasAdvancedScope = Boolean(
    visibleLibraries.length
    || taskFilters.length
    || methodFilters.length
    || scope.year_from
    || scope.year_to,
  )

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {hasAdvancedScope ? (
        <div className="sa-chip-list">
          {visibleLibraries.map((item) => <span key={item} className="sa-chip">库: {item}</span>)}
          {taskFilters.map((item) => <span key={item} className="sa-chip">任务: {item}</span>)}
          {methodFilters.map((item) => <span key={item} className="sa-chip">方法: {item}</span>)}
        </div>
      ) : (
        <p className="sa-note">
          当前项目继承默认知识库范围，没有额外任务、方法或时间筛选。
        </p>
      )}
      <p className="sa-note">
        论文范围：{(scope.paper_ids || []).length} 篇
        {scope.year_from || scope.year_to ? ` · 时间：${scope.year_from || '不限'} - ${scope.year_to || '不限'}` : ''}
      </p>
    </div>
  )
}
