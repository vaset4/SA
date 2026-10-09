'use client'

export function ProjectSwitcher({ project, projects, onSelect }) {
  return (
    <label style={{ display: 'grid', gap: 8, minWidth: 220 }}>
      <span style={{ fontSize: 12, color: 'var(--sa-text-soft)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>当前项目</span>
      <select
        value={project?.project_id || ''}
        onChange={(event) => onSelect(event.target.value)}
        style={{
          border: '1px solid var(--sa-border)',
          borderRadius: 14,
          padding: '12px 14px',
          background: 'rgba(255, 252, 248, 0.96)',
          color: 'var(--sa-text)',
          boxShadow: 'var(--sa-shadow-sm)',
        }}
      >
        <option value="">无项目</option>
        {projects.map((item) => (
          <option key={item.project_id} value={item.project_id}>{item.name}</option>
        ))}
      </select>
    </label>
  )
}
