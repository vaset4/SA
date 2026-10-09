import { LLMConfigCenter } from '../../../components/settings/LLMConfigCenter'
import { PageHeader } from '../../../components/shared/PageHeader'
import { StatusBadge } from '../../../components/shared/StatusBadge'

export default function SettingsPage() {
  return (
    <div style={{ minHeight: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      <PageHeader
        eyebrow="Settings"
        variant="workspace"
        title="设置"
        description="管理 LLM 上游供应商与工作台偏好。"
        status={<StatusBadge tone="neutral">偏好设置</StatusBadge>}
      />

      <main className="sa-workbench">
        <section className="sa-section-card">
          <div className="sa-section-card__header">
            <p className="sa-section-card__eyebrow">LLM Providers</p>
            <h2 className="sa-section-card__title">模型服务配置</h2>
            <p className="sa-section-card__description">
              管理多套上游供应商预设（地址、密钥、模型与专属重试规则），一键切换即时生效，无需重启服务。
            </p>
          </div>
          <LLMConfigCenter />
        </section>
      </main>
    </div>
  )
}
