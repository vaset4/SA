import Link from 'next/link'
import { BRAND } from '../config/brand'

const FEATURE_CARDS = [
  {
    title: '项目工作面',
    description: '先用项目组织范围、论文池和研究沉淀，再决定今天应该推进哪一段工作。',
  },
  {
    title: '双研究入口',
    description: '内部研究回到独立对话页，外部研究单独承接外部证据链和历史记录。',
  },
  {
    title: '知识库与图谱',
    description: '知识库负责管理论文与资料，图谱负责看关系、缺口和下一步值得追踪的节点。',
  },
]

export default function HomePage() {
  return (
    <div className="sa-app-shell">
      <section className="sa-hero">
        <div className="sa-hero__copy">
          <span className="sa-hero__eyebrow">Research Workspace</span>
          <h1 className="sa-hero__title">{BRAND.name}</h1>
          <p className="sa-hero__description">{BRAND.description}</p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Link href={BRAND.cta.primary} className="sa-button sa-button--primary">
              开始使用
            </Link>
            <Link href={BRAND.cta.secondary} className="sa-button sa-button--ghost">
              管理论文库
            </Link>
          </div>
        </div>

        <aside className="sa-hero__aside">
          <div className="sa-panel-card sa-metric-card">
            <p className="sa-section-title">Why It Exists</p>
            <strong>从项目出发，集中管理论文、对话和研究沉淀。</strong>
            <p style={{ color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>
              打开后先选择研究主题，再进入论文库、对话或外部研究。
            </p>
          </div>
          {FEATURE_CARDS.map((item) => (
            <div key={item.title} className="sa-panel-card sa-metric-card">
              <p className="sa-section-title">{item.title}</p>
              <p style={{ margin: 0, color: 'var(--sa-text-soft)', lineHeight: 1.7 }}>{item.description}</p>
            </div>
          ))}
        </aside>
      </section>
    </div>
  )
}
