import Link from 'next/link'

export const metadata = { title: '第三方声明 · 研途' }

/**
 * 【公开发布】【素材署名】展示无需登录即可访问的图标作者和许可
 * @returns {JSX.Element} 第三方声明页面
 * @author xxx
 */
export default function CreditsPage() {
  return (
    <main style={{ maxWidth: 760, margin: '0 auto', padding: '64px 24px', lineHeight: 1.9 }}>
      <Link href="/" style={{ color: 'var(--sa-accent)' }}>← 返回研途</Link>
      <h1 style={{ fontFamily: 'var(--sa-font-display)', fontSize: 36, margin: '32px 0 16px' }}>第三方声明</h1>
      <p style={{ color: 'var(--sa-text-soft)' }}>研途使用以下开源图标。感谢作者与贡献者的作品。</p>
      <section style={{ borderTop: '1px solid var(--border-color)', marginTop: 32, paddingTop: 24 }}>
        <h2 style={{ fontSize: 22 }}>Solar · 480 Design</h2>
        <p>采用 <a href="https://creativecommons.org/licenses/by/4.0/" className="underline">CC BY 4.0</a> 许可。</p>
        <p><a href="https://www.figma.com/community/file/1166831539721848736" className="underline">原作品</a> · <a href="https://icon-sets.iconify.design/solar/" className="underline">图标目录</a></p>
        <p>通过 Iconify 使用，以界面样式调整显示尺寸与颜色，未修改路径图形。作者未对本项目作认可或背书。</p>
      </section>
      <section style={{ borderTop: '1px solid var(--border-color)', marginTop: 24, paddingTop: 24 }}>
        <h2 style={{ fontSize: 22 }}>Lucide · Lucide Icons and Contributors</h2>
        <p>Copyright (c) 2026 Lucide Icons and Contributors。采用 ISC 许可。</p>
        <p>部分图标继承自 Feather，保留 Copyright (c) 2013-present Cole Bemis 和 MIT 许可。</p>
        <p><a href="https://lucide.dev/" className="underline">Lucide 项目</a> · <a href="https://github.com/lucide-icons/lucide/blob/main/LICENSE" className="underline">完整版权与许可声明</a></p>
      </section>
      <p style={{ borderTop: '1px solid var(--border-color)', marginTop: 32, paddingTop: 24, color: 'var(--sa-text-soft)' }}>
        StudyAssistant 原创代码采用 MIT。第三方组件与素材保留各自许可；完整声明和许可文本随源码仓库提供。
      </p>
    </main>
  )
}
