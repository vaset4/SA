import './globals.css'
import 'katex/dist/katex.min.css'
import { Providers } from './providers'
import '../styles/design-tokens.css'
import '../styles/components.css'
import '../styles/batch4-visual-system.css'
import '../styles/batch4-chat-workspace.css'
import '../styles/batch4-evidence-progress.css'
import '../styles/batch4-research-session.css'

export const metadata = {
  title: '研途',
  description: '项目内研究工作台'
}

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
