import { LoginPageClient } from '../../components/auth/LoginPageClient'
import { normalizeWorkspaceRedirect } from '../../lib/workspace-routes'
import Link from 'next/link'

export default function LoginPage({ searchParams }) {
  return <>
    <LoginPageClient redirectTarget={normalizeWorkspaceRedirect(searchParams?.redirect)} />
    <footer style={{ textAlign: 'center', padding: 16, fontSize: 12, color: 'var(--sa-text-soft)' }}>
      <Link href="/credits">第三方声明</Link>
    </footer>
  </>
}
