'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../stores/auth-store'
import { Skeleton } from '../shared/Skeleton'

const DEVELOPER_ROLES = new Set(['developer', 'admin'])

export function DeveloperOnlyRoute({ children, redirectTo = '/projects' }) {
  const auth = useAuthStore()
  const router = useRouter()
  const allowed = auth.ready && DEVELOPER_ROLES.has(auth.user?.role)

  useEffect(() => {
    if (!auth.ready || allowed) return
    router.replace(redirectTo)
  }, [allowed, auth.ready, redirectTo, router])

  if (!auth.ready || !allowed) {
    return (
      <main className="sa-workbench">
        <section className="sa-section-card">
          <Skeleton rows={4} height={18} />
        </section>
      </main>
    )
  }

  return children
}
