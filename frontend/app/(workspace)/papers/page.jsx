import { Suspense } from 'react'
import { PapersPageClient } from '../../../components/library/PapersPageClient'

export default function PapersPage() {
  return (
    <Suspense fallback={null}>
      <PapersPageClient />
    </Suspense>
  )
}
