import { Skeleton } from '../../components/shared/Skeleton'

export default function WorkspaceLoading() {
  return (
    <main className="sa-workbench">
      <section className="sa-section-card">
        <Skeleton rows={5} height={20} />
      </section>
    </main>
  )
}
