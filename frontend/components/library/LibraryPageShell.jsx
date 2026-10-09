'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@heroui/react'
import { PageHeader } from '../shared/PageHeader'
import { StatusBadge } from '../shared/StatusBadge'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { PaperUploadPanel } from './PaperUploadPanel'
import { PaperCardList } from './PaperCardList'

export function LibraryPageShell({ library, tools, onAddToConversation, onAddToProject, projectId = '', initialQuery = '' }) {
  const [query, setQuery] = useState(initialQuery)
  const [sortBy, setSortBy] = useState('title')
  const [viewMode, setViewMode] = useState('card')
  const [pendingDeletePaperId, setPendingDeletePaperId] = useState('')
  const router = useRouter()

  const closeDeleteConfirmation = () => setPendingDeletePaperId('')
  const confirmDeletePaper = () => {
    if (!pendingDeletePaperId) return
    const paperId = pendingDeletePaperId
    closeDeleteConfirmation()
    tools.remove(paperId, library.refresh)
  }
  const pendingDeletePaper = library.papers?.find((paper) => paper?.paper_id === pendingDeletePaperId)

  useEffect(() => {
    setQuery(initialQuery)
  }, [initialQuery])

  return (
    <div className="sa-library-page">
      <PageHeader
        eyebrow="Library"
        variant="workspace"
        title="知识库"
        status={(
          <>
            <StatusBadge tone={library.error ? 'danger' : 'neutral'}>{library.error ? '异常' : '正常'}</StatusBadge>
            <span className="sa-chip">{library.papers?.length || 0} 篇</span>
          </>
        )}
        actions={(
          <Button size="sm" variant="flat" onPress={() => library.refresh(library.selectedPaperId)}>
            刷新
          </Button>
        )}
      />

      <main className="sa-workbench sa-library-workbench">
        <PaperUploadPanel tools={tools} library={library} projectId={projectId} />
        <PaperCardList
          papers={library.papers}
          loading={library.loading}
          query={query}
          setQuery={setQuery}
          sortBy={sortBy}
          setSortBy={setSortBy}
          viewMode={viewMode}
          setViewMode={setViewMode}
          onAddToConversation={onAddToConversation}
          onAddToProject={onAddToProject}
          onOpenGraph={() => { router.push('/graph') }}
          onDelete={(paperId) => setPendingDeletePaperId(paperId)}
        />
      </main>
      <ConfirmDialog
        open={Boolean(pendingDeletePaperId)}
        title="确认从知识库删除论文？"
        description={`将清理原文、索引、图谱及关联记录${pendingDeletePaper?.title ? `：${pendingDeletePaper.title}` : ''}。`}
        confirmLabel="确认删除"
        onCancel={closeDeleteConfirmation}
        onConfirm={confirmDeletePaper}
      />
    </div>
  )
}
