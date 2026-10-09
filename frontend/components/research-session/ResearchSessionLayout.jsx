'use client'

import { EvidenceDeskSection } from './EvidenceDeskSection'
import { ResearchContextSection } from './ResearchContextSection'
import { ResearchProcessSection } from './ResearchProcessSection'
import { ResearchReportBody } from './ResearchReportBody'
import { ResearchSessionSidePanel } from './ResearchSessionSidePanel'

export function ResearchSessionLayout({ model, isDeveloper }) {
  return (
    <div className="sa-research-session-layout">
      <main className="sa-research-session-main">
        <ResearchReportBody model={model} />
        <EvidenceDeskSection model={model} />
        <ResearchContextSection model={model} />
        <ResearchProcessSection model={model} isDeveloper={isDeveloper} />
      </main>
      <ResearchSessionSidePanel model={model} />
    </div>
  )
}
