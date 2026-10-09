'use client'

import { ProjectWorkspaceShell } from '../../../components/project/ProjectWorkspaceShell'
import { useProjectStore } from '../../../stores/project-store'

export default function ProjectsPage() {
  const projectStore = useProjectStore()
  return <ProjectWorkspaceShell projectStore={projectStore} />
}
