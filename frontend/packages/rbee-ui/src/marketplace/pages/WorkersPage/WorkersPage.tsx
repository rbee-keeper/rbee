// TEAM-401: DUMB page - just renders template
import { WorkerListTemplate } from '../../templates/WorkerListTemplate'
import type { WorkersPageProps } from './WorkersPageProps'

export function WorkersPage({ template }: WorkersPageProps) {
  return <WorkerListTemplate {...template} />
}
