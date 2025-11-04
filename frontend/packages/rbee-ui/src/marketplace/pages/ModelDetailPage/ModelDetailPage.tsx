// TEAM-401: DUMB page
import { ModelDetailTemplate } from '../../templates/ModelDetailTemplate'
import type { ModelDetailPageProps } from './ModelDetailPageProps'

export function ModelDetailPage({ template }: ModelDetailPageProps) {
  return <ModelDetailTemplate {...template} />
}
