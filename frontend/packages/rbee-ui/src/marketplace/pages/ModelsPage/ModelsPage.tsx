// TEAM-401: DUMB page - just renders template
import { ModelListTemplate } from '../../templates/ModelListTemplate'
import type { ModelsPageProps } from './ModelsPageProps'

export function ModelsPage({ template }: ModelsPageProps) {
  return <ModelListTemplate {...template} />
}
