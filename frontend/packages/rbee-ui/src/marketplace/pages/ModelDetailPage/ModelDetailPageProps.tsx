// TEAM-401: Full model detail data
import type { ModelDetailTemplateProps } from '../../templates/ModelDetailTemplate'

export interface ModelDetailPageProps {
  seo?: {
    title: string
    description: string
  }
  template: ModelDetailTemplateProps
}
