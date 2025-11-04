// TEAM-401: ALL page data (perfect for SSG)
import type { ModelListTemplateProps } from '../../templates/ModelListTemplate'

export interface ModelsPageProps {
  seo?: {
    title: string
    description: string
  }
  template: ModelListTemplateProps
}

// Example props for SSG
export const defaultModelsPageProps: ModelsPageProps = {
  seo: {
    title: 'Browse AI Models | rbee Marketplace',
    description:
      'Discover and download AI models for your local inference needs. Browse thousands of open-source models optimized for CPU, CUDA, and Metal.',
  },
  template: {
    title: 'AI Models',
    description: 'Browse and download models for local inference',
    models: [],
    filters: { search: '', sort: 'popular' },
  },
}
