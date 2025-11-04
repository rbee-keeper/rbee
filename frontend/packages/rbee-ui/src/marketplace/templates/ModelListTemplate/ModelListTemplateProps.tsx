// TEAM-401: Props for model list template
import type { ModelCardProps } from '../../organisms/ModelCard'

export interface ModelListTemplateProps {
  title: string
  description?: string
  models: Array<ModelCardProps['model']>
  filters?: {
    search: string
    sort: string
  }
  sortOptions?: Array<{ value: string; label: string }>
  onFilterChange?: (filters: { search: string; sort: string }) => void
  onModelAction?: (modelId: string) => void
  isLoading?: boolean
  error?: string
  emptyMessage?: string
  emptyDescription?: string
}

// Default sort options for models
export const defaultModelSortOptions = [
  { value: 'popular', label: 'Most Popular' },
  { value: 'recent', label: 'Recently Added' },
  { value: 'downloads', label: 'Most Downloads' },
  { value: 'likes', label: 'Most Liked' },
  { value: 'name', label: 'Name (A-Z)' },
]
