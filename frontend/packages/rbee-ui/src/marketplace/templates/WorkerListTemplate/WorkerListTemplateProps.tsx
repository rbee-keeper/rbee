// TEAM-401: Props for worker list template
import type { WorkerCardProps } from '../../organisms/WorkerCard'

export interface WorkerListTemplateProps {
  title: string
  description?: string
  workers: Array<WorkerCardProps['worker']>
  filters?: {
    search: string
    sort: string
  }
  sortOptions?: Array<{ value: string; label: string }>
  onFilterChange?: (filters: { search: string; sort: string }) => void
  onWorkerAction?: (workerId: string) => void
  isLoading?: boolean
  error?: string
  emptyMessage?: string
  emptyDescription?: string
}

// Default sort options for workers
export const defaultWorkerSortOptions = [
  { value: 'name', label: 'Name (A-Z)' },
  { value: 'version', label: 'Latest Version' },
  { value: 'type', label: 'Worker Type' },
]
