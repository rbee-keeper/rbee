// TEAM-405: Model list with table template
//! Complete template combining FilterBar + ModelTable
//! Pure presentation + control, NO data fetching
//! NOTE: This component uses hooks and requires 'use client' in consuming code

import { FilterBar } from '../../organisms/FilterBar'
import { ModelTable } from '../../organisms/ModelTable'
import type { ModelTableItem } from '../../organisms/ModelTable'
import { useModelFilters } from '../../hooks'
import type { UseModelFiltersOptions } from '../../hooks'

export interface ModelListTableTemplateProps {
  /** Models to display */
  models: ModelTableItem[]
  
  /** Called when a model is clicked */
  onModelClick?: (modelId: string) => void
  
  /** Loading state */
  isLoading?: boolean
  
  /** Error message */
  error?: string
  
  /** Empty state message */
  emptyMessage?: string
  
  /** Empty state description */
  emptyDescription?: string
  
  /** Filter options */
  filterOptions?: UseModelFiltersOptions
  
  /** Controlled filters (optional) */
  filters?: {
    search: string
    sort: string
    tags: string[]
  }
  
  /** Filter change handler (for controlled mode) */
  onFiltersChange?: (filters: { search: string; sort: string; tags: string[] }) => void
}

/**
 * Complete model list template with filtering and table
 * 
 * Can be used in two modes:
 * 1. **Uncontrolled** (manages its own filter state)
 * 2. **Controlled** (filters passed from parent)
 * 
 * @example Uncontrolled (for client-side filtering)
 * ```tsx
 * <ModelListTableTemplate
 *   models={allModels}
 *   onModelClick={(id) => navigate(`/models/${id}`)}
 * />
 * ```
 * 
 * @example Controlled (for server-side filtering)
 * ```tsx
 * const [filters, setFilters] = useState({ search: '', sort: 'downloads', tags: [] })
 * const { data: models } = useQuery({
 *   queryKey: ['models', filters],
 *   queryFn: () => fetchModels(filters)
 * })
 * 
 * <ModelListTableTemplate
 *   models={models}
 *   filters={filters}
 *   onFiltersChange={setFilters}
 *   onModelClick={(id) => navigate(`/models/${id}`)}
 * />
 * ```
 */
export function ModelListTableTemplate({
  models,
  onModelClick,
  isLoading = false,
  error,
  emptyMessage,
  emptyDescription,
  filterOptions,
  filters: controlledFilters,
  onFiltersChange,
}: ModelListTableTemplateProps) {
  // Use internal state if not controlled
  const internalFilters = useModelFilters(filterOptions)
  
  // Determine if controlled or uncontrolled
  const isControlled = controlledFilters !== undefined && onFiltersChange !== undefined
  
  const filters = isControlled ? controlledFilters : internalFilters.filters
  const setSearch = isControlled
    ? (search: string) => onFiltersChange?.({ ...controlledFilters!, search })
    : internalFilters.setSearch
  const setSort = isControlled
    ? (sort: string) => onFiltersChange?.({ ...controlledFilters!, sort })
    : internalFilters.setSort
  const toggleTag = isControlled
    ? (tagId: string) => {
        const newTags = controlledFilters!.tags.includes(tagId)
          ? controlledFilters!.tags.filter((t) => t !== tagId)
          : [...controlledFilters!.tags, tagId]
        onFiltersChange?.({ ...controlledFilters!, tags: newTags })
      }
    : internalFilters.toggleTag
  const clearFilters = isControlled
    ? () => onFiltersChange?.({ search: '', sort: filterOptions?.defaultSort || 'downloads', tags: [] })
    : internalFilters.clearFilters
  
  const sortOptions = internalFilters.sortOptions
  const filterChips = isControlled
    ? (filterOptions?.availableChips || []).map((chip) => ({
        id: chip.id,
        label: chip.label,
        active: controlledFilters!.tags.includes(chip.id),
      }))
    : internalFilters.filterChips

  return (
    <div className="space-y-6">
      {/* Filters */}
      <FilterBar
        search={filters.search}
        onSearchChange={setSearch}
        sort={filters.sort}
        onSortChange={setSort}
        sortOptions={sortOptions}
        onClearFilters={clearFilters}
        filterChips={filterChips}
        onFilterChipToggle={toggleTag}
      />

      {/* Table */}
      <ModelTable
        models={models}
        onModelClick={onModelClick}
        isLoading={isLoading}
        error={error}
        emptyMessage={emptyMessage}
        emptyDescription={emptyDescription}
      />
    </div>
  )
}
