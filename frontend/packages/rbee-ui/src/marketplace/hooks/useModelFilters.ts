// TEAM-405: Model filters control hook
//! Control layer for model filtering
//! Manages filter state and logic, independent of data source

import { useMemo, useState } from 'react'

// TEAM-472: FilterChip type definition (removed from FilterBar export)
interface FilterChip {
  id: string
  label: string
}

export interface ModelFilters {
  search: string
  sort: string
  tags: string[]
}

export interface UseModelFiltersOptions {
  /** Initial sort value */
  defaultSort?: string

  /** Available sort options */
  sortOptions?: Array<{ value: string; label: string }>

  /** Available filter chips */
  availableChips?: Array<{ id: string; label: string }>
}

export interface UseModelFiltersReturn {
  /** Current filter state */
  filters: ModelFilters

  /** Update search query */
  setSearch: (search: string) => void

  /** Update sort order */
  setSort: (sort: string) => void

  /** Toggle a filter chip */
  toggleTag: (tagId: string) => void

  /** Clear all filters */
  clearFilters: () => void

  /** Sort options for FilterBar */
  sortOptions: Array<{ value: string; label: string }>

  /** Filter chips for FilterBar */
  filterChips: FilterChip[]
}

const DEFAULT_SORT_OPTIONS = [
  { value: 'downloads', label: 'Most Downloads' },
  { value: 'likes', label: 'Most Likes' },
  { value: 'recent', label: 'Recently Added' },
  { value: 'trending', label: 'Trending' },
]

const DEFAULT_CHIPS = [
  { id: 'transformers', label: 'Transformers' },
  { id: 'safetensors', label: 'SafeTensors' },
  { id: 'gguf', label: 'GGUF' },
  { id: 'pytorch', label: 'PyTorch' },
]

/**
 * Hook for managing model filter state
 *
 * @example
 * ```tsx
 * const {
 *   filters,
 *   setSearch,
 *   setSort,
 *   toggleTag,
 *   clearFilters,
 *   sortOptions,
 *   filterChips
 * } = useModelFilters()
 *
 * // Use filters in your data fetching
 * const { data } = useQuery({
 *   queryKey: ['models', filters],
 *   queryFn: () => fetchModels(filters)
 * })
 *
 * // Pass to FilterBar
 * <FilterBar
 *   search={filters.search}
 *   onSearchChange={setSearch}
 *   sort={filters.sort}
 *   onSortChange={setSort}
 *   sortOptions={sortOptions}
 *   filterChips={filterChips}
 *   onFilterChipToggle={toggleTag}
 *   onClearFilters={clearFilters}
 * />
 * ```
 */
export function useModelFilters({
  defaultSort = 'downloads',
  sortOptions = DEFAULT_SORT_OPTIONS,
  availableChips = DEFAULT_CHIPS,
}: UseModelFiltersOptions = {}): UseModelFiltersReturn {
  const [filters, setFilters] = useState<ModelFilters>({
    search: '',
    sort: defaultSort,
    tags: [],
  })

  const setSearch = (search: string) => {
    setFilters((prev) => ({ ...prev, search }))
  }

  const setSort = (sort: string) => {
    setFilters((prev) => ({ ...prev, sort }))
  }

  const toggleTag = (tagId: string) => {
    setFilters((prev) => ({
      ...prev,
      tags: prev.tags.includes(tagId) ? prev.tags.filter((t) => t !== tagId) : [...prev.tags, tagId],
    }))
  }

  const clearFilters = () => {
    setFilters({
      search: '',
      sort: defaultSort,
      tags: [],
    })
  }

  // Generate filter chips with active state
  const filterChips: FilterChip[] = useMemo(
    () =>
      availableChips.map((chip) => ({
        id: chip.id,
        label: chip.label,
        active: filters.tags.includes(chip.id),
      })),
    [availableChips, filters.tags],
  )

  return {
    filters,
    setSearch,
    setSort,
    toggleTag,
    clearFilters,
    sortOptions,
    filterChips,
  }
}
