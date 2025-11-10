// TEAM-422: CivitAI filter definitions for SSG pre-generation
// TEAM-461: Refactored to use generic FilterGroup pattern
// TEAM-429: Import filter types from @rbee/marketplace-node
import type { FilterGroup, FilterConfig as GenericFilterConfig } from '@/lib/filters/types'
import type {
  CivitaiFilters as NodeCivitaiFilters,
  TimePeriod,
  CivitaiModelType,
  BaseModel,
  CivitaiSort,
  NsfwFilter,
} from '@rbee/marketplace-node'

// TEAM-429: Re-export types from marketplace-node
export type { TimePeriod, CivitaiModelType, BaseModel, CivitaiSort, NsfwFilter }

// TEAM-429: Frontend-specific filter interface (extends Node SDK types)
// Note: Frontend uses simpler sort values ('downloads' vs 'Most Downloaded')
export interface CivitaiFilters {
  timePeriod: TimePeriod
  modelType: CivitaiModelType
  baseModel: BaseModel
  sort: 'downloads' | 'likes' | 'newest'  // Frontend-specific sort values
  nsfw?: NsfwFilter  // Optional for frontend
}

// Filter group definitions (left side - actual filters)
export const CIVITAI_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'timePeriod',
    label: 'Time Period',
    options: [
      { label: 'All Time', value: 'AllTime' },
      { label: 'Month', value: 'Month' },
      { label: 'Week', value: 'Week' },
      { label: 'Day', value: 'Day' },
    ],
  },
  {
    id: 'modelType',
    label: 'Model Type',
    options: [
      { label: 'All Types', value: 'All' },
      { label: 'Checkpoint', value: 'Checkpoint' },
      { label: 'LORA', value: 'LORA' },
    ],
  },
  {
    id: 'baseModel',
    label: 'Base Model',
    options: [
      { label: 'All Models', value: 'All' },
      { label: 'SDXL 1.0', value: 'SDXL 1.0' },
      { label: 'SD 1.5', value: 'SD 1.5' },
      { label: 'SD 2.1', value: 'SD 2.1' },
    ],
  },
]

// Sort group definition (right side - sorting only)
export const CIVITAI_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    { label: 'Newest', value: 'newest' },
  ],
}

// Legacy export for backwards compatibility (deprecated)
export const CIVITAI_FILTERS = {
  timePeriod: CIVITAI_FILTER_GROUPS[0].options,
  modelTypes: CIVITAI_FILTER_GROUPS[1].options,
  baseModel: CIVITAI_FILTER_GROUPS[2].options,
} as const

// Pre-generate these popular combinations
// TEAM-460: Added 'filter/' prefix to avoid route conflicts with [slug]
export const PREGENERATED_FILTERS: GenericFilterConfig<CivitaiFilters>[] = [
  // Default view
  { filters: { timePeriod: 'AllTime', modelType: 'All', baseModel: 'All', sort: 'downloads' }, path: '' },

  // Popular time periods
  { filters: { timePeriod: 'Month', modelType: 'All', baseModel: 'All', sort: 'downloads' }, path: 'filter/month' },
  { filters: { timePeriod: 'Week', modelType: 'All', baseModel: 'All', sort: 'downloads' }, path: 'filter/week' },

  // Model type filters
  {
    filters: { timePeriod: 'AllTime', modelType: 'Checkpoint', baseModel: 'All', sort: 'downloads' },
    path: 'filter/checkpoints',
  },
  { filters: { timePeriod: 'AllTime', modelType: 'LORA', baseModel: 'All', sort: 'downloads' }, path: 'filter/loras' },

  // Base model filters
  {
    filters: { timePeriod: 'AllTime', modelType: 'All', baseModel: 'SDXL 1.0', sort: 'downloads' },
    path: 'filter/sdxl',
  },
  { filters: { timePeriod: 'AllTime', modelType: 'All', baseModel: 'SD 1.5', sort: 'downloads' }, path: 'filter/sd15' },

  // Popular combinations
  {
    filters: { timePeriod: 'Month', modelType: 'Checkpoint', baseModel: 'SDXL 1.0', sort: 'downloads' },
    path: 'filter/month/checkpoints/sdxl',
  },
  {
    filters: { timePeriod: 'Month', modelType: 'LORA', baseModel: 'SDXL 1.0', sort: 'downloads' },
    path: 'filter/month/loras/sdxl',
  },
  {
    filters: { timePeriod: 'Week', modelType: 'Checkpoint', baseModel: 'SDXL 1.0', sort: 'downloads' },
    path: 'filter/week/checkpoints/sdxl',
  },
]

// TEAM-429: Convert frontend sort values to API sort values
function convertSortToApi(sort: 'downloads' | 'likes' | 'newest'): CivitaiSort {
  switch (sort) {
    case 'downloads':
      return 'Most Downloaded'
    case 'likes':
      return 'Highest Rated'
    case 'newest':
      return 'Newest'
  }
}

// Helper to build API parameters from filter config
// TEAM-429: Now returns NodeCivitaiFilters for type-safe API calls
// Converts frontend camelCase to Node SDK snake_case
export function buildFilterParams(filters: CivitaiFilters): NodeCivitaiFilters {
  return {
    time_period: filters.timePeriod,
    model_type: filters.modelType,
    base_model: filters.baseModel,
    sort: convertSortToApi(filters.sort),
    nsfw: filters.nsfw || {
      max_level: 'None',
      blur_mature: true,
    },
    page: null,
    limit: 100,
  }
}

// Helper to get filter config from path
export function getFilterFromPath(path: string): CivitaiFilters {
  const found = PREGENERATED_FILTERS.find((f) => f.path === path)
  return found?.filters || PREGENERATED_FILTERS[0].filters
}

// Helper to build URL from filter config
export function buildFilterUrl(filters: Partial<CivitaiFilters>): string {
  const found = PREGENERATED_FILTERS.find(
    (f) =>
      f.filters.timePeriod === (filters.timePeriod || 'AllTime') &&
      f.filters.modelType === (filters.modelType || 'All') &&
      f.filters.baseModel === (filters.baseModel || 'All'),
  )

  if (found) {
    return found.path ? `/models/civitai/${found.path}` : '/models/civitai'
  }

  // Fallback to default
  return '/models/civitai'
}
