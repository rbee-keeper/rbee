// TEAM-XXX RULE ZERO: Import constants from marketplace-node (source of truth)

import {
  type BaseModel,
  CIVITAI_BASE_MODELS,
  CIVITAI_MODEL_TYPES,
  CIVITAI_NSFW_LEVELS,
  CIVITAI_SORTS,
  CIVITAI_TIME_PERIODS,
  CIVITAI_URL_SLUGS,
  type CivitaiModelType,
  type CivitaiSort,
  type NsfwLevel,
  type TimePeriod,
} from '@rbee/marketplace-node'
import type { FilterGroup, FilterConfig as GenericFilterConfig } from '@/lib/filters/types'

// Local interface extending the core types
export interface CivitaiFilters {
  timePeriod: TimePeriod
  modelType: CivitaiModelType
  baseModel: BaseModel
  sort?: CivitaiSort
  nsfwLevel?: NsfwLevel
}

// Filter group definitions (left side - actual filters)
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export const CIVITAI_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'timePeriod',
    label: 'Time Period',
    options: [
      { label: 'All Time', value: CIVITAI_TIME_PERIODS[0] },
      { label: 'Year', value: CIVITAI_TIME_PERIODS[1] },
      { label: 'Month', value: CIVITAI_TIME_PERIODS[2] },
      { label: 'Week', value: CIVITAI_TIME_PERIODS[3] },
      { label: 'Day', value: CIVITAI_TIME_PERIODS[4] },
    ],
  },
  {
    id: 'modelType',
    label: 'Model Type',
    options: [
      { label: 'All Types', value: CIVITAI_MODEL_TYPES[0] },
      { label: 'Checkpoint', value: CIVITAI_MODEL_TYPES[1] },
      { label: 'LORA', value: CIVITAI_MODEL_TYPES[2] },
    ],
  },
  {
    id: 'baseModel',
    label: 'Base Model',
    options: [
      { label: 'All Models', value: CIVITAI_BASE_MODELS[0] },
      { label: 'SDXL 1.0', value: CIVITAI_BASE_MODELS[1] },
      { label: 'SD 1.5', value: CIVITAI_BASE_MODELS[2] },
      { label: 'SD 2.1', value: CIVITAI_BASE_MODELS[3] },
    ],
  },
  {
    id: 'nsfwLevel',
    label: 'Content Rating',
    options: [
      { label: 'PG (Safe for work)', value: CIVITAI_NSFW_LEVELS[0] },
      { label: 'PG-13 (Suggestive)', value: CIVITAI_NSFW_LEVELS[1] },
      { label: 'R (Mature)', value: CIVITAI_NSFW_LEVELS[2] },
      { label: 'X (Explicit)', value: CIVITAI_NSFW_LEVELS[3] },
      { label: 'XXX (Pornographic)', value: CIVITAI_NSFW_LEVELS[4] },
    ],
  },
]

// Sort group definition (right side - sorting only)
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export const CIVITAI_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: CIVITAI_SORTS[0] },
    { label: 'Most Likes', value: CIVITAI_SORTS[1] },
    { label: 'Newest', value: CIVITAI_SORTS[2] },
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
// TEAM-463: Added NSFW filter levels
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export const PREGENERATED_FILTERS: GenericFilterConfig<CivitaiFilters>[] = [
  // Default view (PG - Safe for work)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: '',
  },

  // NSFW filter levels (most important - users want to filter by content rating)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[1]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[1],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[2]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[2],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[3]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[3],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[4]}`,
  },

  // Popular time periods (PG only)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[2],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.TIME_PERIODS[2]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[3],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.TIME_PERIODS[3]}`,
  },

  // Model type filters (PG only)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[1],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.MODEL_TYPES[1]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[2],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.MODEL_TYPES[2]}`,
  },

  // Base model filters (PG only)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[1],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.BASE_MODELS[1]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[0],
      baseModel: CIVITAI_BASE_MODELS[2],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.BASE_MODELS[2]}`,
  },

  // Popular combinations (PG only)
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[2],
      modelType: CIVITAI_MODEL_TYPES[1],
      baseModel: CIVITAI_BASE_MODELS[1],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.TIME_PERIODS[2]}/${CIVITAI_URL_SLUGS.MODEL_TYPES[1]}/${CIVITAI_URL_SLUGS.BASE_MODELS[1]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[2],
      modelType: CIVITAI_MODEL_TYPES[2],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.TIME_PERIODS[2]}/${CIVITAI_URL_SLUGS.MODEL_TYPES[2]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[3],
      modelType: CIVITAI_MODEL_TYPES[1],
      baseModel: CIVITAI_BASE_MODELS[1],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[0],
    },
    path: `filter/${CIVITAI_URL_SLUGS.TIME_PERIODS[3]}/${CIVITAI_URL_SLUGS.MODEL_TYPES[1]}/${CIVITAI_URL_SLUGS.BASE_MODELS[1]}`,
  },

  // NSFW + Model Type combinations
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[1],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[2],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[3]}/${CIVITAI_URL_SLUGS.MODEL_TYPES[1]}`,
  },
  {
    filters: {
      timePeriod: CIVITAI_TIME_PERIODS[0],
      modelType: CIVITAI_MODEL_TYPES[2],
      baseModel: CIVITAI_BASE_MODELS[0],
      sort: CIVITAI_SORTS[0],
      nsfwLevel: CIVITAI_NSFW_LEVELS[2],
    },
    path: `filter/${CIVITAI_URL_SLUGS.NSFW_LEVELS[3]}/${CIVITAI_URL_SLUGS.MODEL_TYPES[2]}`,
  },
]

// Helper to build API parameters from filter config
// TEAM-429: Convert frontend filter values to API parameter format
// TEAM-463: Added NSFW level conversion
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
// Converts frontend constants to API parameters
export function buildFilterParams(filters: CivitaiFilters): CivitaiFilters {
  return {
    timePeriod: filters.timePeriod,
    modelType: filters.modelType,
    baseModel: filters.baseModel,
    sort: filters.sort || CIVITAI_SORTS[0],
    nsfwLevel: filters.nsfwLevel || CIVITAI_NSFW_LEVELS[0],
  }
}

// Helper to get filter config from path
export function getFilterFromPath(path: string): CivitaiFilters {
  const found = PREGENERATED_FILTERS.find((f) => f.path === path)
  return found?.filters || PREGENERATED_FILTERS[0].filters
}

// Helper to build URL from filter config
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export function buildFilterUrl(filters: Partial<CivitaiFilters>): string {
  const found = PREGENERATED_FILTERS.find(
    (f) =>
      f.filters.timePeriod === (filters.timePeriod || CIVITAI_TIME_PERIODS[0]) &&
      f.filters.modelType === (filters.modelType || CIVITAI_MODEL_TYPES[0]) &&
      f.filters.baseModel === (filters.baseModel || CIVITAI_BASE_MODELS[0]),
  )

  if (found) {
    return found.path ? `/models/civitai/${found.path}` : '/models/civitai'
  }

  // Fallback to default
  return '/models/civitai'
}
