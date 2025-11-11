// TEAM-XXX: CivitAI-specific constants
// Source: WASM SDK types from marketplace_sdk.d.ts

// Re-export CivitAI WASM types
export type {
  BaseModel,
  CivitaiFilters,
  CivitaiModelType,
  CivitaiSort,
  NsfwFilter,
  NsfwLevel,
  TimePeriod,
} from '../../wasm/marketplace_sdk'

/**
 * CivitAI Time Periods (API enum values)
 * Source: TimePeriod = "AllTime" | "Year" | "Month" | "Week" | "Day"
 */
export const CIVITAI_TIME_PERIODS = ['AllTime', 'Year', 'Month', 'Week', 'Day'] as const

/**
 * CivitAI Model Types (API enum values)
 * Source: CivitaiModelType = "All" | "Checkpoint" | "LORA" | ...
 */
export const CIVITAI_MODEL_TYPES = ['All', 'Checkpoint', 'LORA'] as const

/**
 * CivitAI Base Models (API enum values)
 * Source: BaseModel = "All" | "SDXL 1.0" | "SD 1.5" | "SD 2.1" | "Pony" | "Flux"
 */
export const CIVITAI_BASE_MODELS = ['All', 'SDXL 1.0', 'SD 1.5', 'SD 2.1'] as const

/**
 * CivitAI Sort Options (API enum values)
 * Source: CivitaiSort = "Most Downloaded" | "Highest Rated" | "Newest"
 */
export const CIVITAI_SORTS = ['Most Downloaded', 'Highest Rated', 'Newest'] as const

/**
 * CivitAI NSFW Levels (API enum values)
 * Source: NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
 *
 * Numeric mapping (from civitai.ts):
 * - 'None': [1]           - PG only
 * - 'Soft': [1, 2]        - PG + PG-13
 * - 'Mature': [1, 2, 4]   - PG + PG-13 + R
 * - 'X': [1, 2, 4, 8]     - Up to X-rated
 * - 'XXX': [1, 2, 4, 8, 16] - ALL levels (default)
 */
export const CIVITAI_NSFW_LEVELS = ['None', 'Soft', 'Mature', 'X', 'XXX'] as const

/**
 * CivitAI URL slug arrays for filter generation
 * These map to API values via SLUG_TO_API constant
 */
export const CIVITAI_URL_SLUGS = {
  NSFW_LEVELS: ['all', 'pg', 'pg13', 'r', 'x'] as const,
  TIME_PERIODS: ['all', 'day', 'week', 'month', 'year'] as const,
  MODEL_TYPES: ['all', 'checkpoints', 'loras'] as const,
  BASE_MODELS: ['all', 'sdxl', 'sd15', 'sd21'] as const,
} as const

/**
 * CivitAI default filter values
 */
export const CIVITAI_DEFAULTS = {
  TIME_PERIOD: 'AllTime',
  MODEL_TYPE: 'All',
  BASE_MODEL: 'All',
  SORT: 'Most Downloaded',
  NSFW_LEVEL: 'XXX',
  BLUR_MATURE: false,
  LIMIT: 100,
} as const
