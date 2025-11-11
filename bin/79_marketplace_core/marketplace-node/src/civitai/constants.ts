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
 * CivitAI Time Periods (enumerated constants)
 * Source: TimePeriod = "AllTime" | "Year" | "Month" | "Week" | "Day"
 */
export const CIVITAI_TIME_PERIOD = {
  ALL_TIME: 'AllTime',
  YEAR: 'Year',
  MONTH: 'Month',
  WEEK: 'Week',
  DAY: 'Day',
} as const

export const CIVITAI_TIME_PERIODS = Object.values(CIVITAI_TIME_PERIOD)

/**
 * CivitAI Model Types (enumerated constants)
 * Source: CivitaiModelType = "All" | "Checkpoint" | "LORA" | ...
 */
export const CIVITAI_MODEL_TYPE = {
  ALL: 'All',
  CHECKPOINT: 'Checkpoint',
  LORA: 'LORA',
} as const

export const CIVITAI_MODEL_TYPES = Object.values(CIVITAI_MODEL_TYPE)

/**
 * CivitAI Base Models (enumerated constants)
 * Source: BaseModel = "All" | "SDXL 1.0" | "SD 1.5" | "SD 2.1" | "Pony" | "Flux"
 */
export const CIVITAI_BASE_MODEL = {
  ALL: 'All',
  SDXL_1_0: 'SDXL 1.0',
  SD_1_5: 'SD 1.5',
  SD_2_1: 'SD 2.1',
} as const

export const CIVITAI_BASE_MODELS = Object.values(CIVITAI_BASE_MODEL)

/**
 * CivitAI Sort Options (enumerated constants)
 * Source: CivitaiSort = "Most Downloaded" | "Highest Rated" | "Newest"
 */
export const CIVITAI_SORT = {
  MOST_DOWNLOADED: 'Most Downloaded',
  HIGHEST_RATED: 'Highest Rated',
  NEWEST: 'Newest',
} as const

export const CIVITAI_SORTS = Object.values(CIVITAI_SORT)

/**
 * CivitAI NSFW Levels (enumerated constants)
 * Source: NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
 *
 * Numeric mapping (from civitai.ts):
 * - 'None': [1]           - PG only
 * - 'Soft': [1, 2]        - PG + PG-13
 * - 'Mature': [1, 2, 4]   - PG + PG-13 + R
 * - 'X': [1, 2, 4, 8]     - Up to X-rated
 * - 'XXX': [1, 2, 4, 8, 16] - ALL levels (default)
 */
export const CIVITAI_NSFW_LEVEL = {
  NONE: 'None',
  SOFT: 'Soft',
  MATURE: 'Mature',
  X: 'X',
  XXX: 'XXX',
} as const

export const CIVITAI_NSFW_LEVELS = Object.values(CIVITAI_NSFW_LEVEL)

/**
 * CivitAI URL slugs (enumerated constants)
 */
export const CIVITAI_URL_SLUG = {
  NSFW_LEVEL: {
    ALL: 'all',
    PG: 'pg',
    PG13: 'pg13',
    R: 'r',
    X: 'x',
  },
  TIME_PERIOD: {
    ALL: 'all',
    DAY: 'day',
    WEEK: 'week',
    MONTH: 'month',
    YEAR: 'year',
  },
  MODEL_TYPE: {
    ALL: 'all',
    CHECKPOINTS: 'checkpoints',
    LORAS: 'loras',
  },
  BASE_MODEL: {
    ALL: 'all',
    SDXL: 'sdxl',
    SD15: 'sd15',
    SD21: 'sd21',
  },
} as const

/**
 * CivitAI URL slug arrays for filter generation
 */
export const CIVITAI_URL_SLUGS = {
  NSFW_LEVELS: Object.values(CIVITAI_URL_SLUG.NSFW_LEVEL),
  TIME_PERIODS: Object.values(CIVITAI_URL_SLUG.TIME_PERIOD),
  MODEL_TYPES: Object.values(CIVITAI_URL_SLUG.MODEL_TYPE),
  BASE_MODELS: Object.values(CIVITAI_URL_SLUG.BASE_MODEL),
} as const

/**
 * CivitAI default filter values
 * TEAM-476: NSFW_LEVEL set to XXX (all levels including explicit) - NO CENSORSHIP
 */
export const CIVITAI_DEFAULTS = {
  TIME_PERIOD: CIVITAI_TIME_PERIOD.ALL_TIME,
  MODEL_TYPE: CIVITAI_MODEL_TYPE.ALL,
  BASE_MODEL: CIVITAI_BASE_MODEL.ALL,
  SORT: CIVITAI_SORT.MOST_DOWNLOADED,
  NSFW_LEVEL: CIVITAI_NSFW_LEVEL.XXX, // ALL NSFW levels [1,2,4,8,16] - NO FILTERING
  BLUR_MATURE: false,
  LIMIT: 100,
} as const
