// TEAM-467 RULE ZERO: Re-export WASM SDK types as SINGLE SOURCE OF TRUTH
// Source: /bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts
//
// Shared between:
// - Next.js marketplace app (/frontend/apps/marketplace)
// - Tauri Keeper app (/bin/00_rbee_keeper/ui)
// - Manifest generation scripts
//
// NO MAGIC STRINGS - Use API enum values directly!

// Re-export WASM SDK types (these are the API contracts)
export type {
  BaseModel,
  CivitaiFilters,
  CivitaiModelType,
  CivitaiSort,
  HuggingFaceFilters,
  HuggingFaceSort,
  NsfwFilter,
  NsfwLevel,
  TimePeriod,
} from '@rbee/marketplace-node'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HUGGINGFACE CONSTANTS (using API enum values)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * HuggingFace Sort Options (API enum values)
 * Source: HuggingFaceSort = "Downloads" | "Likes" | "Recent" | "Trending"
 *
 * NOTE: API only supports 'Downloads' and 'Likes' - 'Recent' causes errors
 */
export const HF_SORTS = ['Downloads', 'Likes'] as const

/**
 * HuggingFace Model Size Categories (UI-only filter, not in API)
 * Based on parameter count (small <7B, medium 7-13B, large >13B)
 */
export const HF_SIZES = ['All', 'Small', 'Medium', 'Large'] as const

/**
 * HuggingFace License Filters (UI-only filter, not in API)
 */
export const HF_LICENSES = ['All', 'Apache', 'MIT', 'Other'] as const

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CIVITAI CONSTANTS (using API enum values)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MAGIC STRING CONSTANTS - URL Slugs & Filter Values
// TEAM-XXX: NO MORE MAGIC STRINGS! Use these constants everywhere!
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Default filter values (API enum values)
 */
export const FILTER_DEFAULTS = {
  // CivitAI defaults
  CIVITAI_TIME_PERIOD: 'AllTime',
  CIVITAI_MODEL_TYPE: 'All',
  CIVITAI_BASE_MODEL: 'All',
  CIVITAI_SORT: 'Most Downloaded',
  CIVITAI_NSFW_LEVEL: 'XXX',
  CIVITAI_BLUR_MATURE: false,
  CIVITAI_LIMIT: 100,
  
  // HuggingFace defaults
  HF_SORT: 'Downloads',
  HF_SIZE: 'All',
  HF_LICENSE: 'All',
  HF_LIMIT: 100,
} as const

/**
 * URL slugs for filter paths (lowercase, URL-friendly)
 * Maps to API enum values via parser functions
 */
export const URL_SLUGS = {
  // CivitAI model types
  CHECKPOINTS: 'checkpoints',
  LORAS: 'loras',
  
  // CivitAI base models
  SDXL: 'sdxl',
  SD15: 'sd15',
  SD21: 'sd21',
  PONY: 'pony',
  FLUX: 'flux',
  
  // CivitAI time periods
  WEEK: 'week',
  MONTH: 'month',
  YEAR: 'year',
  DAY: 'day',
  ALL_TIME: 'all',
  
  // CivitAI NSFW levels (URL slugs)
  PG: 'pg',
  PG13: 'pg13',
  R: 'r',
  X: 'x',
  
  // HuggingFace sorts
  DOWNLOADS: 'downloads',
  LIKES: 'likes',
  RECENT: 'recent',
  TRENDING: 'trending',
  
  // HuggingFace sizes
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
  
  // HuggingFace licenses
  APACHE: 'apache',
  MIT: 'mit',
  OTHER: 'other',
  
  // Common
  ALL: 'all',
  FILTER_PREFIX: 'filter',
} as const

/**
 * URL slug to API enum value mappings
 */
export const SLUG_TO_API = {
  // CivitAI model types
  [URL_SLUGS.CHECKPOINTS]: 'Checkpoint',
  [URL_SLUGS.LORAS]: 'LORA',
  
  // CivitAI base models
  [URL_SLUGS.SDXL]: 'SDXL 1.0',
  [URL_SLUGS.SD15]: 'SD 1.5',
  [URL_SLUGS.SD21]: 'SD 2.1',
  
  // CivitAI time periods
  [URL_SLUGS.WEEK]: 'Week',
  [URL_SLUGS.MONTH]: 'Month',
  [URL_SLUGS.YEAR]: 'Year',
  [URL_SLUGS.DAY]: 'Day',
  
  // CivitAI NSFW levels
  [URL_SLUGS.PG]: 'None',
  [URL_SLUGS.PG13]: 'Soft',
  [URL_SLUGS.R]: 'Mature',
  [URL_SLUGS.X]: 'X',
  
  // HuggingFace (most are already API values)
  [URL_SLUGS.DOWNLOADS]: 'Downloads',
  [URL_SLUGS.LIKES]: 'Likes',
  [URL_SLUGS.SMALL]: 'Small',
  [URL_SLUGS.MEDIUM]: 'Medium',
  [URL_SLUGS.LARGE]: 'Large',
  [URL_SLUGS.APACHE]: 'Apache',
  [URL_SLUGS.MIT]: 'MIT',
  [URL_SLUGS.OTHER]: 'Other',
} as const

/**
 * Display labels for UI
 */
export const DISPLAY_LABELS = {
  // CivitAI sorts
  MOST_DOWNLOADED: 'Most Downloaded',
  HIGHEST_RATED: 'Highest Rated',
  NEWEST: 'Newest',
  
  // HuggingFace sorts
  MOST_LIKED: 'Most Liked',
  RECENTLY_UPDATED: 'Recently Updated',
  TRENDING: 'Trending',
  
  // Sizes
  SMALL_MODELS: 'Small Models',
  MEDIUM_MODELS: 'Medium Models',
  LARGE_MODELS: 'Large Models',
  
  // Licenses
  APACHE_2_0: 'Apache 2.0',
  MIT_LICENSE: 'MIT',
  OTHER_LICENSE: 'Other License',
  
  // Common
  ALL_MODELS: 'All Models',
} as const

/**
 * Model size heuristics (for HuggingFace filtering by name)
 */
export const MODEL_SIZE_PATTERNS = {
  SMALL: ['7b', '3b', '1b'],
  MEDIUM: ['13b', '8b'],
  LARGE: ['70b', '34b', '30b'],
} as const

/**
 * License detection patterns (for HuggingFace filtering)
 */
export const LICENSE_PATTERNS = {
  APACHE: 'apache',
  MIT: 'mit',
} as const

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// URL SLUG ARRAYS - For filter path generation
// These are lowercase URL-friendly versions, separate from API enum values
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
 * HuggingFace URL slug arrays for filter generation
 */
export const HF_URL_SLUGS = {
  SORTS: ['downloads', 'likes'] as const,
  SIZES: ['all', 'small', 'medium', 'large'] as const,
  LICENSES: ['all', 'apache', 'mit', 'other'] as const,
} as const
