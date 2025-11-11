// TEAM-XXX: HuggingFace-specific constants
// Source: WASM SDK types from marketplace_sdk.d.ts

// Re-export HuggingFace WASM types
export type {
  HuggingFaceFilters,
  HuggingFaceSort,
} from '../../wasm/marketplace_sdk'

/**
 * HuggingFace Sort Options (enumerated constants)
 * Source: HuggingFaceSort = "Downloads" | "Likes" | "Recent" | "Trending"
 *
 * NOTE: API only supports 'Downloads' and 'Likes' - 'Recent' causes errors
 */
export const HF_SORT = {
  DOWNLOADS: 'Downloads',
  LIKES: 'Likes',
} as const

export const HF_SORTS = Object.values(HF_SORT)

/**
 * HuggingFace Model Size Categories (enumerated constants)
 * UI-only filter, not in API. Based on parameter count (small <7B, medium 7-13B, large >13B)
 */
export const HF_SIZE = {
  ALL: 'All',
  SMALL: 'Small',
  MEDIUM: 'Medium',
  LARGE: 'Large',
} as const

export const HF_SIZES = Object.values(HF_SIZE)

/**
 * HuggingFace License Filters (enumerated constants)
 * UI-only filter, not in API
 */
export const HF_LICENSE = {
  ALL: 'All',
  APACHE: 'Apache',
  MIT: 'MIT',
  OTHER: 'Other',
} as const

export const HF_LICENSES = Object.values(HF_LICENSE)

/**
 * HuggingFace URL slugs (enumerated constants)
 */
export const HF_URL_SLUG = {
  SORT: {
    DOWNLOADS: 'downloads',
    LIKES: 'likes',
  },
  SIZE: {
    ALL: 'all',
    SMALL: 'small',
    MEDIUM: 'medium',
    LARGE: 'large',
  },
  LICENSE: {
    ALL: 'all',
    APACHE: 'apache',
    MIT: 'mit',
    OTHER: 'other',
  },
} as const

/**
 * HuggingFace URL slug arrays for filter generation
 */
export const HF_URL_SLUGS = {
  SORTS: Object.values(HF_URL_SLUG.SORT),
  SIZES: Object.values(HF_URL_SLUG.SIZE),
  LICENSES: Object.values(HF_URL_SLUG.LICENSE),
} as const

/**
 * HuggingFace default filter values
 */
export const HF_DEFAULTS = {
  SORT: HF_SORT.DOWNLOADS,
  SIZE: HF_SIZE.ALL,
  LICENSE: HF_LICENSE.ALL,
  LIMIT: 100,
} as const

/**
 * Model size thresholds (in billions of parameters)
 * 
 * Best practice: Use safetensors.parameters.total from HF API when available.
 * This provides accurate parameter counts from model metadata.
 * 
 * Fallback: Parse model name/tags for size indicators (7B, 13B, etc.)
 * 
 * Thresholds based on common LLM categories:
 * - Small: < 7B parameters (e.g., 1B, 3B, 6B models)
 * - Medium: 7B - 20B parameters (e.g., 7B, 8B, 13B models)
 * - Large: > 20B parameters (e.g., 30B, 34B, 70B+ models)
 */
export const MODEL_SIZE_THRESHOLDS = {
  SMALL_MAX: 7_000_000_000, // 7 billion
  MEDIUM_MAX: 20_000_000_000, // 20 billion
} as const

/**
 * Model size name patterns (fallback heuristic when safetensors data unavailable)
 * Matches common naming conventions: "model-7b", "llama-13B", "3b-instruct", etc.
 */
export const MODEL_SIZE_PATTERNS = {
  SMALL: ['1b', '2b', '3b', '6b', '1.5b', '2.5b'],
  MEDIUM: ['7b', '8b', '9b', '10b', '11b', '12b', '13b', '14b', '15b'],
  LARGE: ['20b', '30b', '33b', '34b', '40b', '65b', '70b', '120b', '180b'],
} as const

/**
 * License detection patterns (for HuggingFace filtering)
 * 
 * Best practice: Use model.cardData.license from HF API when available.
 * Fallback: Search in tags, model card, or siblings for license indicators.
 */
export const LICENSE_PATTERNS = {
  APACHE: 'apache',
  MIT: 'mit',
} as const
