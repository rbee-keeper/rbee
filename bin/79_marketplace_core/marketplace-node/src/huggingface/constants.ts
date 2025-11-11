// TEAM-XXX: HuggingFace-specific constants
// Source: WASM SDK types from marketplace_sdk.d.ts

// Re-export HuggingFace WASM types
export type {
  HuggingFaceFilters,
  HuggingFaceSort,
} from '../../wasm/marketplace_sdk'

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

/**
 * HuggingFace URL slug arrays for filter generation
 */
export const HF_URL_SLUGS = {
  SORTS: ['downloads', 'likes'] as const,
  SIZES: ['all', 'small', 'medium', 'large'] as const,
  LICENSES: ['all', 'apache', 'mit', 'other'] as const,
} as const

/**
 * HuggingFace default filter values
 */
export const HF_DEFAULTS = {
  SORT: 'Downloads',
  SIZE: 'All',
  LICENSE: 'All',
  LIMIT: 100,
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
