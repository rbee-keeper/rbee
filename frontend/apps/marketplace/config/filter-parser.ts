// TEAM-XXX: SINGLE SOURCE OF TRUTH for filter parsing
// Converts filter paths (e.g., "filter/week/loras/sdxl") to API parameters
// Used by manifest generation and potentially runtime filtering

import type { CivitaiFilters, NsfwLevel } from '@rbee/marketplace-node'
import { FILTER_DEFAULTS, SLUG_TO_API, URL_SLUGS } from '@rbee/marketplace-node'

/**
 * NSFW level mapping: URL slug → CivitAI API enum value
 * TEAM-XXX: Use SLUG_TO_API constant from marketplace-node
 */
const NSFW_LEVEL_MAP: Record<string, NsfwLevel> = {
  [URL_SLUGS.PG]: SLUG_TO_API[URL_SLUGS.PG] as NsfwLevel,
  [URL_SLUGS.PG13]: SLUG_TO_API[URL_SLUGS.PG13] as NsfwLevel,
  [URL_SLUGS.R]: SLUG_TO_API[URL_SLUGS.R] as NsfwLevel,
  [URL_SLUGS.X]: SLUG_TO_API[URL_SLUGS.X] as NsfwLevel,
  // No 'all' mapping - defaults to 'XXX' (all levels including explicit)
}

/**
 * Parse CivitAI filter path to API parameters
 * 
 * @param filterPath - Filter path like "filter/week/loras/sdxl" or "filter/pg/checkpoints"
 * @returns CivitaiFilters object for the SDK
 * 
 * @example
 * ```typescript
 * const filters = parseCivitAIFilter('filter/week/loras/sdxl')
 * // Returns: { time_period: 'Week', model_type: 'LORA', base_model: 'SDXL 1.0', ... }
 * ```
 */
export function parseCivitAIFilter(filterPath: string): Partial<CivitaiFilters> {
  const filterParts = filterPath.replace('filter/', '').split('/')
  
  // Start with defaults from FILTER_DEFAULTS
  const filters: Partial<CivitaiFilters> = {
    time_period: FILTER_DEFAULTS.CIVITAI_TIME_PERIOD,
    model_type: FILTER_DEFAULTS.CIVITAI_MODEL_TYPE,
    base_model: FILTER_DEFAULTS.CIVITAI_BASE_MODEL,
    sort: FILTER_DEFAULTS.CIVITAI_SORT,
    nsfw: {
      max_level: FILTER_DEFAULTS.CIVITAI_NSFW_LEVEL,
      blur_mature: FILTER_DEFAULTS.CIVITAI_BLUR_MATURE,
    },
    limit: FILTER_DEFAULTS.CIVITAI_LIMIT,
  }
  
  // Parse each part using SLUG_TO_API mappings
  for (const part of filterParts) {
    // Model types
    if (part === URL_SLUGS.CHECKPOINTS) {
      filters.model_type = SLUG_TO_API[URL_SLUGS.CHECKPOINTS] as any
    } else if (part === URL_SLUGS.LORAS) {
      filters.model_type = SLUG_TO_API[URL_SLUGS.LORAS] as any
    }
    // Base models
    else if (part === URL_SLUGS.SDXL) {
      filters.base_model = SLUG_TO_API[URL_SLUGS.SDXL] as any
    } else if (part === URL_SLUGS.SD15) {
      filters.base_model = SLUG_TO_API[URL_SLUGS.SD15] as any
    }
    // Time periods
    else if (part === URL_SLUGS.WEEK) {
      filters.time_period = SLUG_TO_API[URL_SLUGS.WEEK] as any
    } else if (part === URL_SLUGS.MONTH) {
      filters.time_period = SLUG_TO_API[URL_SLUGS.MONTH] as any
    }
    // NSFW levels
    else if (part in NSFW_LEVEL_MAP) {
      filters.nsfw = {
        max_level: NSFW_LEVEL_MAP[part],
        blur_mature: FILTER_DEFAULTS.CIVITAI_BLUR_MATURE,
      }
    }
  }
  
  return filters
}

/**
 * Validate that a filter path is valid
 * 
 * @param filterPath - Filter path to validate
 * @returns true if valid, false otherwise
 */
export function isValidCivitAIFilter(filterPath: string): boolean {
  const filterParts = filterPath.replace('filter/', '').split('/')
  
  const validParts = new Set([
    // Model types
    URL_SLUGS.CHECKPOINTS,
    URL_SLUGS.LORAS,
    // Base models
    URL_SLUGS.SDXL,
    URL_SLUGS.SD15,
    // Time periods
    URL_SLUGS.WEEK,
    URL_SLUGS.MONTH,
    // NSFW levels
    URL_SLUGS.PG,
    URL_SLUGS.PG13,
    URL_SLUGS.R,
    URL_SLUGS.X,
  ])
  
  return filterParts.every(part => validParts.has(part))
}
