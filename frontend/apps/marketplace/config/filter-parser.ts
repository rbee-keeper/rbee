// TEAM-XXX: SINGLE SOURCE OF TRUTH for filter parsing
// Converts filter paths (e.g., "filter/week/loras/sdxl") to API parameters
// Used by manifest generation and potentially runtime filtering
// TEAM-XXX RULE ZERO: Using constants from marketplace-node (source of truth)

import {
  CIVITAI_BASE_MODELS,
  CIVITAI_DEFAULTS,
  CIVITAI_MODEL_TYPES,
  CIVITAI_NSFW_LEVELS,
  CIVITAI_TIME_PERIODS,
  CIVITAI_URL_SLUGS,
  type NsfwLevel,
} from '@rbee/marketplace-node'
import type { CivitaiFilters } from '@/app/models/civitai/filters'

/**
 * NSFW level mapping: URL slug → CivitAI API enum value
 * TEAM-XXX RULE ZERO: Using constants from marketplace-node
 */
const NSFW_LEVEL_MAP: Record<string, NsfwLevel> = {
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[1]]: CIVITAI_NSFW_LEVELS[0],
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[2]]: CIVITAI_NSFW_LEVELS[1],
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[3]]: CIVITAI_NSFW_LEVELS[2],
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[4]]: CIVITAI_NSFW_LEVELS[3],
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

  // Start with defaults from marketplace-node
  const filters: Partial<CivitaiFilters> = {
    timePeriod: CIVITAI_DEFAULTS.TIME_PERIOD,
    modelType: CIVITAI_DEFAULTS.MODEL_TYPE,
    baseModel: CIVITAI_DEFAULTS.BASE_MODEL,
    sort: CIVITAI_DEFAULTS.SORT,
    nsfwLevel: CIVITAI_DEFAULTS.NSFW_LEVEL,
  }

  // Parse each part using constants from marketplace-node
  for (const part of filterParts) {
    // Model types - use URL slug constants
    if (part === CIVITAI_URL_SLUGS.MODEL_TYPES[1]) {
      filters.modelType = CIVITAI_MODEL_TYPES[1]
    } else if (part === CIVITAI_URL_SLUGS.MODEL_TYPES[2]) {
      filters.modelType = CIVITAI_MODEL_TYPES[2]
    }
    // Base models - use URL slug constants
    else if (part === CIVITAI_URL_SLUGS.BASE_MODELS[1]) {
      filters.baseModel = CIVITAI_BASE_MODELS[1]
    } else if (part === CIVITAI_URL_SLUGS.BASE_MODELS[2]) {
      filters.baseModel = CIVITAI_BASE_MODELS[2]
    } else if (part === CIVITAI_URL_SLUGS.BASE_MODELS[3]) {
      filters.baseModel = CIVITAI_BASE_MODELS[3]
    }
    // Time periods - use URL slug constants
    else if (part === CIVITAI_URL_SLUGS.TIME_PERIODS[0]) {
      filters.timePeriod = CIVITAI_TIME_PERIODS[0]
    } else if (part === CIVITAI_URL_SLUGS.TIME_PERIODS[1]) {
      filters.timePeriod = CIVITAI_TIME_PERIODS[1]
    } else if (part === CIVITAI_URL_SLUGS.TIME_PERIODS[2]) {
      filters.timePeriod = CIVITAI_TIME_PERIODS[2]
    } else if (part === CIVITAI_URL_SLUGS.TIME_PERIODS[3]) {
      filters.timePeriod = CIVITAI_TIME_PERIODS[3]
    } else if (part === CIVITAI_URL_SLUGS.TIME_PERIODS[4]) {
      filters.timePeriod = CIVITAI_TIME_PERIODS[4]
    }
    // NSFW levels - use mapping from marketplace-node constants
    else if (part in NSFW_LEVEL_MAP) {
      filters.nsfwLevel = NSFW_LEVEL_MAP[part]
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

  // TEAM-XXX RULE ZERO: Use constants from marketplace-node instead of hardcoded arrays
  const validParts: Set<string> = new Set([
    // Model types from URL slug constants
    ...CIVITAI_URL_SLUGS.MODEL_TYPES,
    // Base models from URL slug constants
    ...CIVITAI_URL_SLUGS.BASE_MODELS,
    // Time periods from URL slug constants
    ...CIVITAI_URL_SLUGS.TIME_PERIODS,
    // NSFW levels from URL slug constants
    ...CIVITAI_URL_SLUGS.NSFW_LEVELS,
  ])

  return filterParts.every((part) => validParts.has(part))
}
