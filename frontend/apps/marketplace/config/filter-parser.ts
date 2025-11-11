// TEAM-XXX: SINGLE SOURCE OF TRUTH for filter parsing
// Converts filter paths (e.g., "filter/week/loras/sdxl") to API parameters
// Used by manifest generation and potentially runtime filtering

import type { CivitaiFilters, NsfwLevel } from '@rbee/marketplace-node'

/**
 * NSFW level mapping: URL slug → CivitAI API enum value
 */
const NSFW_LEVEL_MAP: Record<string, NsfwLevel> = {
  'pg': 'None' as NsfwLevel,
  'pg13': 'Soft' as NsfwLevel,
  'r': 'Mature' as NsfwLevel,
  'x': 'X' as NsfwLevel,
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
  
  // Start with defaults
  const filters: Partial<CivitaiFilters> = {
    time_period: 'AllTime',
    model_type: 'All',
    base_model: 'All',
    sort: 'Most Downloaded',
    nsfw: {
      max_level: 'XXX',
      blur_mature: false,
    },
    limit: 100,
  }
  
  // Parse each part using hardcoded mappings
  for (const part of filterParts) {
    // Model types
    if (part === 'checkpoints') {
      filters.model_type = 'Checkpoint' as any
    } else if (part === 'loras') {
      filters.model_type = 'LORA' as any
    }
    // Base models
    else if (part === 'sdxl') {
      filters.base_model = 'SDXL 1.0' as any
    } else if (part === 'sd15') {
      filters.base_model = 'SD 1.5' as any
    }
    // Time periods
    else if (part === 'week') {
      filters.time_period = 'Week' as any
    } else if (part === 'month') {
      filters.time_period = 'Month' as any
    }
    // NSFW levels
    else if (part in NSFW_LEVEL_MAP) {
      filters.nsfw = {
        max_level: NSFW_LEVEL_MAP[part],
        blur_mature: false,
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
    'checkpoints',
    'loras',
    // Base models
    'sdxl',
    'sd15',
    // Time periods
    'week',
    'month',
    'year',
    'day',
    // NSFW levels
    'pg',
    'pg13',
    'r',
    'x',
  ])
  
  return filterParts.every(part => validParts.has(part))
}
