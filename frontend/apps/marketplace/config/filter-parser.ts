// TEAM-467: SINGLE SOURCE OF TRUTH for filter parsing
// Converts filter paths (e.g., "filter/week/loras/sdxl") to API parameters
// Used by manifest generation and potentially runtime filtering

import type { CivitaiFilters, NsfwLevel } from '@rbee/marketplace-node'

/**
 * NSFW level mapping: URL slug → CivitAI API enum value
 * 
 * TEAM-467: SINGLE SOURCE - Don't duplicate this mapping!
 * 
 * Source of truth: CivitAI API + WASM contract
 * - WASM contract: /bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts
 *   export type NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
 * 
 * - CivitAI API numeric values (from civitai.ts):
 *   'None': [1]           - PG only
 *   'Soft': [1, 2]        - PG + PG-13
 *   'Mature': [1, 2, 4]   - PG + PG-13 + R
 *   'X': [1, 2, 4, 8]     - PG + PG-13 + R + X
 *   'XXX': [1, 2, 4, 8, 16] - ALL levels (default)
 */
const NSFW_LEVEL_MAP: Record<string, NsfwLevel> = {
  'pg': 'None',      // CivitAI API idiomatic: PG only
  'pg13': 'Soft',    // CivitAI API idiomatic: PG + PG-13
  'r': 'Mature',     // CivitAI API idiomatic: up to R-rated
  'x': 'X',          // CivitAI API idiomatic: up to X-rated
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
      max_level: 'XXX',  // Default: all NSFW levels
      blur_mature: false,
    },
    limit: 100,
  }
  
  // Parse each part
  for (const part of filterParts) {
    // Model types
    if (part === 'checkpoints') {
      filters.model_type = 'Checkpoint'
    } else if (part === 'loras') {
      filters.model_type = 'LORA'
    }
    // Base models
    else if (part === 'sdxl') {
      filters.base_model = 'SDXL 1.0'
    } else if (part === 'sd15') {
      filters.base_model = 'SD 1.5'
    }
    // Time periods
    else if (part === 'week') {
      filters.time_period = 'Week'
    } else if (part === 'month') {
      filters.time_period = 'Month'
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
    'checkpoints', 'loras',
    // Base models
    'sdxl', 'sd15',
    // Time periods
    'week', 'month',
    // NSFW levels
    'pg', 'pg13', 'r', 'x',
  ])
  
  return filterParts.every(part => validParts.has(part))
}
