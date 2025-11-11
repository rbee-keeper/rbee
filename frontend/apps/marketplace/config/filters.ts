// TEAM-464: Shared filter configuration
// Used by both manifest generation and filter pages
// Single source of truth for all filter combinations
// TEAM-XXX RULE ZERO: Import constants from marketplace-node (source of truth)
import { CIVITAI_URL_SLUGS, HF_URL_SLUGS } from '@rbee/marketplace-node'

/**
 * CivitAI filter paths
 * TEAM-467: PROGRAMMATICALLY generate ALL combinations
 * Uses SHARED URL slug constants from marketplace-node (source of truth)
 */
function generateAllCivitAIFilterCombinations(): string[] {
  const nsfw = CIVITAI_URL_SLUGS.NSFW_LEVELS
  const timePeriods = CIVITAI_URL_SLUGS.TIME_PERIODS
  const modelTypes = CIVITAI_URL_SLUGS.MODEL_TYPES
  const baseModels = CIVITAI_URL_SLUGS.BASE_MODELS

  const filters = new Set<string>()

  // Generate ALL combinations: nsfw × time × type × base
  for (const nsfwLevel of nsfw) {
    for (const period of timePeriods) {
      for (const type of modelTypes) {
        for (const base of baseModels) {
          // Skip the default combination (all/all/all/all)
          if (nsfwLevel === nsfw[0] && period === timePeriods[0] && type === modelTypes[0] && base === baseModels[0]) {
            continue
          }

          // Build filter path - include ALL non-default values
          const parts: string[] = []
          if (nsfwLevel !== nsfw[0]) parts.push(nsfwLevel)
          if (period !== timePeriods[0]) parts.push(period)
          if (type !== modelTypes[0]) parts.push(type)
          if (base !== baseModels[0]) parts.push(base)

          // Add if there's at least one filter
          if (parts.length > 0) {
            filters.add(`filter/${parts.join('/')}`)
          }
        }
      }
    }
  }

  return Array.from(filters).sort()
}

// Generate all combinations at module load time
export const CIVITAI_FILTERS = generateAllCivitAIFilterCombinations() as readonly string[]

/**
 * HuggingFace filter paths
 * TEAM-467: PROGRAMMATICALLY generate ALL combinations
 * Uses SHARED URL slug constants from marketplace-node (source of truth)
 */
function generateAllHFFilterCombinations(): string[] {
  const sorts = HF_URL_SLUGS.SORTS
  const sizes = HF_URL_SLUGS.SIZES
  const licenses = HF_URL_SLUGS.LICENSES

  const filters = new Set<string>() // Use Set to avoid duplicates

  // Generate ALL combinations: sort × size × license
  for (const sort of sorts) {
    for (const size of sizes) {
      for (const license of licenses) {
        // Skip the default combination (downloads/all/all)
        if (sort === sorts[0] && size === sizes[0] && license === licenses[0]) {
          continue
        }

        // Build filter path - include ALL non-default values
        const parts: string[] = []
        if (sort !== sorts[0]) parts.push(sort)
        if (size !== sizes[0]) parts.push(size)
        if (license !== licenses[0]) parts.push(license)

        // Add if there's at least one filter
        if (parts.length > 0) {
          filters.add(`filter/${parts.join('/')}`)
        }
      }
    }
  }

  return Array.from(filters).sort()
}

// Generate all combinations at module load time
export const HF_FILTERS = generateAllHFFilterCombinations() as readonly string[]

/**
 * Type-safe filter types
 */
export type CivitAIFilter = (typeof CIVITAI_FILTERS)[number]
export type HFFilter = (typeof HF_FILTERS)[number]

/**
 * Filter metadata for display
 */
export interface FilterMetadata {
  label: string
  description?: string
  category: 'sort' | 'type' | 'base-model' | 'nsfw' | 'license' | 'language' | 'library' | 'size'
}

/**
 * CivitAI filter metadata
 * TEAM-XXX RULE ZERO: Use marketplace-node constants for metadata keys
 */
export const CIVITAI_FILTER_METADATA: Record<string, FilterMetadata> = {
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[1]]: {
    label: 'PG (Safe)',
    category: 'nsfw',
  },
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[2]]: {
    label: 'PG-13',
    category: 'nsfw',
  },
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[3]]: {
    label: 'R (Mature)',
    category: 'nsfw',
  },
  [CIVITAI_URL_SLUGS.NSFW_LEVELS[4]]: {
    label: 'X (Adult)',
    category: 'nsfw',
  },
  [CIVITAI_URL_SLUGS.TIME_PERIODS[2]]: {
    label: 'This Month',
    category: 'sort',
  },
  [CIVITAI_URL_SLUGS.TIME_PERIODS[3]]: {
    label: 'This Week',
    category: 'sort',
  },
  [CIVITAI_URL_SLUGS.MODEL_TYPES[1]]: {
    label: 'Checkpoints',
    description: 'Full model checkpoints',
    category: 'type',
  },
  [CIVITAI_URL_SLUGS.MODEL_TYPES[2]]: {
    label: 'LoRAs',
    description: 'Low-Rank Adaptations',
    category: 'type',
  },
  [CIVITAI_URL_SLUGS.BASE_MODELS[1]]: {
    label: 'SDXL',
    description: 'Stable Diffusion XL models',
    category: 'base-model',
  },
  [CIVITAI_URL_SLUGS.BASE_MODELS[2]]: {
    label: 'SD 1.5',
    description: 'Stable Diffusion 1.5 models',
    category: 'base-model',
  },
  [CIVITAI_URL_SLUGS.BASE_MODELS[3]]: {
    label: 'SD 2.1',
    description: 'Stable Diffusion 2.1 models',
    category: 'base-model',
  },
}

/**
 * HuggingFace filter metadata
 * TEAM-XXX RULE ZERO: Use marketplace-node constants for metadata keys
 */
export const HF_FILTER_METADATA: Record<string, FilterMetadata> = {
  [HF_URL_SLUGS.SORTS[1]]: {
    label: 'Most Liked',
    category: 'sort',
  },
  [HF_URL_SLUGS.SORTS[0]]: {
    label: 'Most Downloaded',
    category: 'sort',
  },
  [HF_URL_SLUGS.SIZES[1]]: {
    label: 'Small (<7B)',
    category: 'size',
  },
  [HF_URL_SLUGS.SIZES[2]]: {
    label: 'Medium (7B-13B)',
    category: 'size',
  },
  [HF_URL_SLUGS.SIZES[3]]: {
    label: 'Large (>13B)',
    category: 'size',
  },
  [HF_URL_SLUGS.LICENSES[1]]: {
    label: 'Apache 2.0',
    category: 'license',
  },
  [HF_URL_SLUGS.LICENSES[2]]: {
    label: 'MIT',
    category: 'license',
  },
  [HF_URL_SLUGS.LICENSES[3]]: {
    label: 'Other',
    category: 'license',
  },
}

/**
 * Get all filters for manifest generation
 */
export function getAllCivitAIFilters(): readonly string[] {
  return CIVITAI_FILTERS
}

export function getAllHFFilters(): readonly string[] {
  return HF_FILTERS
}

/**
 * Get filter metadata
 */
export function getCivitAIFilterMetadata(filter: string): FilterMetadata | undefined {
  return CIVITAI_FILTER_METADATA[filter]
}

export function getHFFilterMetadata(filter: string): FilterMetadata | undefined {
  return HF_FILTER_METADATA[filter]
}
