// TEAM-464: Shared filter configuration
// Used by both manifest generation and filter pages
// Single source of truth for all filter combinations

import {
  CIVITAI_URL_SLUGS,
  HF_URL_SLUGS,
} from '@rbee/marketplace-node'

/**
 * CivitAI filter paths
 * TEAM-467: PROGRAMMATICALLY generate ALL combinations
 * Uses SHARED URL slug constants (NOT API enum values)
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
          if (
            nsfwLevel === 'all' &&
            period === 'all' &&
            type === 'all' &&
            base === 'all'
          ) {
            continue
          }
          
          // Build filter path - include ALL non-default values
          const parts: string[] = []
          if (nsfwLevel !== 'all') parts.push(nsfwLevel)
          if (period !== 'all') parts.push(period)
          if (type !== 'all') parts.push(type)
          if (base !== 'all') parts.push(base)
          
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
 * Uses SHARED URL slug constants (NOT API enum values)
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
        if (sort === 'downloads' && size === 'all' && license === 'all') {
          continue
        }
        
        // Build filter path - include ALL non-default values
        const parts: string[] = []
        if (sort !== 'downloads') parts.push(sort)
        if (size !== 'all') parts.push(size)
        if (license !== 'all') parts.push(license)
        
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
export type CivitAIFilter = typeof CIVITAI_FILTERS[number]
export type HFFilter = typeof HF_FILTERS[number]

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
 */
export const CIVITAI_FILTER_METADATA: Record<string, FilterMetadata> = {
  'AllTime/All/All/downloads/Soft': {
    label: 'Most Downloaded (All Time)',
    category: 'sort',
  },
  'AllTime/All/All/likes/Soft': {
    label: 'Most Liked (All Time)',
    category: 'sort',
  },
  'AllTime/All/All/rating/Soft': {
    label: 'Highest Rated (All Time)',
    category: 'sort',
  },
  'Week/All/All/downloads/Soft': {
    label: 'Most Downloaded (This Week)',
    category: 'sort',
  },
  'Month/All/All/downloads/Soft': {
    label: 'Most Downloaded (This Month)',
    category: 'sort',
  },
  'filter/checkpoints': {
    label: 'Checkpoints',
    description: 'Full model checkpoints',
    category: 'type',
  },
  'filter/loras': {
    label: 'LoRAs',
    description: 'Low-Rank Adaptations',
    category: 'type',
  },
  'filter/sdxl': {
    label: 'SDXL',
    description: 'Stable Diffusion XL models',
    category: 'base-model',
  },
  'filter/sd15': {
    label: 'SD 1.5',
    description: 'Stable Diffusion 1.5 models',
    category: 'base-model',
  },
  'filter/flux': {
    label: 'Flux',
    description: 'Flux models',
    category: 'base-model',
  },
  'filter/pg': {
    label: 'PG (Safe)',
    category: 'nsfw',
  },
  'filter/pg13': {
    label: 'PG-13',
    category: 'nsfw',
  },
  'filter/r': {
    label: 'R (Mature)',
    category: 'nsfw',
  },
  'filter/x': {
    label: 'X (Adult)',
    category: 'nsfw',
  },
}

/**
 * HuggingFace filter metadata
 */
export const HF_FILTER_METADATA: Record<string, FilterMetadata> = {
  'likes': {
    label: 'Most Liked',
    category: 'sort',
  },
  'recent': {
    label: 'Recently Updated',
    category: 'sort',
  },
  'trending': {
    label: 'Trending',
    category: 'sort',
  },
  'downloads': {
    label: 'Most Downloaded',
    category: 'sort',
  },
  'text-generation': {
    label: 'Text Generation',
    description: 'Language models for text generation',
    category: 'type',
  },
  'sentence-similarity': {
    label: 'Sentence Similarity',
    description: 'Models for comparing sentences',
    category: 'type',
  },
  'small': {
    label: 'Small (<1GB)',
    category: 'size',
  },
  'medium': {
    label: 'Medium (1-10GB)',
    category: 'size',
  },
  'large': {
    label: 'Large (>10GB)',
    category: 'size',
  },
  'apache-2.0': {
    label: 'Apache 2.0',
    category: 'license',
  },
  'mit': {
    label: 'MIT',
    category: 'license',
  },
  'transformers': {
    label: 'Transformers',
    category: 'library',
  },
  'en': {
    label: 'English',
    category: 'language',
  },
  'multilingual': {
    label: 'Multilingual',
    category: 'language',
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
