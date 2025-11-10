// TEAM-464: Shared filter configuration
// Used by both manifest generation and filter pages
// Single source of truth for all filter combinations

/**
 * CivitAI filter paths
 * These correspond to routes like /models/civitai/[...filter]
 * SOURCED FROM: app/models/civitai/filters.ts PREGENERATED_FILTERS
 */
export const CIVITAI_FILTERS = [
  // NSFW filter levels (most important)
  'filter/pg',      // PG (Safe for work)
  'filter/pg13',    // PG-13 (Suggestive)
  'filter/r',       // R (Mature)
  'filter/x',       // X (Explicit)
  'filter/xxx',     // XXX (Pornographic)
  
  // Popular time periods (PG only)
  'filter/month',
  'filter/week',
  
  // Model type filters (PG only)
  'filter/checkpoints',
  'filter/loras',
  
  // Base model filters (PG only)
  'filter/sdxl',
  'filter/sd15',
  
  // Popular combinations (PG only)
  'filter/month/checkpoints/sdxl',
  'filter/month/loras/sdxl',
  'filter/week/checkpoints/sdxl',
  
  // NSFW + Model Type combinations
  'filter/r/checkpoints',
  'filter/r/loras',
] as const

/**
 * HuggingFace filter paths
 * These correspond to routes like /models/huggingface/[...filter]
 * SOURCED FROM: app/models/huggingface/filters.ts PREGENERATED_HF_FILTERS
 * 
 * NOTE: HuggingFace API only supports `limit` parameter
 * All filtering/sorting done CLIENT-SIDE after fetch
 */
export const HF_FILTERS = [
  // Sorting
  'filter/likes',
  'filter/recent',
  
  // Size filters (client-side)
  'filter/small',
  'filter/medium',
  'filter/large',
  
  // License filters (client-side)
  'filter/apache',
  'filter/mit',
  
  // Combined filters (client-side)
  'filter/small/apache',
  'filter/likes/small',
] as const

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
