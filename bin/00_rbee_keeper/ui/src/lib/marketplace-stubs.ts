// TEAM-476: Temporary stubs for marketplace-node imports
// TODO: Replace with client-side fetcher implementation

// ============================================================================
// TYPES
// ============================================================================

export type TimePeriod = 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
export type CivitaiModelType = 'All' | 'Checkpoint' | 'LORA' | 'TextualInversion' | 'Hypernetwork' | 'AestheticGradient' | 'Controlnet' | 'Poses'
export type BaseModel = 'All' | 'SD 1.5' | 'SD 2.1' | 'SDXL 1.0' | 'SDXL Turbo' | 'Pony' | 'Illustrious'
export type CivitaiSort = 'Most Downloaded' | 'Most Liked' | 'Newest' | 'Updated'
export type HuggingFaceSort = 'trending' | 'downloads' | 'likes' | 'updated' | 'created'

export interface FilterableModel {
  id: string
  name: string
  description?: string
  downloads?: number
  likes?: number
  tags?: string[]
  type?: string
  [key: string]: any
}

// ============================================================================
// CONSTANTS
// ============================================================================

export const CIVITAI_DEFAULTS = {
  TIME_PERIOD: 'Month' as TimePeriod,
  MODEL_TYPE: 'All' as CivitaiModelType,
  BASE_MODEL: 'All' as BaseModel,
  SORT: 'Most Downloaded' as CivitaiSort,
  NSFW_LEVEL: 'XXX',
  BLUR_MATURE: false,
  LIMIT: 100,
} as const

export const HF_DEFAULTS = {
  SORT: 'trending' as HuggingFaceSort,
  SIZE: 'All',
  LICENSE: 'All',
  LIMIT: 100,
} as const

export const HF_SIZES = ['All', '<1B', '1B-3B', '3B-7B', '7B+'] as const
export const HF_LICENSES = ['All', 'apache-2.0', 'mit', 'cc-by-4.0', 'other'] as const

// ============================================================================
// FILTER FUNCTIONS
// ============================================================================

/**
 * TODO: Implement client-side CivitAI filtering
 * This should filter models based on the provided filters
 */
export function applyCivitAIFilters(
  models: FilterableModel[],
  filters: {
    timePeriod?: TimePeriod
    modelType?: CivitaiModelType
    baseModel?: BaseModel
    sort?: CivitaiSort
  }
): FilterableModel[] {
  console.warn('[STUB] applyCivitAIFilters called - TODO: implement client-side filtering')
  
  // TODO: Implement actual filtering logic
  // For now, just return all models
  return models
}

/**
 * TODO: Implement client-side HuggingFace filtering
 * This should filter models based on the provided filters
 */
export function applyHuggingFaceFilters(
  models: FilterableModel[],
  filters: {
    sort?: HuggingFaceSort
    size?: typeof HF_SIZES[number]
    license?: typeof HF_LICENSES[number]
  }
): FilterableModel[] {
  console.warn('[STUB] applyHuggingFaceFilters called - TODO: implement client-side filtering')
  
  // TODO: Implement actual filtering logic
  // For now, just return all models
  return models
}

/**
 * TODO: Build filter description for HuggingFace
 */
export function buildHuggingFaceFilterDescription(filters: {
  sort?: HuggingFaceSort
  size?: typeof HF_SIZES[number]
  license?: typeof HF_LICENSES[number]
}): string {
  console.warn('[STUB] buildHuggingFaceFilterDescription called - TODO: implement')
  
  // TODO: Build actual description
  return 'All models'
}
