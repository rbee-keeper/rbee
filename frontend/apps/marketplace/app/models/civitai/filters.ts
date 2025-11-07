// TEAM-422: CivitAI filter definitions for SSG pre-generation

export const CIVITAI_FILTERS = {
  // Time period - Most impactful for discovery
  timePeriod: [
    { label: 'All Time', value: 'AllTime', sort: 'Most Downloaded' },
    { label: 'Month', value: 'Month', sort: 'Most Downloaded' },
    { label: 'Week', value: 'Week', sort: 'Most Downloaded' },
    { label: 'Day', value: 'Day', sort: 'Most Downloaded' },
  ],
  
  // Model types - Core filter
  modelTypes: [
    { label: 'Checkpoint', value: 'Checkpoint' },
    { label: 'LORA', value: 'LORA' },
    { label: 'All Types', value: 'All' },
  ],
  
  // Base model - Important for compatibility
  baseModel: [
    { label: 'SDXL 1.0', value: 'SDXL 1.0' },
    { label: 'SD 1.5', value: 'SD 1.5' },
    { label: 'SD 2.1', value: 'SD 2.1' },
    { label: 'All Models', value: 'All' },
  ],
} as const

export type TimePeriod = 'AllTime' | 'Month' | 'Week' | 'Day'
export type ModelType = 'All' | 'Checkpoint' | 'LORA'
export type BaseModel = 'All' | 'SDXL 1.0' | 'SD 1.5' | 'SD 2.1'

export interface FilterConfig {
  timePeriod: TimePeriod
  modelType: ModelType
  baseModel: BaseModel
  path: string
}

// Pre-generate these popular combinations
// TEAM-460: Added 'filter/' prefix to avoid route conflicts with [slug]
export const PREGENERATED_FILTERS: FilterConfig[] = [
  // Default view
  { timePeriod: 'AllTime', modelType: 'All', baseModel: 'All', path: '' },
  
  // Popular time periods
  { timePeriod: 'Month', modelType: 'All', baseModel: 'All', path: 'filter/month' },
  { timePeriod: 'Week', modelType: 'All', baseModel: 'All', path: 'filter/week' },
  
  // Model type filters
  { timePeriod: 'AllTime', modelType: 'Checkpoint', baseModel: 'All', path: 'filter/checkpoints' },
  { timePeriod: 'AllTime', modelType: 'LORA', baseModel: 'All', path: 'filter/loras' },
  
  // Base model filters
  { timePeriod: 'AllTime', modelType: 'All', baseModel: 'SDXL 1.0', path: 'filter/sdxl' },
  { timePeriod: 'AllTime', modelType: 'All', baseModel: 'SD 1.5', path: 'filter/sd15' },
  
  // Popular combinations
  { timePeriod: 'Month', modelType: 'Checkpoint', baseModel: 'SDXL 1.0', path: 'filter/month/checkpoints/sdxl' },
  { timePeriod: 'Month', modelType: 'LORA', baseModel: 'SDXL 1.0', path: 'filter/month/loras/sdxl' },
  { timePeriod: 'Week', modelType: 'Checkpoint', baseModel: 'SDXL 1.0', path: 'filter/week/checkpoints/sdxl' },
]

// Helper to build API parameters from filter config
export function buildFilterParams(config: FilterConfig) {
  const params: {
    limit?: number
    types?: string[]
    sort?: string
    period?: string
    baseModel?: string
  } = {
    limit: 100,
  }
  
  // Model types
  if (config.modelType !== 'All') {
    params.types = [config.modelType]
  } else {
    params.types = ['Checkpoint', 'LORA']
  }
  
  // Time period (affects sort)
  if (config.timePeriod !== 'AllTime') {
    params.period = config.timePeriod
  }
  params.sort = 'Most Downloaded'
  
  // Base model (would need API support)
  if (config.baseModel !== 'All') {
    params.baseModel = config.baseModel
  }
  
  return params
}

// Helper to get filter config from path
export function getFilterFromPath(path: string): FilterConfig {
  const found = PREGENERATED_FILTERS.find(f => f.path === path)
  return found || PREGENERATED_FILTERS[0]
}

// Helper to build URL from filter config
export function buildFilterUrl(config: Partial<FilterConfig>): string {
  const found = PREGENERATED_FILTERS.find(
    f => 
      f.timePeriod === (config.timePeriod || 'AllTime') &&
      f.modelType === (config.modelType || 'All') &&
      f.baseModel === (config.baseModel || 'All')
  )
  
  if (found) {
    return found.path ? `/models/civitai/${found.path}` : '/models/civitai'
  }
  
  // Fallback to default
  return '/models/civitai'
}
