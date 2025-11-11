// TEAM-XXX: Shared constants across all providers

/**
 * URL slugs for filter paths (lowercase, URL-friendly)
 * Maps to API enum values via parser functions
 */
export const URL_SLUGS = {
  // CivitAI model types
  CHECKPOINTS: 'checkpoints',
  LORAS: 'loras',
  
  // CivitAI base models
  SDXL: 'sdxl',
  SD15: 'sd15',
  SD21: 'sd21',
  PONY: 'pony',
  FLUX: 'flux',
  
  // CivitAI time periods
  WEEK: 'week',
  MONTH: 'month',
  YEAR: 'year',
  DAY: 'day',
  ALL_TIME: 'all',
  
  // CivitAI NSFW levels (URL slugs)
  PG: 'pg',
  PG13: 'pg13',
  R: 'r',
  X: 'x',
  
  // HuggingFace sorts
  DOWNLOADS: 'downloads',
  LIKES: 'likes',
  RECENT: 'recent',
  TRENDING: 'trending',
  
  // HuggingFace sizes
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
  
  // HuggingFace licenses
  APACHE: 'apache',
  MIT: 'mit',
  OTHER: 'other',
  
  // Common
  ALL: 'all',
  FILTER_PREFIX: 'filter',
} as const

/**
 * URL slug to API enum value mappings
 */
export const SLUG_TO_API = {
  // CivitAI model types
  [URL_SLUGS.CHECKPOINTS]: 'Checkpoint',
  [URL_SLUGS.LORAS]: 'LORA',
  
  // CivitAI base models
  [URL_SLUGS.SDXL]: 'SDXL 1.0',
  [URL_SLUGS.SD15]: 'SD 1.5',
  [URL_SLUGS.SD21]: 'SD 2.1',
  
  // CivitAI time periods
  [URL_SLUGS.WEEK]: 'Week',
  [URL_SLUGS.MONTH]: 'Month',
  [URL_SLUGS.YEAR]: 'Year',
  [URL_SLUGS.DAY]: 'Day',
  
  // CivitAI NSFW levels
  [URL_SLUGS.PG]: 'None',
  [URL_SLUGS.PG13]: 'Soft',
  [URL_SLUGS.R]: 'Mature',
  [URL_SLUGS.X]: 'X',
  
  // HuggingFace (most are already API values)
  [URL_SLUGS.DOWNLOADS]: 'Downloads',
  [URL_SLUGS.LIKES]: 'Likes',
  [URL_SLUGS.SMALL]: 'Small',
  [URL_SLUGS.MEDIUM]: 'Medium',
  [URL_SLUGS.LARGE]: 'Large',
  [URL_SLUGS.APACHE]: 'Apache',
  [URL_SLUGS.MIT]: 'MIT',
  [URL_SLUGS.OTHER]: 'Other',
} as const

/**
 * Display labels for UI
 */
export const DISPLAY_LABELS = {
  // CivitAI sorts
  MOST_DOWNLOADED: 'Most Downloaded',
  HIGHEST_RATED: 'Highest Rated',
  NEWEST: 'Newest',
  
  // HuggingFace sorts
  MOST_LIKED: 'Most Liked',
  RECENTLY_UPDATED: 'Recently Updated',
  TRENDING: 'Trending',
  
  // Sizes
  SMALL_MODELS: 'Small Models',
  MEDIUM_MODELS: 'Medium Models',
  LARGE_MODELS: 'Large Models',
  
  // Licenses
  APACHE_2_0: 'Apache 2.0',
  MIT_LICENSE: 'MIT',
  OTHER_LICENSE: 'Other License',
  
  // Common
  ALL_MODELS: 'All Models',
} as const

/**
 * Combined filter defaults (backwards compatibility)
 * @deprecated Import from civitai/constants or huggingface/constants instead
 */
export const FILTER_DEFAULTS = {
  // CivitAI defaults
  CIVITAI_TIME_PERIOD: 'AllTime',
  CIVITAI_MODEL_TYPE: 'All',
  CIVITAI_BASE_MODEL: 'All',
  CIVITAI_SORT: 'Most Downloaded',
  CIVITAI_NSFW_LEVEL: 'XXX',
  CIVITAI_BLUR_MATURE: false,
  CIVITAI_LIMIT: 100,
  
  // HuggingFace defaults
  HF_SORT: 'Downloads',
  HF_SIZE: 'All',
  HF_LICENSE: 'All',
  HF_LIMIT: 100,
} as const
