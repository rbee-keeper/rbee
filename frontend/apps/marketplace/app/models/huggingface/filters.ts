// TEAM-461: HuggingFace model filter definitions - SSG-compatible filtering system
import type { FilterConfig, FilterGroup } from '@/lib/filters/types'

export interface HuggingFaceFilters {
  sort: 'downloads' | 'likes'  // TEAM-467: Removed 'recent' - API doesn't support it
  size: 'all' | 'small' | 'medium' | 'large'
  license: 'all' | 'apache' | 'mit' | 'other'
}

// Filter group definitions (left side - actual filters)
export const HUGGINGFACE_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: 'all' },
      { label: 'Small (<7B)', value: 'small' },
      { label: 'Medium (7B-13B)', value: 'medium' },
      { label: 'Large (>13B)', value: 'large' },
    ],
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { label: 'All Licenses', value: 'all' },
      { label: 'Apache 2.0', value: 'apache' },
      { label: 'MIT', value: 'mit' },
      { label: 'Other', value: 'other' },
    ],
  },
]

// Sort group definition (right side - sorting only)
// TEAM-467: Removed 'recent' - HuggingFace API doesn't support it
export const HUGGINGFACE_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    // 'Recently Updated' removed - HuggingFace API returns "Bad Request"
  ],
}

import { HF_SORTS, HF_SIZES, HF_LICENSES } from '@/config/filter-constants'

// TEAM-467: PROGRAMMATICALLY generate ALL filter combinations
// Uses SHARED constants from filter-constants.ts
function generateAllHFFilterConfigs(): FilterConfig<HuggingFaceFilters>[] {
  const sorts = HF_SORTS
  const sizes = HF_SIZES
  const licenses = HF_LICENSES
  
  const configs: FilterConfig<HuggingFaceFilters>[] = []
  
  // Generate all combinations: sort × size × license
  for (const sort of sorts) {
    for (const size of sizes) {
      for (const license of licenses) {
        // Skip the default combination (downloads/all/all) - that's the base route
        if (sort === 'downloads' && size === 'all' && license === 'all') {
          configs.push({
            filters: { sort, size, license },
            path: ''  // Default route
          })
          continue
        }
        
        // Build filter path - include ALL non-default values
        const parts: string[] = []
        if (sort !== 'downloads') parts.push(sort)
        if (size !== 'all') parts.push(size)
        if (license !== 'all') parts.push(license)
        
        // Add if there's at least one filter
        if (parts.length > 0) {
          configs.push({
            filters: { sort, size, license },
            path: `filter/${parts.join('/')}`
          })
        }
      }
    }
  }
  
  return configs
}

// TEAM-467: Generate all filter combinations at module load time
// This MUST match the manifest generation in config/filters.ts
export const PREGENERATED_HF_FILTERS: FilterConfig<HuggingFaceFilters>[] = generateAllHFFilterConfigs()

/**
 * Build URL from HuggingFace filter configuration
 */
export function buildHFFilterUrl(filters: Partial<HuggingFaceFilters>): string {
  const found = PREGENERATED_HF_FILTERS.find(
    (f) =>
      f.filters.sort === (filters.sort || 'downloads') &&
      f.filters.size === (filters.size || 'all') &&
      f.filters.license === (filters.license || 'all'),
  )

  return found?.path ? `/models/huggingface/${found.path}` : '/models/huggingface'
}

/**
 * Get filter configuration from URL path
 */
export function getHFFilterFromPath(path: string): HuggingFaceFilters {
  const found = PREGENERATED_HF_FILTERS.find((f) => f.path === path)
  return found?.filters || PREGENERATED_HF_FILTERS[0].filters
}

/**
 * Build filter description for display
 */
export function buildHFFilterDescription(filters: HuggingFaceFilters): string {
  const parts: string[] = []

  if (filters.sort !== 'downloads') {
    parts.push(filters.sort === 'likes' ? 'Most Liked' : 'Most Downloaded')
  }

  if (filters.size !== 'all') {
    if (filters.size === 'small') parts.push('Small Models')
    else if (filters.size === 'medium') parts.push('Medium Models')
    else parts.push('Large Models')
  }

  if (filters.license !== 'all') {
    parts.push(filters.license === 'apache' ? 'Apache 2.0' : filters.license === 'mit' ? 'MIT' : 'Other License')
  }

  return parts.length > 0 ? parts.join(' · ') : 'All Models'
}

/**
 * Build API parameters from filter config
 * TEAM-467: Removed 'recent' - API doesn't support it
 */
export function buildHFFilterParams(filters: HuggingFaceFilters) {
  const params: {
    limit?: number
    sort?: 'popular' | 'trending'  // Removed 'recent'
    // Add more API params as needed
  } = {
    limit: 100,
  }

  // Sort parameter - map to API values
  if (filters.sort === 'likes') {
    params.sort = 'trending'
  } else {
    params.sort = 'popular'
  }

  // Size and license filtering would need to be done client-side or via API if supported

  return params
}
