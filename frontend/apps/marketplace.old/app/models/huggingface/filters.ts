// TEAM-461: HuggingFace model filter definitions - SSG-compatible filtering system
// TEAM-XXX RULE ZERO: Import constants from marketplace-node (source of truth)

import type { HuggingFaceSort } from '@rbee/marketplace-node'
import { HF_DEFAULTS, HF_LICENSES, HF_SIZES, HF_SORTS, HF_URL_SLUGS } from '@rbee/marketplace-node'
import type { FilterConfig, FilterGroup } from '@/lib/filters/types'

export interface HuggingFaceFilters {
  sort: HuggingFaceSort // TEAM-467: Removed 'recent' - API doesn't support it
  size: (typeof HF_SIZES)[number]
  license: (typeof HF_LICENSES)[number]
}

// Filter group definitions (left side - actual filters)
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export const HUGGINGFACE_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: HF_SIZES[0] },
      { label: 'Small (<7B)', value: HF_SIZES[1] },
      { label: 'Medium (7B-13B)', value: HF_SIZES[2] },
      { label: 'Large (>13B)', value: HF_SIZES[3] },
    ],
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { label: 'All Licenses', value: HF_LICENSES[0] },
      { label: 'Apache 2.0', value: HF_LICENSES[1] },
      { label: 'MIT', value: HF_LICENSES[2] },
      { label: 'Other', value: HF_LICENSES[3] },
    ],
  },
]

// Sort group definition (right side - sorting only)
// TEAM-467: Removed 'recent' - HuggingFace API doesn't support it
// TEAM-XXX RULE ZERO: Using constants from marketplace-node
export const HUGGINGFACE_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: HF_SORTS[0] },
    { label: 'Most Likes', value: HF_SORTS[1] },
    // 'Recently Updated' removed - HuggingFace API returns "Bad Request"
  ],
}

// TEAM-XXX RULE ZERO: Using constants from marketplace-node instead of local duplicates

// TEAM-467: PROGRAMMATICALLY generate ALL filter combinations
// Uses SHARED constants from marketplace-node (source of truth)
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
        if (sort === HF_SORTS[0] && size === HF_SIZES[0] && license === HF_LICENSES[0]) {
          configs.push({
            filters: { sort, size, license },
            path: '', // Default route
          })
          continue
        }

        // Build filter path - include ALL non-default values
        const parts: string[] = []
        if (sort !== HF_SORTS[0]) parts.push(HF_URL_SLUGS.SORTS[sorts.indexOf(sort as (typeof HF_SORTS)[number])])
        if (size !== HF_SIZES[0]) parts.push(HF_URL_SLUGS.SIZES[sizes.indexOf(size as (typeof HF_SIZES)[number])])
        if (license !== HF_LICENSES[0])
          parts.push(HF_URL_SLUGS.LICENSES[licenses.indexOf(license as (typeof HF_LICENSES)[number])])

        // Add if there's at least one filter
        if (parts.length > 0) {
          configs.push({
            filters: { sort, size, license },
            path: `filter/${parts.join('/')}`,
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
 * TEAM-XXX RULE ZERO: Using constants from marketplace-node
 */
export function buildHFFilterUrl(filters: Partial<HuggingFaceFilters>): string {
  const found = PREGENERATED_HF_FILTERS.find(
    (f) =>
      f.filters.sort === (filters.sort || HF_SORTS[0]) &&
      f.filters.size === (filters.size || HF_SIZES[0]) &&
      f.filters.license === (filters.license || HF_LICENSES[0]),
  )

  return found?.path ? `/models/huggingface/${found.path}` : '/models/huggingface'
}

/**
 * Get filter configuration from URL path
 * TEAM-XXX RULE ZERO: Using constants from marketplace-node
 */
export function getHFFilterFromPath(path: string): HuggingFaceFilters {
  const found = PREGENERATED_HF_FILTERS.find((f) => f.path === path)
  return found?.filters || PREGENERATED_HF_FILTERS[0].filters
}

/**
 * Build filter description for display
 * TEAM-XXX RULE ZERO: Using constants from marketplace-node
 */
export function buildHFFilterDescription(filters: HuggingFaceFilters): string {
  const parts: string[] = []

  if (filters.sort !== HF_SORTS[0]) {
    parts.push(filters.sort === HF_SORTS[1] ? 'Most Liked' : 'Most Downloaded')
  }

  if (filters.size !== HF_SIZES[0]) {
    if (filters.size === HF_SIZES[1]) parts.push('Small Models')
    else if (filters.size === HF_SIZES[2]) parts.push('Medium Models')
    else parts.push('Large Models')
  }

  if (filters.license !== HF_LICENSES[0]) {
    parts.push(
      filters.license === HF_LICENSES[1] ? 'Apache 2.0' : filters.license === HF_LICENSES[2] ? 'MIT' : 'Other License',
    )
  }

  return parts.length > 0 ? parts.join(' · ') : 'All Models'
}

/**
 * Build API parameters from filter config
 * TEAM-467: Removed 'recent' - API doesn't support it
 * TEAM-XXX RULE ZERO: Using constants from marketplace-node
 */
export function buildHFFilterParams(filters: HuggingFaceFilters) {
  const params: {
    limit?: number
    sort?: 'popular' | 'trending' // Removed 'recent'
    // Add more API params as needed
  } = {
    limit: HF_DEFAULTS.LIMIT,
  }

  // Sort parameter - map to API values
  if (filters.sort === HF_SORTS[1]) {
    params.sort = 'trending'
  } else {
    params.sort = 'popular'
  }

  // Size and license filtering would need to be done client-side or via API if supported

  return params
}
