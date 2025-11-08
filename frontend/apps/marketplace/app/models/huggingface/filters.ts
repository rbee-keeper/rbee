// TEAM-461: HuggingFace model filter definitions - SSG-compatible filtering system
import type { FilterGroup, FilterConfig } from '@/lib/filters/types'

export interface HuggingFaceFilters {
  sort: 'downloads' | 'likes' | 'recent'
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
export const HUGGINGFACE_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    { label: 'Recently Updated', value: 'recent' },
  ],
}

// Pre-generated filter combinations for SSG
export const PREGENERATED_HF_FILTERS: FilterConfig<HuggingFaceFilters>[] = [
  // Default view
  { filters: { sort: 'downloads', size: 'all', license: 'all' }, path: '' },
  
  // By sort
  { filters: { sort: 'likes', size: 'all', license: 'all' }, path: 'filter/likes' },
  { filters: { sort: 'recent', size: 'all', license: 'all' }, path: 'filter/recent' },
  
  // By size
  { filters: { sort: 'downloads', size: 'small', license: 'all' }, path: 'filter/small' },
  { filters: { sort: 'downloads', size: 'medium', license: 'all' }, path: 'filter/medium' },
  { filters: { sort: 'downloads', size: 'large', license: 'all' }, path: 'filter/large' },
  
  // By license
  { filters: { sort: 'downloads', size: 'all', license: 'apache' }, path: 'filter/apache' },
  { filters: { sort: 'downloads', size: 'all', license: 'mit' }, path: 'filter/mit' },
  
  // Popular combinations
  { filters: { sort: 'downloads', size: 'small', license: 'apache' }, path: 'filter/small/apache' },
  { filters: { sort: 'likes', size: 'small', license: 'all' }, path: 'filter/likes/small' },
]

/**
 * Build URL from HuggingFace filter configuration
 */
export function buildHFFilterUrl(filters: Partial<HuggingFaceFilters>): string {
  const found = PREGENERATED_HF_FILTERS.find(
    f => 
      f.filters.sort === (filters.sort || 'downloads') &&
      f.filters.size === (filters.size || 'all') &&
      f.filters.license === (filters.license || 'all')
  )
  
  return found?.path ? `/models/huggingface/${found.path}` : '/models/huggingface'
}

/**
 * Get filter configuration from URL path
 */
export function getHFFilterFromPath(path: string): HuggingFaceFilters {
  const found = PREGENERATED_HF_FILTERS.find(f => f.path === path)
  return found?.filters || PREGENERATED_HF_FILTERS[0].filters
}

/**
 * Build filter description for display
 */
export function buildHFFilterDescription(filters: HuggingFaceFilters): string {
  const parts: string[] = []
  
  if (filters.sort === 'likes') parts.push('Most Liked')
  else if (filters.sort === 'recent') parts.push('Recently Updated')
  else parts.push('Most Downloaded')
  
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
 */
export function buildHFFilterParams(filters: HuggingFaceFilters) {
  const params: {
    limit?: number
    sort?: 'popular' | 'recent' | 'trending'
    // Add more API params as needed
  } = {
    limit: 100,
  }
  
  // Sort parameter - map to API values
  if (filters.sort === 'likes') {
    params.sort = 'trending'
  } else if (filters.sort === 'recent') {
    params.sort = 'recent'
  } else {
    params.sort = 'popular'
  }
  
  // Size and license filtering would need to be done client-side or via API if supported
  
  return params
}
