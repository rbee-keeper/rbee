// TEAM-XXX: HuggingFace filter utilities
// Business logic for filtering and sorting HuggingFace models

import { DISPLAY_LABELS } from '../shared/constants'
import { HF_DEFAULTS, LICENSE_PATTERNS, MODEL_SIZE_PATTERNS } from './constants'

/**
 * Generic model interface for filtering
 */
export interface FilterableModel {
  id: string
  name: string
  downloads?: number | null
  likes?: number | null
  tags: string[]
  license?: string | null
  [key: string]: unknown // Allow additional fields
}

/**
 * HuggingFace filter options
 */
export interface HuggingFaceFilterOptions {
  size?: string
  license?: string
  sort?: string
}

/**
 * Filter HuggingFace models by size and license
 */
export function filterHuggingFaceModels(
  models: FilterableModel[],
  options: HuggingFaceFilterOptions,
): FilterableModel[] {
  let result = [...models]

  // Filter by size (based on model name heuristics)
  if (options.size && options.size !== HF_DEFAULTS.SIZE) {
    result = result.filter((model) => {
      const name = model.name.toLowerCase()
      if (options.size === 'Small') {
        return MODEL_SIZE_PATTERNS.SMALL.some((pattern) => name.includes(pattern))
      } else if (options.size === 'Medium') {
        return MODEL_SIZE_PATTERNS.MEDIUM.some((pattern) => name.includes(pattern))
      } else {
        // Large
        return MODEL_SIZE_PATTERNS.LARGE.some((pattern) => name.includes(pattern))
      }
    })
  }

  // Filter by license (if available in model data)
  if (options.license && options.license !== HF_DEFAULTS.LICENSE && result.length > 0) {
    result = result.filter((model) => {
      const license = model.license?.toLowerCase() ?? ''
      if (options.license === 'Apache') return license.includes(LICENSE_PATTERNS.APACHE)
      if (options.license === 'MIT') return license.includes(LICENSE_PATTERNS.MIT)
      return !license.includes(LICENSE_PATTERNS.APACHE) && !license.includes(LICENSE_PATTERNS.MIT)
    })
  }

  return result
}

/**
 * Sort HuggingFace models by downloads, likes, or other criteria
 */
export function sortHuggingFaceModels(models: FilterableModel[], sortBy: string): FilterableModel[] {
  const result = [...models]

  result.sort((a, b) => {
    if (sortBy === HF_DEFAULTS.SORT || sortBy === 'Downloads') {
      return (b.downloads || 0) - (a.downloads || 0)
    }
    if (sortBy === 'Likes') {
      return (b.likes || 0) - (a.likes || 0)
    }
    // Recent - would need updatedAt field
    return 0
  })

  return result
}

/**
 * Build filter description for HuggingFace filters
 */
export function buildHuggingFaceFilterDescription(options: HuggingFaceFilterOptions): string {
  const parts: string[] = []

  if (options.sort === 'Likes') parts.push(DISPLAY_LABELS.MOST_LIKED)
  else parts.push(DISPLAY_LABELS.MOST_DOWNLOADED)

  if (options.size && options.size !== HF_DEFAULTS.SIZE) {
    if (options.size === 'Small') parts.push(DISPLAY_LABELS.SMALL_MODELS)
    else if (options.size === 'Medium') parts.push(DISPLAY_LABELS.MEDIUM_MODELS)
    else parts.push(DISPLAY_LABELS.LARGE_MODELS)
  }

  if (options.license && options.license !== HF_DEFAULTS.LICENSE) {
    parts.push(
      options.license === 'Apache'
        ? DISPLAY_LABELS.APACHE_2_0
        : options.license === 'MIT'
          ? DISPLAY_LABELS.MIT_LICENSE
          : DISPLAY_LABELS.OTHER_LICENSE,
    )
  }

  return parts.length > 0 ? parts.join(' · ') : DISPLAY_LABELS.ALL_MODELS
}

/**
 * Apply all filters and sorting to HuggingFace models
 */
export function applyHuggingFaceFilters(
  models: FilterableModel[],
  options: HuggingFaceFilterOptions,
): FilterableModel[] {
  let result = filterHuggingFaceModels(models, options)
  if (options.sort) {
    result = sortHuggingFaceModels(result, options.sort)
  }
  return result
}
