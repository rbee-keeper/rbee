// TEAM-XXX: HuggingFace filter utilities
// Business logic for filtering and sorting HuggingFace models

import type { FilterableModel } from '../shared/index.js'
import { HF_DEFAULTS, HF_LICENSE, HF_SIZE, HF_SORT, LICENSE_PATTERNS, MODEL_SIZE_PATTERNS } from './constants'

export type { FilterableModel }

// TEAM-XXX: Display labels for filter descriptions
const DISPLAY_LABELS = {
  MOST_DOWNLOADED: 'Most Downloaded',
  MOST_LIKED: 'Most Liked',
  SMALL_MODELS: 'Small Models',
  MEDIUM_MODELS: 'Medium Models',
  LARGE_MODELS: 'Large Models',
  APACHE_2_0: 'Apache 2.0',
  MIT_LICENSE: 'MIT',
  OTHER_LICENSE: 'Other License',
  ALL_MODELS: 'All Models',
} as const

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

  // Filter by size
  // NOTE: This is a FALLBACK heuristic based on model names (e.g., "7b", "13B").
  // BEST PRACTICE: Use model.safetensors.parameters.total from HF API for accurate size detection.
  // The frontend/API layer should fetch full model data and use MODEL_SIZE_THRESHOLDS instead.
  if (options.size && options.size !== HF_DEFAULTS.SIZE) {
    result = result.filter((model) => {
      const name = model.name.toLowerCase()
      if (options.size === HF_SIZE.SMALL) {
        return MODEL_SIZE_PATTERNS.SMALL.some((pattern) => name.includes(pattern))
      } else if (options.size === HF_SIZE.MEDIUM) {
        return MODEL_SIZE_PATTERNS.MEDIUM.some((pattern) => name.includes(pattern))
      } else {
        // Large
        return MODEL_SIZE_PATTERNS.LARGE.some((pattern) => name.includes(pattern))
      }
    })
  }

  // Filter by license
  // NOTE: This is a FALLBACK heuristic based on license string matching.
  // BEST PRACTICE: Use model.cardData.license from HF API for accurate license detection.
  // The frontend/API layer should fetch full model data with cardData included.
  if (options.license && options.license !== HF_DEFAULTS.LICENSE && result.length > 0) {
    result = result.filter((model) => {
      const license = model.license?.toLowerCase() ?? ''
      if (options.license === HF_LICENSE.APACHE) return license.includes(LICENSE_PATTERNS.APACHE)
      if (options.license === HF_LICENSE.MIT) return license.includes(LICENSE_PATTERNS.MIT)
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
    if (sortBy === HF_SORT.DOWNLOADS) {
      return (b.downloads || 0) - (a.downloads || 0)
    }
    if (sortBy === HF_SORT.LIKES) {
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

  if (options.sort === HF_SORT.LIKES) parts.push(DISPLAY_LABELS.MOST_LIKED)
  else parts.push(DISPLAY_LABELS.MOST_DOWNLOADED)

  if (options.size && options.size !== HF_DEFAULTS.SIZE) {
    if (options.size === HF_SIZE.SMALL) parts.push(DISPLAY_LABELS.SMALL_MODELS)
    else if (options.size === HF_SIZE.MEDIUM) parts.push(DISPLAY_LABELS.MEDIUM_MODELS)
    else parts.push(DISPLAY_LABELS.LARGE_MODELS)
  }

  if (options.license && options.license !== HF_DEFAULTS.LICENSE) {
    parts.push(
      options.license === HF_LICENSE.APACHE
        ? DISPLAY_LABELS.APACHE_2_0
        : options.license === HF_LICENSE.MIT
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
