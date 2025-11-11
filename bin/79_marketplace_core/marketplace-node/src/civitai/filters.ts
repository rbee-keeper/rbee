// TEAM-XXX: CivitAI filter utilities
// Business logic for filtering and sorting CivitAI models

import type { FilterableModel } from '../shared/index.js'
import { CIVITAI_DEFAULTS, CIVITAI_SORT } from './constants'

export type { FilterableModel }

/**
 * CivitAI filter options
 */
export interface CivitAIFilterOptions {
  modelType?: string
  baseModel?: string
  sort?: string
}

/**
 * Filter CivitAI models by type and base model
 */
export function filterCivitAIModels(models: FilterableModel[], options: CivitAIFilterOptions): FilterableModel[] {
  let result = [...models]

  // Filter by model type (if available in tags)
  if (options.modelType && options.modelType !== CIVITAI_DEFAULTS.MODEL_TYPE) {
    const modelType = options.modelType.toLowerCase()
    result = result.filter((model) => {
      const tags = model.tags.map((t) => t.toLowerCase())
      return tags.includes(modelType)
    })
  }

  // Filter by base model (if available in tags)
  if (options.baseModel && options.baseModel !== CIVITAI_DEFAULTS.BASE_MODEL) {
    const baseModel = options.baseModel.toLowerCase().replace(/\s/g, '')
    result = result.filter((model) => {
      const tags = model.tags.map((t) => t.toLowerCase())
      return tags.some((tag) => tag.includes(baseModel))
    })
  }

  return result
}

/**
 * Sort CivitAI models by downloads, likes, or other criteria
 */
export function sortCivitAIModels(models: FilterableModel[], sortBy: string): FilterableModel[] {
  const result = [...models]

  result.sort((a, b) => {
    if (sortBy === CIVITAI_SORT.MOST_DOWNLOADED) {
      return (b.downloads || 0) - (a.downloads || 0)
    }
    if (sortBy === CIVITAI_SORT.HIGHEST_RATED) {
      return (b.likes || 0) - (a.likes || 0)
    }
    // Newest - would need createdAt field
    return 0
  })

  return result
}

/**
 * Apply all filters and sorting to CivitAI models
 */
export function applyCivitAIFilters(models: FilterableModel[], options: CivitAIFilterOptions): FilterableModel[] {
  let result = filterCivitAIModels(models, options)
  if (options.sort) {
    result = sortCivitAIModels(result, options.sort)
  }
  return result
}
