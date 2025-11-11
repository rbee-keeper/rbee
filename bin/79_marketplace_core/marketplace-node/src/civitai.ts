// TEAM-460: CivitAI API integration
// TEAM-463: Now uses canonical types from WASM-generated artifacts-contract
// TEAM-429: Updated to use CivitaiFilters for type-safe filtering
// API Documentation: https://github.com/civitai/civitai/wiki/REST-API-Reference

// TEAM-463: Import canonical CivitAI types from WASM bindings
// These are generated from artifacts-contract via tsify
// TEAM-429: Now includes filter types from WASM bindings
import type {
  CivitaiModel,
  CivitaiModelVersion,
  CivitaiStats,
  CivitaiCreator,
  CivitaiFile,
  CivitaiImage,
  CivitaiFilters,
  TimePeriod,
  CivitaiModelType,
  BaseModel,
  CivitaiSort,
  NsfwLevel,
  NsfwFilter,
} from '../wasm/marketplace_sdk'

// TEAM-429: Re-export filter types for convenience
export type { TimePeriod, CivitaiModelType, BaseModel, CivitaiSort, NsfwLevel, NsfwFilter, CivitaiFilters }

// TEAM-463: Type aliases for backward compatibility
export type CivitAIModel = CivitaiModel
export type CivitAIModelVersion = CivitaiModelVersion

// TEAM-463: API response wrapper (not in contract, specific to API pagination)
export interface CivitAISearchResponse {
  items: CivitAIModel[]
  metadata: {
    totalItems: number
    currentPage: number
    pageSize: number
    totalPages: number
    nextPage?: string
  }
}

/**
 * Fetch models from CivitAI API
 *
 * TEAM-429: Now uses CivitaiFilters for type-safe filtering
 * @param filters - Filter configuration
 * @returns Array of CivitAI models
 */
export async function fetchCivitAIModels(
  filters: CivitaiFilters,
): Promise<CivitAIModel[]> {
  // TEAM-429: Build query params from filters
  const params = new URLSearchParams({
    limit: String(filters.limit),
    sort: filters.sort,
  })

  // Page (optional)
  if (filters.page !== null) {
    params.append('page', String(filters.page))
  }

  // Model types
  if (filters.model_type !== 'All') {
    params.append('types', filters.model_type)
  } else {
    // Default: Checkpoint and LORA
    params.append('types', 'Checkpoint')
    params.append('types', 'LORA')
  }

  // Time period
  if (filters.time_period !== 'AllTime') {
    params.append('period', filters.time_period)
  }

  // Base model
  if (filters.base_model !== 'All') {
    params.append('baseModel', filters.base_model)
  }

  // NSFW filtering - convert level to numeric values
  // TEAM-467: FAIL FAST - Validate NSFW level is valid BEFORE lookup
  const nsfwLevelMap: Record<NsfwLevel, number[]> = {
    'None': [1],
    'Soft': [1, 2],
    'Mature': [1, 2, 4],
    'X': [1, 2, 4, 8],
    'XXX': [1, 2, 4, 8, 16],
  }
  
  const nsfwLevels = nsfwLevelMap[filters.nsfw.max_level]
  if (!nsfwLevels) {
    const validLevels = Object.keys(nsfwLevelMap).join(', ')
    throw new Error(
      `❌ FATAL: Invalid NSFW level "${filters.nsfw.max_level}". Valid values: ${validLevels}\n` +
      `💡 Hint: Use 'XXX' for all NSFW levels, or 'None' for PG-only content.`
    )
  }
  
  nsfwLevels.forEach((level: number) => {
    params.append('nsfwLevel', String(level))
  })

  const url = `https://civitai.com/api/v1/models?${params}`

  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`)
    }

    const data: CivitAISearchResponse = await response.json()
    return data.items
  } catch (error) {
    console.error('[marketplace-node] CivitAI API error:', error)
    throw error
  }
}

/**
 * Fetch a specific model from CivitAI by ID
 *
 * @param modelId - CivitAI model ID
 * @returns CivitAI model details
 */
export async function fetchCivitAIModel(modelId: number): Promise<CivitAIModel> {
  const url = `https://civitai.com/api/v1/models/${modelId}`

  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`)
    }

    return await response.json()
  } catch (error) {
    console.error('[marketplace-node] CivitAI API error:', error)
    throw error
  }
}
