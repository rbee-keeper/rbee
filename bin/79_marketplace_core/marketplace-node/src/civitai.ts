// TEAM-460: CivitAI API integration
// TEAM-463: Now uses canonical types from WASM-generated artifacts-contract
// API Documentation: https://github.com/civitai/civitai/wiki/REST-API-Reference

// TEAM-463: Import canonical CivitAI types from WASM bindings
// These are generated from artifacts-contract via tsify
import type {
  CivitaiModel,
  CivitaiModelVersion,
  CivitaiStats,
  CivitaiCreator,
  CivitaiFile,
  CivitaiImage,
} from '../wasm/marketplace_sdk'

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
 * @param options - Search options
 * @returns Array of CivitAI models
 */
export async function fetchCivitAIModels(
  options: {
    query?: string
    limit?: number
    page?: number
    types?: string[]
    sort?: 'Highest Rated' | 'Most Downloaded' | 'Newest'
    nsfw?: boolean
    period?: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
    baseModel?: string
  } = {},
): Promise<CivitAIModel[]> {
  const {
    query,
    limit = 20,
    page = 1,
    types = ['Checkpoint', 'LORA'],
    sort = 'Most Downloaded',
    nsfw, // No default - show all content
    period,
    baseModel,
  } = options

  const params = new URLSearchParams({
    limit: String(limit),
    page: String(page),
    sort,
  })

  if (query) {
    params.append('query', query)
  }

  if (nsfw !== undefined) {
    params.append('nsfw', String(nsfw))
  }

  // TEAM-422: CivitAI API requires multiple 'types' parameters, not comma-separated
  // Correct: ?types=Checkpoint&types=LORA
  // Wrong: ?types=Checkpoint,LORA
  if (types.length > 0) {
    types.forEach((type) => {
      params.append('types', type)
    })
  }

  // TEAM-422: Add period filter for time-based filtering
  if (period && period !== 'AllTime') {
    params.append('period', period)
  }

  // TEAM-422: Add baseModel filter for compatibility filtering
  if (baseModel) {
    params.append('baseModel', baseModel)
  }

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
