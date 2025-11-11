// TEAM-460: CivitAI API integration
// TEAM-463: Now uses canonical types from WASM-generated artifacts-contract
// TEAM-429: Updated to use CivitaiFilters for type-safe filtering
// API Documentation: https://github.com/civitai/civitai/wiki/REST-API-Reference

// TEAM-463: Import canonical CivitAI types from WASM bindings
// These are generated from artifacts-contract via tsify
// TEAM-429: Now includes filter types from WASM bindings
import type {
  BaseModel,
  CivitaiFilters,
  CivitaiModel,
  CivitaiModelType,
  CivitaiModelVersion,
  CivitaiSort,
  NsfwFilter,
  NsfwLevel,
  TimePeriod,
} from '../../wasm/marketplace_sdk'

// Import enumerated constants to avoid magic strings
import { CIVITAI_BASE_MODEL, CIVITAI_MODEL_TYPE, CIVITAI_NSFW_LEVEL, CIVITAI_TIME_PERIOD } from './constants'

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
 * Fetch with retry logic and exponential backoff
 * TEAM-471: Added retry logic to handle 504 Gateway Timeout and rate limiting
 * TEAM-476: Added options parameter for headers (API key authentication)
 * @param url - URL to fetch
 * @param maxRetries - Maximum number of retries (default: 3)
 * @param options - Fetch options (headers, etc.)
 * @returns Response object
 */
async function fetchWithRetry(url: string, maxRetries = 3, options?: RequestInit): Promise<Response> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options)
      
      // Retry on 502 Bad Gateway, 504 Gateway Timeout, 524 Cloudflare Timeout, or 429 Too Many Requests
      if (response.status === 502 || response.status === 504 || response.status === 524 || response.status === 429) {
        const delay = Math.min(1000 * Math.pow(2, i), 10000) // Exponential backoff, max 10s
        console.log(`  ⏳ CivitAI ${response.status} - Retry ${i + 1}/${maxRetries} after ${delay}ms...`)
        await new Promise(resolve => setTimeout(resolve, delay))
        continue
      }
      
      // Return successful response or non-retryable error
      return response
    } catch (error) {
      // Network error - retry with exponential backoff
      if (i === maxRetries - 1) {
        console.error(`  ❌ Network error after ${maxRetries} retries:`, error)
        throw error
      }
      const delay = Math.min(1000 * Math.pow(2, i), 10000)
      console.log(`  ⏳ Network error - Retry ${i + 1}/${maxRetries} after ${delay}ms...`)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
  throw new Error(`Max retries (${maxRetries}) exceeded`)
}

/**
 * Fetch models from CivitAI API
 *
 * TEAM-429: Now uses CivitaiFilters for type-safe filtering
 * TEAM-471: Added retry logic with exponential backoff
 * @param filters - Filter configuration
 * @returns Array of CivitAI models
 */
export async function fetchCivitAIModels(filters: CivitaiFilters): Promise<CivitAIModel[]> {
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
  if (filters.model_type !== CIVITAI_MODEL_TYPE.ALL) {
    params.append('types', filters.model_type)
  } else {
    // Default: Checkpoint and LORA
    params.append('types', CIVITAI_MODEL_TYPE.CHECKPOINT)
    params.append('types', CIVITAI_MODEL_TYPE.LORA)
  }

  // Time period
  if (filters.time_period !== CIVITAI_TIME_PERIOD.ALL_TIME) {
    params.append('period', filters.time_period)
  }

  // Base model
  if (filters.base_model !== CIVITAI_BASE_MODEL.ALL) {
    params.append('baseModel', filters.base_model)
  }

  // NSFW filtering - convert level to numeric values
  // TEAM-467: FAIL FAST - Validate NSFW level is valid BEFORE lookup
  const nsfwLevelMap: Record<NsfwLevel, number[]> = {
    [CIVITAI_NSFW_LEVEL.NONE]: [1],
    [CIVITAI_NSFW_LEVEL.SOFT]: [1, 2],
    [CIVITAI_NSFW_LEVEL.MATURE]: [1, 2, 4],
    [CIVITAI_NSFW_LEVEL.X]: [1, 2, 4, 8],
    [CIVITAI_NSFW_LEVEL.XXX]: [1, 2, 4, 8, 16],
  }

  const nsfwLevels = nsfwLevelMap[filters.nsfw.max_level]
  if (!nsfwLevels) {
    const validLevels = Object.keys(nsfwLevelMap).join(', ')
    throw new Error(
      `❌ FATAL: Invalid NSFW level "${filters.nsfw.max_level}". Valid values: ${validLevels}\n` +
        `💡 Hint: Use '${CIVITAI_NSFW_LEVEL.XXX}' for all NSFW levels, or '${CIVITAI_NSFW_LEVEL.NONE}' for PG-only content.`,
    )
  }

  nsfwLevels.forEach((level: number) => {
    params.append('nsfwLevel', String(level))
  })

  // TEAM-476: Add API key for NSFW content access
  // CivitAI requires authentication to return NSFW content
  const CIVITAI_API_KEY = 'redacted'
  
  const url = `https://civitai.com/api/v1/models?${params}`

  // TEAM-476: FAIL FAST - Log the actual API URL being called
  console.log('[CivitAI API] Fetching:', url)
  console.log('[CivitAI API] NSFW levels being requested:', nsfwLevels)
  console.log('[CivitAI API] Using API key:', CIVITAI_API_KEY ? 'YES' : 'NO')

  try {
    // TEAM-471: Use retry logic instead of direct fetch
    // TEAM-476: Add Authorization header with API key for NSFW content
    const response = await fetchWithRetry(url, 3, {
      headers: {
        'Authorization': `Bearer ${CIVITAI_API_KEY}`,
      },
    })
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
 * TEAM-471: Added retry logic with exponential backoff
 * @param modelId - CivitAI model ID
 * @returns CivitAI model details
 */
export async function fetchCivitAIModel(modelId: number): Promise<CivitAIModel> {
  const url = `https://civitai.com/api/v1/models/${modelId}`

  try {
    // TEAM-471: Use retry logic instead of direct fetch
    const response = await fetchWithRetry(url, 3)
    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`)
    }

    return await response.json()
  } catch (error) {
    console.error('[marketplace-node] CivitAI API error:', error)
    throw error
  }
}
