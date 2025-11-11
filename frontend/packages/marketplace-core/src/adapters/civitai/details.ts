// TEAM-476: CivitAI details API - Fetch single model by ID

import type { MarketplaceModel } from '../common'
import type { CivitAIModel } from './types'
import { convertCivitAIModel } from './list'

/**
 * CivitAI API base URL
 */
const CIVITAI_API_BASE = 'https://civitai.com/api/v1'

/**
 * CivitAI API key (optional, required for NSFW content)
 * Set via environment variable or pass as parameter
 */
const CIVITAI_API_KEY = process.env.CIVITAI_API_KEY || ''

/**
 * Fetch a single model by ID (DETAILS API)
 *
 * @param modelId - Model ID (number)
 * @param apiKey - Optional API key for NSFW content
 * @returns Normalized MarketplaceModel
 */
export async function fetchCivitAIModel(modelId: number, apiKey?: string): Promise<MarketplaceModel> {
  const url = `${CIVITAI_API_BASE}/models/${modelId}`

  console.log('[CivitAI API] Fetching model:', url)

  try {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    // Add API key if provided
    const key = apiKey || CIVITAI_API_KEY
    if (key) {
      headers.Authorization = `Bearer ${key}`
    }

    const response = await fetch(url, { headers })

    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`)
    }

    const model: CivitAIModel = await response.json()

    return convertCivitAIModel(model)
  } catch (error) {
    console.error('[CivitAI API] Error:', error)
    throw error
  }
}
