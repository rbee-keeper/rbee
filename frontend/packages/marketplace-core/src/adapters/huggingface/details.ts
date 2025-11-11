// TEAM-476: HuggingFace details API - Fetch single model by ID

import type { MarketplaceModel } from '../common'
import type { HuggingFaceModel } from './types'
import { convertHFModel } from './list'

/**
 * HuggingFace Hub API base URL
 */
const HF_API_BASE = 'https://huggingface.co/api'

/**
 * Fetch a single model by ID
 *
 * @param modelId - Model ID (e.g., "meta-llama/Llama-2-7b-hf")
 * @returns Normalized MarketplaceModel
 */
export async function fetchHuggingFaceModel(modelId: string): Promise<MarketplaceModel> {
  const url = `${HF_API_BASE}/models/${modelId}`

  console.log('[HuggingFace API] Fetching model:', url)

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`HuggingFace API error: ${response.status} ${response.statusText}`)
    }

    const model: HuggingFaceModel = await response.json()

    return convertHFModel(model)
  } catch (error) {
    console.error('[HuggingFace API] Error:', error)
    throw error
  }
}
