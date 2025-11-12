// TEAM-476: HuggingFace details API - Fetch single model by ID
// TEAM-478: Added README fetching

import type { MarketplaceModel } from '../common'
import type { HuggingFaceModel } from './types'
import { convertHFModel } from './list'

/**
 * HuggingFace Hub API base URL
 */
const HF_API_BASE = 'https://huggingface.co/api'

/**
 * Fetch model README content
 *
 * @param modelId - Model ID (e.g., "meta-llama/Llama-2-7b-hf")
 * @returns README markdown content or null if not found
 */
export async function fetchHuggingFaceModelReadme(modelId: string): Promise<string | null> {
  // HuggingFace README is at: https://huggingface.co/{modelId}/raw/main/README.md
  const url = `https://huggingface.co/${modelId}/raw/main/README.md`

  console.log('[HuggingFace API] Fetching README:', url)

  try {
    const response = await fetch(url)

    if (!response.ok) {
      console.warn('[HuggingFace API] README not found:', response.status)
      return null
    }

    return await response.text()
  } catch (error) {
    console.error('[HuggingFace API] README fetch error:', error)
    return null
  }
}

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
