// TEAM-476: HuggingFace list API - Fetch models from HuggingFace Hub

import type { MarketplaceModel, PaginatedResponse } from '../common'
import type { HuggingFaceListModelsParams, HuggingFaceModel } from './types'

/**
 * HuggingFace Hub API base URL
 */
const HF_API_BASE = 'https://huggingface.co/api'

/**
 * Convert HuggingFace model to normalized MarketplaceModel
 *
 * @param model - Raw HuggingFace model from API
 * @returns Normalized MarketplaceModel
 */
export function convertHFModel(model: HuggingFaceModel): MarketplaceModel {
  // Extract author from model ID (e.g., "meta-llama/Llama-2-7b-hf" → "meta-llama")
  const author = model.author || model.id.split('/')[0] || 'unknown'

  // Extract name from model ID (e.g., "meta-llama/Llama-2-7b-hf" → "Llama-2-7b-hf")
  const name = model.modelId || model.id.split('/')[1] || model.id

  // Determine if model is NSFW (conservative: if gated or no explicit safe tags)
  const nsfw = model.gated === true || model.gated === 'manual' || model.gated === 'auto'

  // Get model size from safetensors metadata
  const sizeBytes = model.safetensors?.total

  // Get license
  const license = model.cardData?.license || model.library_name

  // Get model type (pipeline_tag or library_name)
  const type = model.pipeline_tag || model.library_name || 'unknown'

  return {
    id: model.id,
    name,
    // description: Not available in list API, would need separate call - omit instead of undefined
    // imageUrl: HuggingFace doesn't provide preview images in list API - omit instead of undefined
    author,
    downloads: model.downloads || 0,
    likes: model.likes || 0,
    tags: model.tags || [],
    type,
    nsfw,
    ...(sizeBytes !== undefined && { sizeBytes }), // Only include if available
    createdAt: model.createdAt ? new Date(model.createdAt) : new Date(),
    updatedAt: model.lastModified ? new Date(model.lastModified) : new Date(),
    url: `https://huggingface.co/${model.id}`,
    ...(license && { license }), // Only include if available
    metadata: {
      // Store HuggingFace-specific data
      sha: model.sha,
      private: model.private,
      gated: model.gated,
      disabled: model.disabled,
      library_name: model.library_name,
      pipeline_tag: model.pipeline_tag,
      trendingScore: model.trendingScore,
      transformersInfo: model.transformersInfo,
    },
  }
}

/**
 * MVP-compatible task for llm-worker-rbee
 * Source: bin/80-global-worker-catalog/src/data.ts line 112
 * TEAM-484: Only show models we can actually run
 */
const MVP_PIPELINE_TAG = 'text-generation'

/**
 * MVP-compatible library for llm-worker-rbee
 * Source: bin/80-global-worker-catalog/src/data.ts line 115
 */
const MVP_LIBRARY = 'transformers'

/**
 * Fetch models from HuggingFace Hub API
 *
 * API Docs: https://huggingface.co/docs/hub/en/api
 *
 * @param params - Query parameters for filtering/sorting
 * @returns Paginated response with normalized MarketplaceModel[]
 *
 * TEAM-484: Conditional MVP compatibility - applies constraints only when no specific workers selected
 */
export async function fetchHuggingFaceModels(
  params: HuggingFaceListModelsParams = {},
): Promise<PaginatedResponse<MarketplaceModel>> {
  // TEAM-484: Conditional MVP compatibility logic
  // Apply MVP constraints only when no specific pipeline_tag, library, or filter is provided
  // This allows full filtering when users make explicit selections via worker-driven filters
  const shouldApplyMVPConstraints = !params.pipeline_tag && !params.library && !params.filter
  
  const finalParams: HuggingFaceListModelsParams = shouldApplyMVPConstraints ? {
    ...params,
    // Apply MVP defaults only for unfiltered requests
    pipeline_tag: MVP_PIPELINE_TAG,
    library: MVP_LIBRARY,
  } : params

  // Build query string
  const queryParams = new URLSearchParams()

  if (finalParams.search) queryParams.append('search', finalParams.search)
  if (finalParams.author) queryParams.append('author', finalParams.author)
  if (finalParams.sort) queryParams.append('sort', finalParams.sort)
  if (finalParams.direction) queryParams.append('direction', String(finalParams.direction))
  if (finalParams.limit) queryParams.append('limit', String(finalParams.limit))
  if (finalParams.full !== undefined) queryParams.append('full', String(finalParams.full))
  if (finalParams.config !== undefined) queryParams.append('config', String(finalParams.config))

  // TEAM-501: Fixed filter syntax - HuggingFace API uses direct query params, not filter=key:value
  // Direct query parameters (conditionally MVP-enforced)
  if (finalParams.pipeline_tag) queryParams.append('pipeline_tag', finalParams.pipeline_tag)
  if (finalParams.library) queryParams.append('library', finalParams.library)

  // Handle additional filter parameter (can be string or array) - for tags, etc.
  if (finalParams.filter) {
    if (Array.isArray(finalParams.filter)) {
      for (const f of finalParams.filter) {
        queryParams.append('filter', f)
      }
    } else {
      queryParams.append('filter', finalParams.filter)
    }
  }

  // Additional filters (if provided)
  if (finalParams.language) queryParams.append('language', finalParams.language)
  if (finalParams.dataset) queryParams.append('dataset', finalParams.dataset)
  if (finalParams.license) queryParams.append('license', finalParams.license)

  const url = `${HF_API_BASE}/models?${queryParams.toString()}`

  console.log('[HuggingFace API] Fetching:', url)

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`HuggingFace API error: ${response.status} ${response.statusText}`)
    }

    // HuggingFace returns array directly, not wrapped
    const models: HuggingFaceModel[] = await response.json()

    // Parse Link header for pagination
    const linkHeader = response.headers.get('Link')
    const hasNext = linkHeader ? linkHeader.includes('rel="next"') : false

    // Convert to MarketplaceModel format
    const items = models.map(convertHFModel)

    console.log(`[HuggingFace API] Fetched ${items.length} models`)

    return {
      items,
      meta: {
        page: 1, // HuggingFace doesn't use page numbers, uses Link header
        limit: params.limit || 100,
        hasNext,
        // total: HuggingFace doesn't provide total count - omit instead of undefined
      },
    }
  } catch (error) {
    console.error('[HuggingFace API] Error:', error)
    throw error
  }
}
