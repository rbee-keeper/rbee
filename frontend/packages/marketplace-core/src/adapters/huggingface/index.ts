// TEAM-476: HuggingFace adapter - Fetch from HuggingFace Hub API and normalize to MarketplaceModel

import type {
  HuggingFaceListModelsParams,
  HuggingFaceModel,
  MarketplaceModel,
  PaginatedResponse,
} from '../..'

/**
 * HuggingFace Hub API base URL
 */
const HF_API_BASE = 'https://huggingface.co/api'

/**
 * Fetch models from HuggingFace Hub API
 * 
 * API Docs: https://huggingface.co/docs/hub/en/api
 * 
 * @param params - Query parameters for filtering/sorting
 * @returns Paginated response with normalized MarketplaceModel[]
 */
export async function fetchHuggingFaceModels(
  params: HuggingFaceListModelsParams = {}
): Promise<PaginatedResponse<MarketplaceModel>> {
  // Build query string
  const queryParams = new URLSearchParams()

  if (params.search) queryParams.append('search', params.search)
  if (params.author) queryParams.append('author', params.author)
  if (params.sort) queryParams.append('sort', params.sort)
  if (params.direction) queryParams.append('direction', String(params.direction))
  if (params.limit) queryParams.append('limit', String(params.limit))
  if (params.full !== undefined) queryParams.append('full', String(params.full))
  if (params.config !== undefined) queryParams.append('config', String(params.config))

  // Handle filter parameter (can be string or array)
  if (params.filter) {
    if (Array.isArray(params.filter)) {
      params.filter.forEach((f) => queryParams.append('filter', f))
    } else {
      queryParams.append('filter', params.filter)
    }
  }

  // Additional filters
  if (params.pipeline_tag) queryParams.append('filter', `pipeline_tag:${params.pipeline_tag}`)
  if (params.library) queryParams.append('filter', `library:${params.library}`)
  if (params.language) queryParams.append('filter', `language:${params.language}`)
  if (params.dataset) queryParams.append('filter', `dataset:${params.dataset}`)
  if (params.license) queryParams.append('filter', `license:${params.license}`)

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
        total: undefined, // HuggingFace doesn't provide total count
      },
    }
  } catch (error) {
    console.error('[HuggingFace API] Error:', error)
    throw error
  }
}

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

  // Get primary image (if available from widget data)
  const imageUrl = undefined // HuggingFace doesn't provide preview images in list API

  // Get license
  const license = model.cardData?.license || model.library_name

  // Get model type (pipeline_tag or library_name)
  const type = model.pipeline_tag || model.library_name || 'unknown'

  return {
    id: model.id,
    name,
    description: undefined, // Not available in list API, would need separate call
    author,
    downloads: model.downloads || 0,
    likes: model.likes || 0,
    tags: model.tags || [],
    type,
    nsfw,
    imageUrl,
    sizeBytes,
    createdAt: model.createdAt ? new Date(model.createdAt) : new Date(),
    updatedAt: model.lastModified ? new Date(model.lastModified) : new Date(),
    url: `https://huggingface.co/${model.id}`,
    license,
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
