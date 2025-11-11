// TEAM-476: CivitAI adapter - Fetch from CivitAI API and normalize to MarketplaceModel

import type {
  CivitAIListModelsParams,
  CivitAIModel,
  MarketplaceModel,
  PaginatedResponse,
} from '../..'

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
 * Fetch models from CivitAI API (LIST API)
 * 
 * API Docs: https://developer.civitai.com/docs/api/public-rest
 * 
 * @param params - Query parameters for filtering/sorting
 * @param apiKey - Optional API key for NSFW content
 * @returns Paginated response with normalized MarketplaceModel[]
 */
export async function fetchCivitAIModels(
  params: CivitAIListModelsParams = {},
  apiKey?: string
): Promise<PaginatedResponse<MarketplaceModel>> {
  // Build query string
  const queryParams = new URLSearchParams()

  if (params.limit) queryParams.append('limit', String(params.limit))
  if (params.page) queryParams.append('page', String(params.page))
  if (params.query) queryParams.append('query', params.query)
  if (params.tag) queryParams.append('tag', params.tag)
  if (params.username) queryParams.append('username', params.username)
  if (params.sort) queryParams.append('sort', params.sort)
  if (params.period) queryParams.append('period', params.period)
  if (params.rating !== undefined) queryParams.append('rating', String(params.rating))
  if (params.favorites !== undefined) queryParams.append('favorites', String(params.favorites))
  if (params.hidden !== undefined) queryParams.append('hidden', String(params.hidden))
  if (params.primaryFileOnly !== undefined) queryParams.append('primaryFileOnly', String(params.primaryFileOnly))
  if (params.cursor) queryParams.append('cursor', params.cursor)

  // Handle types array
  if (params.types && params.types.length > 0) {
    params.types.forEach((type) => queryParams.append('types', type))
  }

  // Handle NSFW levels array (bit flags: 1,2,4,8,16)
  if (params.nsfwLevel && params.nsfwLevel.length > 0) {
    params.nsfwLevel.forEach((level) => queryParams.append('nsfw', String(level)))
  }

  // Handle base models array
  if (params.baseModels && params.baseModels.length > 0) {
    params.baseModels.forEach((model) => queryParams.append('baseModels', model))
  }

  // Handle commercial use array
  if (params.allowCommercialUse && params.allowCommercialUse.length > 0) {
    params.allowCommercialUse.forEach((use) => queryParams.append('allowCommercialUse', use))
  }

  // Handle boolean filters
  if (params.allowDerivatives !== undefined) queryParams.append('allowDerivatives', String(params.allowDerivatives))
  if (params.allowDifferentLicense !== undefined) queryParams.append('allowDifferentLicense', String(params.allowDifferentLicense))
  if (params.allowNoCredit !== undefined) queryParams.append('allowNoCredit', String(params.allowNoCredit))
  if (params.supercedes_nsfw !== undefined) queryParams.append('supercedes_nsfw', String(params.supercedes_nsfw))
  if (params.browsingLevel !== undefined) queryParams.append('browsingLevel', String(params.browsingLevel))

  const url = `${CIVITAI_API_BASE}/models?${queryParams.toString()}`

  console.log('[CivitAI API] Fetching:', url)

  try {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    // Add API key if provided (required for NSFW content)
    const key = apiKey || CIVITAI_API_KEY
    if (key) {
      headers['Authorization'] = `Bearer ${key}`
      console.log('[CivitAI API] Using API key for NSFW content')
    }

    const response = await fetch(url, { headers })

    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()

    // CivitAI returns { items: [], metadata: {} }
    const models: CivitAIModel[] = data.items || []
    const metadata = data.metadata || {}

    // Convert to MarketplaceModel format
    const items = models.map(convertCivitAIModel)

    console.log(`[CivitAI API] Fetched ${items.length} models`)

    return {
      items,
      meta: {
        page: metadata.currentPage || params.page || 1,
        limit: metadata.pageSize || params.limit || 100,
        total: metadata.totalItems,
        hasNext: !!metadata.nextPage || !!metadata.nextCursor,
        nextCursor: metadata.nextCursor,
      },
    }
  } catch (error) {
    console.error('[CivitAI API] Error:', error)
    throw error
  }
}

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
      headers['Authorization'] = `Bearer ${key}`
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

/**
 * Convert CivitAI model to normalized MarketplaceModel
 * 
 * @param model - Raw CivitAI model from API
 * @returns Normalized MarketplaceModel
 */
export function convertCivitAIModel(model: CivitAIModel): MarketplaceModel {
  // Get primary model version (first one)
  const primaryVersion = model.modelVersions?.[0]

  // Get primary image from primary version
  const primaryImage = primaryVersion?.images?.[0]

  // Get file size from primary version's primary file
  const primaryFile = primaryVersion?.files?.find((f) => f.primary) || primaryVersion?.files?.[0]
  const sizeBytes = primaryFile ? primaryFile.sizeKB * 1024 : undefined

  // Get created/updated dates
  const createdAt = primaryVersion?.createdAt ? new Date(primaryVersion.createdAt) : new Date()
  const updatedAt = primaryVersion?.updatedAt ? new Date(primaryVersion.updatedAt) : new Date()

  // Determine NSFW flag
  const nsfw = model.nsfw || false

  // Get description (may contain HTML)
  const description = model.description
    ? model.description.replace(/<[^>]*>/g, '').substring(0, 200) // Strip HTML, limit to 200 chars
    : undefined

  return {
    id: String(model.id),
    name: model.name,
    description,
    author: model.creator.username,
    downloads: model.stats.downloadCount,
    likes: model.stats.favoriteCount,
    tags: model.tags || [],
    type: model.type,
    nsfw,
    imageUrl: primaryImage?.url,
    sizeBytes,
    createdAt,
    updatedAt,
    url: `https://civitai.com/models/${model.id}`,
    license: model.allowCommercialUse?.join(', '),
    metadata: {
      // Store CivitAI-specific data
      modelId: model.id,
      poi: model.poi,
      nsfwLevel: model.nsfwLevel,
      allowNoCredit: model.allowNoCredit,
      allowCommercialUse: model.allowCommercialUse,
      allowDerivatives: model.allowDerivatives,
      allowDifferentLicense: model.allowDifferentLicense,
      stats: model.stats,
      mode: model.mode,
      status: model.status,
      primaryVersion: primaryVersion
        ? {
            id: primaryVersion.id,
            name: primaryVersion.name,
            baseModel: primaryVersion.baseModel,
            downloadUrl: primaryVersion.downloadUrl,
          }
        : undefined,
    },
  }
}
