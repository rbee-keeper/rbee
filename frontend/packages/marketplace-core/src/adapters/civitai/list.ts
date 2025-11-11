// TEAM-476: CivitAI list API - Fetch models from CivitAI

import type { MarketplaceModel, PaginatedResponse } from '../common'
import type { CivitAIListModelsParams, CivitAIModel } from './types'

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
 * Convert CivitAI model to normalized MarketplaceModel
 *
 * @param model - Raw CivitAI model from API
 * @returns Normalized MarketplaceModel
 */
export function convertCivitAIModel(model: CivitAIModel): MarketplaceModel {
  // Primary version (first one, as before)
  const primaryVersion = model.modelVersions?.[0]

  // Primary image from primary version
  const primaryImage = primaryVersion?.images?.[0]

  // Primary file (prefer explicitly primary, then first)
  const primaryFile = primaryVersion?.files?.find((f) => f.primary) ?? primaryVersion?.files?.[0]

  // Size in bytes (guard against undefined / NaN)
  const sizeBytes = typeof primaryFile?.sizeKB === 'number' ? Math.max(0, primaryFile.sizeKB) * 1024 : undefined

  // Created/updated dates (retain behavior: default to "now" if missing)
  const createdAt = primaryVersion?.createdAt ? new Date(primaryVersion.createdAt) : new Date()

  const updatedAt = primaryVersion?.updatedAt ? new Date(primaryVersion.updatedAt) : new Date()

  // NSFW flag (same behavior, but normalized to boolean)
  const nsfw = Boolean(model.nsfw)

  // Description: strip HTML, collapse whitespace, truncate to 200 chars
  const description = model.description
    ? model.description
        .replace(/<[^>]*>/g, ' ') // remove tags
        .replace(/\s+/g, ' ') // collapse whitespace
        .trim()
        .slice(0, 200)
    : null

  const author = (model as any)?.creator?.username ?? (model as any)?.creator?.name ?? 'Unknown'

  const downloads = model.stats?.downloadCount ?? 0
  const likes = model.stats?.favoriteCount ?? 0
  const tags = model.tags ?? []

  const imageUrl = primaryImage?.url
  const license = model.allowCommercialUse?.length ? model.allowCommercialUse.join(', ') : undefined

  return {
    id: String(model.id),
    name: model.name,
    ...(description && { description }),
    author,
    downloads,
    likes,
    tags,
    type: model.type,
    nsfw,
    ...(imageUrl && { imageUrl }),
    ...(sizeBytes !== undefined && { sizeBytes }),
    createdAt,
    updatedAt,
    url: `https://civitai.com/models/${model.id}`,
    ...(license && { license }),
    metadata: {
      // CivitAI-specific data passthrough
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
  apiKey?: string,
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
    for (const type of params.types) {
      queryParams.append('types', type)
    }
  }

  // Handle NSFW levels array (bit flags: 1,2,4,8,16)
  if (params.nsfwLevel && params.nsfwLevel.length > 0) {
    for (const level of params.nsfwLevel) {
      queryParams.append('nsfw', String(level))
    }
  }

  // Handle base models array
  if (params.baseModels && params.baseModels.length > 0) {
    for (const model of params.baseModels) {
      queryParams.append('baseModels', model)
    }
  }

  // Handle commercial use array
  if (params.allowCommercialUse && params.allowCommercialUse.length > 0) {
    for (const use of params.allowCommercialUse) {
      queryParams.append('allowCommercialUse', use)
    }
  }

  // Handle boolean filters
  if (params.allowDerivatives !== undefined) queryParams.append('allowDerivatives', String(params.allowDerivatives))
  if (params.allowDifferentLicense !== undefined)
    queryParams.append('allowDifferentLicense', String(params.allowDifferentLicense))
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
      headers.Authorization = `Bearer ${key}`
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
