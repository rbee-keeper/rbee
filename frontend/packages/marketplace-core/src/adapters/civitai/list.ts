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
 * MVP-compatible base models for sd-worker-rbee
 * Source: bin/80-global-worker-catalog/src/data.ts lines 242-277
 * TEAM-484: Only show models we can actually run
 */
const MVP_BASE_MODELS = [
  // SD 1.x series
  'SD 1.4',
  'SD 1.5',
  'SD 1.5 LCM',
  'SD 1.5 Hyper',
  // SD 2.x series
  'SD 2.0',
  'SD 2.0 768',
  'SD 2.1',
  'SD 2.1 768',
  'SD 2.1 Unclip',
  // SDXL series
  'SDXL 0.9',
  'SDXL 1.0',
  'SDXL 1.0 LCM',
  'SDXL Distilled',
  'SDXL Turbo',
  'SDXL Lightning',
  'SDXL Hyper',
  // FLUX series
  'Flux.1 D',
  'Flux.1 S',
  'Flux.1 Krea',
  'Flux.1 Kontext',
  // Anime/Character models (SDXL-based)
  'Illustrious',
  'Pony',
  'NoobAI',
]

/**
 * MVP-compatible model types for sd-worker-rbee
 * Source: bin/80-global-worker-catalog/src/data.ts lines 238-241
 */
const MVP_MODEL_TYPES = ['Checkpoint', 'LORA']

/**
 * Fetch models from CivitAI API (LIST API)
 *
 * API Docs: https://developer.civitai.com/docs/api/public-rest
 *
 * @param params - Query parameters for filtering/sorting
 * @param apiKey - Optional API key for NSFW content
 * @returns Paginated response with normalized MarketplaceModel[]
 *
 * TEAM-484: Enforces MVP compatibility - only returns models sd-worker-rbee can run
 */
export async function fetchCivitAIModels(
  params: CivitAIListModelsParams = {},
  apiKey?: string,
): Promise<PaginatedResponse<MarketplaceModel>> {
  // TEAM-484: Merge user params with MVP compatibility filters
  const mvpParams: CivitAIListModelsParams = {
    ...params,
    // Override types to only MVP-compatible (Checkpoint, LORA)
    types: params.types?.length
      ? (params.types.filter((t) => MVP_MODEL_TYPES.includes(t)) as CivitAIListModelsParams['types'])
      : (MVP_MODEL_TYPES as CivitAIListModelsParams['types']),
    // Override baseModels to only MVP-compatible
    baseModels: params.baseModels?.length
      ? (params.baseModels.filter((m) => MVP_BASE_MODELS.includes(m)) as CivitAIListModelsParams['baseModels'])
      : (MVP_BASE_MODELS as CivitAIListModelsParams['baseModels']),
  }

  // Build query string
  const queryParams = new URLSearchParams()

  if (mvpParams.limit) queryParams.append('limit', String(mvpParams.limit))
  if (mvpParams.page) queryParams.append('page', String(mvpParams.page))
  if (mvpParams.query) queryParams.append('query', mvpParams.query)
  if (mvpParams.tag) queryParams.append('tag', mvpParams.tag)
  if (mvpParams.username) queryParams.append('username', mvpParams.username)
  if (mvpParams.sort) queryParams.append('sort', mvpParams.sort)
  if (mvpParams.period) queryParams.append('period', mvpParams.period)
  if (mvpParams.rating !== undefined) queryParams.append('rating', String(mvpParams.rating))
  if (mvpParams.favorites !== undefined) queryParams.append('favorites', String(mvpParams.favorites))
  if (mvpParams.hidden !== undefined) queryParams.append('hidden', String(mvpParams.hidden))
  if (mvpParams.primaryFileOnly !== undefined)
    queryParams.append('primaryFileOnly', String(mvpParams.primaryFileOnly))
  if (mvpParams.cursor) queryParams.append('cursor', mvpParams.cursor)

  // Handle types array
  if (mvpParams.types && mvpParams.types.length > 0) {
    for (const type of mvpParams.types) {
      queryParams.append('types', type)
    }
  }

  // Handle NSFW levels array (bit flags: 1,2,4,8,16)
  if (mvpParams.nsfwLevel && mvpParams.nsfwLevel.length > 0) {
    for (const level of mvpParams.nsfwLevel) {
      queryParams.append('nsfw', String(level))
    }
  }

  // Handle base models array
  if (mvpParams.baseModels && mvpParams.baseModels.length > 0) {
    for (const model of mvpParams.baseModels) {
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
