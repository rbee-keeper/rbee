// TEAM-460: CivitAI API integration
// API Documentation: https://github.com/civitai/civitai/wiki/REST-API-Reference

export interface CivitAIModelVersion {
  id: number
  modelId: number
  name: string
  createdAt: string
  updatedAt: string
  trainedWords?: string[]
  baseModel: string
  files: Array<{
    name: string
    id: number
    sizeKB: number
    type: string
    metadata?: {
      fp?: string
      size?: string
      format?: string
    }
    downloadUrl: string
    primary?: boolean
  }>
  images?: Array<{
    url: string
    nsfw: boolean
    width: number
    height: number
  }>
  downloadUrl: string
}

export interface CivitAIModel {
  id: number
  name: string
  description?: string
  type: string
  poi?: boolean
  nsfw?: boolean
  allowNoCredit?: boolean
  allowCommercialUse?: string
  allowDerivatives?: boolean
  allowDifferentLicense?: boolean
  stats?: {
    downloadCount?: number
    favoriteCount?: number
    commentCount?: number
    ratingCount?: number
    rating?: number
  }
  creator?: {
    username?: string
    image?: string
  }
  tags?: string[]
  modelVersions?: CivitAIModelVersion[]
}

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
    nsfw = false,
    period,
    baseModel,
  } = options

  const params = new URLSearchParams({
    limit: String(limit),
    page: String(page),
    sort,
    nsfw: String(nsfw),
  })

  if (query) {
    params.append('query', query)
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
