// TEAM-460: CivitAI API integration
// TEAM-461: Added automatic image fallback support

const CIVITAI_API_BASE = 'https://civitai.com/api/v1'

export interface CivitaiImage {
  url: string
  nsfw: boolean
  width: number
  height: number
}

export interface CivitaiFile {
  name: string
  id: number
  sizeKB: number
  type: string
  metadata?: {
    format?: string
    size?: string
    fp?: string
  }
  pickleScanResult?: string
  virusScanResult?: string
  scannedAt?: string
  primary?: boolean
  downloadUrl: string
}

export interface CivitaiModelVersion {
  id: number
  modelId: number
  name: string
  createdAt: string
  updatedAt: string
  trainedWords: string[]
  baseModel: string
  files: CivitaiFile[]
  images: CivitaiImage[]
  downloadCount: number
}

export interface CivitaiModel {
  id: number
  name: string
  description: string
  type: string
  nsfw: boolean
  tags: string[]
  creator: {
    username: string
    image?: string
  }
  stats: {
    downloadCount: number
    favoriteCount: number
    commentCount: number
    ratingCount: number
    rating: number
  }
  modelVersions: CivitaiModelVersion[]
}

export interface CivitaiListResponse {
  items: CivitaiModel[]
  metadata: {
    totalItems: number
    currentPage: number
    pageSize: number
    totalPages: number
  }
}

export interface ListCivitaiModelsOptions {
  limit?: number
  page?: number
  query?: string
  tag?: string
  username?: string
  types?: string[]
  sort?: string
  period?: string
  rating?: number
  favorites?: boolean
  hidden?: boolean
  primaryFileOnly?: boolean
  allowNoCredit?: boolean
  allowDerivatives?: boolean
  allowDifferentLicenses?: boolean
  allowCommercialUse?: string
  nsfw?: boolean
  baseModel?: string
}

/**
 * List models from CivitAI with optional filters
 * TEAM-461: Returns models with fallback images from the images array
 */
export async function listCivitaiModels(
  options: ListCivitaiModelsOptions = {}
): Promise<CivitaiListResponse> {
  const params = new URLSearchParams()
  
  // Add all options as query params
  Object.entries(options).forEach(([key, value]) => {
    if (value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach(v => params.append(key, v.toString()))
      } else {
        params.append(key, value.toString())
      }
    }
  })
  
  const url = `${CIVITAI_API_BASE}/models?${params.toString()}`
  
  try {
    const response = await fetch(url, {
      next: { revalidate: 3600 } // Cache for 1 hour
    })
    
    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('[civitai] Failed to list models:', error)
    throw error
  }
}

/**
 * Get a single model by ID
 */
export async function getCivitaiModel(id: number): Promise<CivitaiModel> {
  const url = `${CIVITAI_API_BASE}/models/${id}`
  
  try {
    const response = await fetch(url, {
      next: { revalidate: 3600 }
    })
    
    if (!response.ok) {
      throw new Error(`CivitAI API error: ${response.status}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error(`[civitai] Failed to get model ${id}:`, error)
    throw error
  }
}

/**
 * Get compatible CivitAI models (filtered for SFW, compatible formats)
 * TEAM-461: Returns models with automatic fallback images
 */
export async function getCompatibleCivitaiModels(
  options: ListCivitaiModelsOptions = {}
): Promise<any[]> {
  const defaultOptions: ListCivitaiModelsOptions = {
    limit: 100,
    nsfw: false, // Only SFW models
    sort: 'Highest Rated',
    types: ['Checkpoint', 'LORA'],
    ...options
  }
  
  const response = await listCivitaiModels(defaultOptions)
  
  // TEAM-461: Transform to include fallback images
  return response.items.map(model => {
    const latestVersion = model.modelVersions[0]
    const primaryFile = latestVersion?.files.find(f => f.primary) || latestVersion?.files[0]
    
    // Get all SFW images for fallback support
    const allImages = latestVersion?.images.filter(img => !img.nsfw) || []
    const primaryImage = allImages[0]?.url
    const fallbackImages = allImages.slice(1, 4).map(img => img.url) // Up to 3 fallbacks
    
    return {
      id: model.id.toString(),
      name: model.name,
      description: model.description,
      author: model.creator.username,
      imageUrl: primaryImage,
      fallbackImages, // TEAM-461: Automatic fallback support
      tags: model.tags.slice(0, 5),
      downloads: model.stats.downloadCount,
      likes: model.stats.favoriteCount,
      size: primaryFile ? `${(primaryFile.sizeKB / 1024 / 1024).toFixed(2)} GB` : 'Unknown',
      baseModel: latestVersion?.baseModel || 'Unknown',
      type: model.type,
    }
  })
}
