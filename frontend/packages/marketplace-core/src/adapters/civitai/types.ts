// TEAM-476: CivitAI API types based on https://github.com/civitai/civitai/wiki/REST-API-Reference

/**
 * CivitAI Model Types
 */
export type CivitAIModelType =
  | 'Checkpoint'
  | 'TextualInversion'
  | 'Hypernetwork'
  | 'AestheticGradient'
  | 'LORA'
  | 'Controlnet'
  | 'Poses'

/**
 * CivitAI Sort Options
 */
export type CivitAISort = 'Highest Rated' | 'Most Downloaded' | 'Newest'

/**
 * CivitAI Time Period
 */
export type CivitAITimePeriod = 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'

/**
 * CivitAI Base Models
 */
export type CivitAIBaseModel =
  | 'SD 1.4'
  | 'SD 1.5'
  | 'SD 2.0'
  | 'SD 2.0 768'
  | 'SD 2.1'
  | 'SD 2.1 768'
  | 'SD 2.1 Unclip'
  | 'SDXL 0.9'
  | 'SDXL 1.0'
  | 'SDXL 1.0 LCM'
  | 'SDXL Distilled'
  | 'SDXL Turbo'
  | 'SD 3'
  | 'SD 3.5'
  | 'Pony'
  | 'Illustrious'
  | 'Flux.1 D'
  | 'Flux.1 S'
  | 'Other'

/**
 * NSFW Levels (bit flags)
 * 1 = None, 2 = Soft, 4 = Mature, 8 = X, 16 = XXX
 */
export type CivitAINSFWLevel = 1 | 2 | 4 | 8 | 16

/**
 * CivitAI Model Status
 */
export type CivitAIModelStatus = 'Published' | 'Archived' | 'TakenDown'

/**
 * CivitAI File Format
 */
export type CivitAIFileFormat = 'SafeTensor' | 'PickleTensor' | 'Other'

/**
 * CivitAI File Size
 */
export type CivitAIFileSize = 'full' | 'pruned'

/**
 * CivitAI File Precision
 */
export type CivitAIFilePrecision = 'fp16' | 'fp32'

/**
 * CivitAI Commercial Use
 */
export type CivitAICommercialUse = 'None' | 'Image' | 'Rent' | 'Sell'

/**
 * CivitAI Model File
 */
export interface CivitAIModelFile {
  id: number
  sizeKB: number
  name: string
  type: CivitAIFileFormat
  pickleScanResult?: string
  pickleScanMessage?: string
  virusScanResult?: string
  virusScanMessage?: string
  scannedAt?: string
  metadata?: {
    format?: CivitAIFileFormat
    size?: CivitAIFileSize
    fp?: CivitAIFilePrecision
  }
  hashes?: {
    AutoV1?: string
    AutoV2?: string
    SHA256?: string
    CRC32?: string
    BLAKE3?: string
    AutoV3?: string
  }
  primary?: boolean
  downloadUrl?: string
}

/**
 * CivitAI Model Image
 */
export interface CivitAIModelImage {
  id: number
  url: string
  nsfwLevel: CivitAINSFWLevel
  width: number
  height: number
  hash: string
  type?: string
  hasMeta?: boolean
  onSite?: boolean
  meta?: Record<string, unknown>
}

/**
 * CivitAI Model Version
 */
export interface CivitAIModelVersion {
  id: number
  modelId: number
  name: string
  createdAt: string
  updatedAt: string
  publishedAt?: string
  trainedWords?: string[]
  trainingStatus?: string
  trainingDetails?: string
  baseModel?: CivitAIBaseModel
  baseModelType?: string
  earlyAccessTimeFrame?: number
  description?: string
  stats?: {
    downloadCount: number
    ratingCount: number
    rating: number
    thumbsUpCount: number
  }
  files: CivitAIModelFile[]
  images: CivitAIModelImage[]
  downloadUrl?: string
}

/**
 * CivitAI Model Creator
 */
export interface CivitAIModelCreator {
  username: string
  image?: string
}

/**
 * CivitAI Model Stats
 */
export interface CivitAIModelStats {
  downloadCount: number
  favoriteCount: number
  thumbsUpCount: number
  thumbsDownCount: number
  commentCount: number
  ratingCount: number
  rating: number
  tippedAmountCount: number
}

/**
 * CivitAI Model (full response)
 */
export interface CivitAIModel {
  id: number
  name: string
  description?: string
  type: CivitAIModelType
  poi?: boolean
  nsfw: boolean
  nsfwLevel?: CivitAINSFWLevel
  allowNoCredit?: boolean
  allowCommercialUse?: CivitAICommercialUse[]
  allowDerivatives?: boolean
  allowDifferentLicense?: boolean
  stats: CivitAIModelStats
  creator: CivitAIModelCreator
  tags: string[]
  modelVersions: CivitAIModelVersion[]
  mode?: string
  status?: CivitAIModelStatus
}

/**
 * CivitAI List Models Response
 */
export interface CivitAIListModelsResponse {
  items: CivitAIModel[]
  metadata: {
    totalItems?: number
    currentPage?: number
    pageSize?: number
    totalPages?: number
    nextPage?: string
    prevPage?: string
    nextCursor?: string
    prevCursor?: string
  }
}

/**
 * CivitAI List Models Query Parameters
 */
export interface CivitAIListModelsParams {
  /** Limit the number of results */
  limit?: number | undefined

  /** Page number */
  page?: number | undefined

  /** Search query */
  query?: string | undefined

  /** Filter by tag */
  tag?: string | undefined

  /** Filter by username */
  username?: string | undefined

  /** Filter by model types */
  types?: CivitAIModelType[] | undefined

  /** Sort order */
  sort?: CivitAISort | undefined

  /** Time period */
  period?: CivitAITimePeriod | undefined

  /** Rating filter (1-5) */
  rating?: number | undefined

  /** Favorites filter */
  favorites?: boolean | undefined

  /** Hidden filter */
  hidden?: boolean | undefined

  /** Primary file only */
  primaryFileOnly?: boolean | undefined

  /** Allow commercial use */
  allowCommercialUse?: CivitAICommercialUse[] | undefined

  /** Allow derivatives */
  allowDerivatives?: boolean | undefined

  /** Allow different license */
  allowDifferentLicense?: boolean | undefined

  /** Allow no credit */
  allowNoCredit?: boolean | undefined

  /** Browse level (NSFW filter) */
  browsingLevel?: number | undefined

  /** NSFW levels (bit flags: 1,2,4,8,16) */
  nsfwLevel?: CivitAINSFWLevel[] | undefined

  /** Supercedes NSFW */
  supercedes_nsfw?: boolean | undefined

  /** Base models */
  baseModels?: CivitAIBaseModel[] | undefined

  /** Cursor for pagination */
  cursor?: string | undefined
}
