// TEAM-XXX RULE ZERO: Organized marketplace-node exports
// NO MORE MIXED PROVIDERS IN ONE FILE!
// Structure: civitai/, huggingface/, shared/

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CIVITAI EXPORTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// CivitAI types
export type {
  BaseModel,
  CivitAIFilterOptions,
  CivitAIModel,
  CivitaiFilters,
  CivitaiModelType,
  CivitaiSort,
  NsfwFilter,
  NsfwLevel,
  TimePeriod,
} from './civitai/index.js'
// CivitAI API functions
// CivitAI constants
// CivitAI filter utilities
export {
  applyCivitAIFilters,
  CIVITAI_BASE_MODELS,
  CIVITAI_DEFAULTS,
  CIVITAI_MODEL_TYPES,
  CIVITAI_NSFW_LEVELS,
  CIVITAI_SORTS,
  CIVITAI_TIME_PERIODS,
  CIVITAI_URL_SLUGS,
  fetchCivitAIModel,
  fetchCivitAIModels,
  filterCivitAIModels,
  sortCivitAIModels,
} from './civitai/index.js'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HUGGINGFACE EXPORTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// HuggingFace types
export type {
  HFAddedToken,
  HFModel,
  HuggingFaceFilterOptions,
  HuggingFaceFilters,
  HuggingFaceSort,
} from './huggingface/index.js'
// HuggingFace API functions
// HuggingFace constants
// HuggingFace filter utilities
export {
  applyHuggingFaceFilters,
  buildHuggingFaceFilterDescription,
  fetchHFModel,
  fetchHFModelReadme,
  fetchHFModels,
  filterHuggingFaceModels,
  HF_DEFAULTS,
  HF_LICENSES,
  HF_SIZES,
  HF_SORTS,
  HF_URL_SLUGS,
  LICENSE_PATTERNS,
  MODEL_SIZE_PATTERNS,
  sortHuggingFaceModels,
} from './huggingface/index.js'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SHARED EXPORTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Shared utilities and types used by both providers
export type { FilterableModel } from './shared/index.js'
export { formatBytes } from './shared/index.js'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LEGACY EXPORTS (for backwards compatibility during migration)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Re-export types
export type { CompatibilityResult, Model, ModelFile, ModelMetadata, SearchOptions, Worker } from './types.js'
export type { WorkerCatalogEntry } from './workers.js'
// Worker catalog functions
export { getWorker, listWorkers } from './workers.js'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HIGH-LEVEL API FUNCTIONS (convenience wrappers)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import { type CivitAIModel, fetchCivitAIModels } from './civitai/civitai.js'
import type { CivitaiFilters } from './civitai/constants.js'
import { CIVITAI_DEFAULTS } from './civitai/index.js'
import { fetchHFModel, fetchHFModelReadme, fetchHFModels, type HFModel } from './huggingface/huggingface.js'
import { formatBytes } from './shared/index.js'
import type { Model, SearchOptions } from './types.js'

/**
 * Convert HuggingFace model to our Model type
 */
function convertHFModel(hf: HFModel): Model {
  const parts = hf.id.split('/')
  const name = parts.length >= 2 ? parts[1] : hf.id
  const author = parts.length >= 2 ? parts[0] : hf.author || null

  const totalBytes = hf.siblings?.reduce((sum: number, file) => sum + (file.size || 0), 0) || 0

  return {
    id: hf.id,
    name,
    author: author || undefined,
    description: hf.cardData?.model_description || hf.description || '',
    downloads: hf.downloads || 0,
    likes: hf.likes || 0,
    size: formatBytes(totalBytes),
    tags: hf.tags || [],
    source: 'huggingface',
    createdAt: hf.createdAt,
    lastModified: hf.lastModified,
    config: hf.config,
    siblings: hf.siblings?.map((s) => ({ filename: s.rfilename, size: s.size || 0 })) || [],
  }
}

export async function searchHuggingFaceModels(query: string, options: SearchOptions = {}): Promise<Model[]> {
  const { limit = 50 } = options
  const hfModels = await fetchHFModels(query, { limit, sort: 'downloads' })
  return hfModels.map(convertHFModel)
}

export async function listHuggingFaceModels(options: SearchOptions = {}): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort
  const hfModels = await fetchHFModels(undefined, { limit, sort: sortParam })
  return hfModels.map(convertHFModel)
}

export async function getHuggingFaceModel(modelId: string): Promise<Model> {
  const hfModel = await fetchHFModel(modelId)
  return convertHFModel(hfModel)
}

export async function getRawHuggingFaceModel(modelId: string): Promise<HFModel> {
  return fetchHFModel(modelId)
}

export async function getHuggingFaceModelReadme(modelId: string, revision?: string): Promise<string | null> {
  return fetchHFModelReadme(modelId, revision)
}

function convertCivitAIModel(civitai: CivitAIModel): Model {
  const latestVersion = civitai.modelVersions?.[0]
  const totalBytes =
    latestVersion?.files?.reduce((sum: number, file: { sizeKb: number }) => sum + file.sizeKb * 1024, 0) || 0
  const imageUrl = latestVersion?.images?.find((img: { nsfw: boolean }) => !img.nsfw)?.url
  const author = civitai.creator?.username || 'Unknown'
  const description = civitai.description?.substring(0, 500) || `${civitai.type} model by ${author}`
  const downloads = civitai.stats?.downloadCount || 0
  const likes = civitai.stats?.favoriteCount || 0
  const tags = civitai.tags || []

  return {
    id: `civitai-${civitai.id}`,
    name: civitai.name || 'Unnamed Model',
    author,
    description,
    downloads,
    likes,
    size: formatBytes(totalBytes),
    tags,
    source: 'civitai' as const,
    imageUrl,
    createdAt: latestVersion?.createdAt ?? undefined,
    lastModified: latestVersion?.updatedAt ?? undefined,
  }
}

export function createDefaultCivitaiFilters(): CivitaiFilters {
  return {
    time_period: CIVITAI_DEFAULTS.TIME_PERIOD,
    model_type: CIVITAI_DEFAULTS.MODEL_TYPE,
    base_model: CIVITAI_DEFAULTS.BASE_MODEL,
    sort: CIVITAI_DEFAULTS.SORT,
    nsfw: {
      max_level: CIVITAI_DEFAULTS.NSFW_LEVEL,
      blur_mature: CIVITAI_DEFAULTS.BLUR_MATURE,
    },
    page: null,
    limit: CIVITAI_DEFAULTS.LIMIT,
  }
}

export async function getCompatibleCivitaiModels(filters?: Partial<CivitaiFilters>): Promise<Model[]> {
  const mergedFilters: CivitaiFilters = {
    ...createDefaultCivitaiFilters(),
    ...filters,
  }
  const civitaiModels = await fetchCivitAIModels(mergedFilters)
  return civitaiModels.map(convertCivitAIModel)
}

export async function getCivitaiModel(modelId: number): Promise<CivitAIModel> {
  const { fetchCivitAIModel } = await import('./civitai/civitai.js')
  return fetchCivitAIModel(modelId)
}
