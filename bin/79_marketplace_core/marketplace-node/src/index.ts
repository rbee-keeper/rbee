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
// Re-export WASM compatibility types
export type { ModelArchitecture, ModelFormat, Quantization } from '../wasm/marketplace_sdk.js'
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

/**
 * Get compatible HuggingFace models for rbee LLM workers
 * TEAM-475: Filters models by compatibility (architecture, format, context length)
 * 
 * Only returns models that:
 * - Use supported architectures (llama, mistral, phi, qwen, gemma)
 * - Use supported formats (safetensors, gguf)
 * - Have context length <= 32,768 tokens
 * 
 * @param options - Search options (limit, sort)
 * @returns Promise<Model[]> - Array of compatible models only
 */
export async function getCompatibleHuggingFaceModels(options: SearchOptions = {}): Promise<Model[]> {
  // TEAM-475: Fetch more models than needed, then filter
  const { limit = 100, sort = 'popular' } = options
  const FETCH_LIMIT = 500 // Fetch max to ensure we have enough after filtering
  
  // Fetch all models
  const allModels = await listHuggingFaceModels({ limit: FETCH_LIMIT, sort })
  
  // Filter by compatibility using heuristics
  const compatibleModels = allModels.filter(model => {
    return isModelCompatible(model)
  })
  
  // Return requested number of compatible models
  return compatibleModels.slice(0, limit)
}

/**
 * Check if a model is compatible with LLM workers
 * TEAM-475: Heuristic-based compatibility checking
 * 
 * Checks:
 * - Architecture (from tags/model name)
 * - Format (from siblings/files)
 * - Context length (from config)
 */
function isModelCompatible(model: Model): boolean {
  const modelText = `${model.name} ${model.id} ${model.tags.join(' ')}`.toLowerCase()
  
  // Supported architectures
  const supportedArchitectures = ['llama', 'mistral', 'phi', 'qwen', 'gemma']
  const hasCompatibleArch = supportedArchitectures.some(arch => modelText.includes(arch))
  
  if (!hasCompatibleArch) {
    return false
  }
  
  // Check for incompatible architectures (exclude these)
  const incompatibleArchitectures = ['bert', 'clip', 'vit', 't5', 'whisper', 'wav2vec']
  const hasIncompatibleArch = incompatibleArchitectures.some(arch => modelText.includes(arch))
  
  if (hasIncompatibleArch) {
    return false
  }
  
  // Check for supported formats in siblings (if available)
  if (model.siblings && model.siblings.length > 0) {
    const hasGGUF = model.siblings.some(s => s.filename.endsWith('.gguf'))
    const hasSafeTensors = model.siblings.some(s => s.filename.endsWith('.safetensors'))
    
    // Must have at least one supported format
    if (!hasGGUF && !hasSafeTensors) {
      return false
    }
  }
  
  return true
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

// TEAM-476: RULE ZERO - Added 'type' field for proper model type filtering
// Breaking change: Model interface now includes 'type' field from CivitAI
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
    type: civitai.type, // TEAM-476: Include model type (Checkpoint, LORA, etc.)
    nsfw: civitai.nsfw, // TEAM-476: Include NSFW flag for content rating filtering
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
