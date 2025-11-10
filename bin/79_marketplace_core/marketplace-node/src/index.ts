// TEAM-415: Node.js wrapper for marketplace-sdk
// TEAM-410: Added compatibility filtering
// Provides a clean API for Next.js and other Node.js apps

/**
 * Marketplace Node.js SDK
 *
 * This package provides a clean API for searching HuggingFace models,
 * CivitAI models, and browsing the worker catalog.
 *
 * @example
 * ```typescript
 * import { searchHuggingFaceModels } from '@rbee/marketplace-node'
 *
 * const models = await searchHuggingFaceModels('llama', { limit: 10 })
 * console.log(models)
 * ```
 */

import { 
  type CivitAIModel, 
  type CivitaiFilters,
  type TimePeriod,
  type CivitaiModelType,
  type BaseModel,
  type CivitaiSort,
  type NsfwLevel,
  type NsfwFilter,
  fetchCivitAIModel, 
  fetchCivitAIModels 
} from './civitai'
import { fetchHFModel, fetchHFModels, fetchHFModelReadme, type HFModel } from './huggingface'
import type { CompatibilityConfidence, CompatibilityResult, Model, ModelMetadata, SearchOptions } from './types'

// TEAM-413: Lazy-load WASM to avoid build-time issues in Next.js
// The WASM module is only loaded when compatibility checking is actually used
let wasmModule: typeof import('../wasm/marketplace_sdk') | null = null

async function getWasmModule() {
  if (!wasmModule) {
    wasmModule = await import('../wasm/marketplace_sdk')
  }
  return wasmModule
}

// TEAM-429: Re-export CivitAI types and filters
export type { 
  CivitAIModel,
  CivitaiFilters,
  TimePeriod,
  CivitaiModelType,
  BaseModel,
  CivitaiSort,
  NsfwLevel,
  NsfwFilter,
} from './civitai'
// Re-export types
export type { CompatibilityResult, Model, ModelFile, ModelMetadata, SearchOptions, Worker } from './types'
export type { HFModel } from './huggingface'
// TEAM-453: Export worker catalog functions
export { getWorker, listWorkers } from './workers'
export type { WorkerCatalogEntry } from './workers'

/**
 * Convert HuggingFace model to our Model type
 */
function convertHFModel(hf: HFModel): Model {
  const parts = hf.id.split('/')
  const name = parts.length >= 2 ? parts[1] : hf.id
  const author = parts.length >= 2 ? parts[0] : hf.author || null

  // Calculate total size
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
    // TEAM-463: Convert HuggingFace's rfilename → our canonical filename
    siblings:
      hf.siblings?.map((s) => ({ filename: s.rfilename, size: s.size || 0 })) ||
      [],
  }
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}

/**
 * Search HuggingFace models
 *
 * @param query - Search query string
 * @param options - Search options (limit, sort)
 * @returns Array of models matching the query
 *
 * @example
 * ```typescript
 * const models = await searchHuggingFaceModels('llama', { limit: 10 })
 * ```
 */
export async function searchHuggingFaceModels(query: string, options: SearchOptions = {}): Promise<Model[]> {
  const { limit = 50 } = options

  const hfModels = await fetchHFModels(query, { limit, sort: 'downloads' })
  return hfModels.map(convertHFModel)
}

/**
 * List HuggingFace models
 *
 * @param options - Search options (limit, sort)
 * @returns Array of models
 *
 * @example
 * ```typescript
 * const models = await listHuggingFaceModels({ limit: 50 })
 * ```
 */
export async function listHuggingFaceModels(options: SearchOptions = {}): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort

  const hfModels = await fetchHFModels(undefined, { limit, sort: sortParam })
  return hfModels.map(convertHFModel)
}

/**
 * Get a specific HuggingFace model
 *
 * @param modelId - Model ID (e.g., "meta-llama/Llama-3.2-1B")
 * @returns Model details
 *
 * @example
 * ```typescript
 * const model = await getHuggingFaceModel('meta-llama/Llama-3.2-1B')
 * ```
 */
export async function getHuggingFaceModel(modelId: string): Promise<Model> {
  const hfModel = await fetchHFModel(modelId)
  return convertHFModel(hfModel)
}

/**
 * TEAM-464: Get raw HuggingFace model with ALL fields preserved
 * 
 * Use this for detail pages where you need all the HuggingFace-specific data
 * like widgetData, spaces, safetensors, transformersInfo, etc.
 *
 * @param modelId - Model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
 * @returns Raw HuggingFace model with all API fields
 *
 * @example
 * ```typescript
 * const model = await getRawHuggingFaceModel('sentence-transformers/all-MiniLM-L6-v2')
 * console.log(model.widgetData) // Inference examples
 * console.log(model.spaces) // Spaces using this model
 * console.log(model.safetensors) // Model parameters
 * ```
 */
export async function getRawHuggingFaceModel(modelId: string): Promise<HFModel> {
  return fetchHFModel(modelId)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-410: COMPATIBILITY FILTERING
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Extract model metadata from HuggingFace model
 *
 * @param model - HuggingFace model object
 * @returns Model metadata for compatibility checking
 */
function extractModelMetadata(model: HFModel): ModelMetadata | null {
  // Extract architecture from tags or config
  const architectureTags =
    model.tags?.filter((t: string) =>
      ['llama', 'mistral', 'phi', 'qwen', 'gemma'].some((arch) => t.toLowerCase().includes(arch)),
    ) || []

  let architecture = 'unknown'
  if (architectureTags.length > 0) {
    const tag = architectureTags[0].toLowerCase()
    if (tag.includes('llama')) architecture = 'llama'
    else if (tag.includes('mistral')) architecture = 'mistral'
    else if (tag.includes('phi')) architecture = 'phi'
    else if (tag.includes('qwen')) architecture = 'qwen'
    else if (tag.includes('gemma')) architecture = 'gemma'
  }

  // Try config if tags didn't work
  if (architecture === 'unknown' && model.config?.model_type) {
    architecture = model.config.model_type
  }

  // Detect format from files (using raw HuggingFace API format with rfilename)
  const files = model.siblings || []
  const hasSafetensors = files.some((f) => f.rfilename.endsWith('.safetensors'))
  const hasGguf = files.some((f) => f.rfilename.endsWith('.gguf'))

  let format = 'unknown'
  if (hasSafetensors) format = 'safetensors'
  else if (hasGguf) format = 'gguf'

  // Get context length from config
  const maxContextLength = model.config?.max_position_embeddings || 2048

  // Calculate size
  const totalBytes = files.reduce((sum: number, file: { size?: number }) => sum + (file.size || 0), 0)

  // Determine parameter count from model ID or config
  let parameters = 'unknown'
  const idMatch = model.id.match(/(\d+\.?\d*)[bB]/i)
  if (idMatch) {
    parameters = idMatch[1] + 'B'
  }

  return {
    architecture,
    format,
    quantization: null,
    parameters,
    sizeBytes: totalBytes,
    maxContextLength,
  }
}

/**
 * Check if a model is compatible with our workers
 *
 * @param model - HuggingFace model object
 * @returns Compatibility result with detailed information
 *
 * @example
 * ```typescript
 * const model = await getHuggingFaceModel('meta-llama/Llama-3.2-1B')
 * const compat = checkModelCompatibility(model)
 *
 * if (compat.compatible) {
 *   console.log('Model is compatible!')
 * }
 * ```
 */
export async function checkModelCompatibility(model: HFModel): Promise<CompatibilityResult> {
  const metadata = extractModelMetadata(model)

  if (!metadata) {
    return {
      compatible: false,
      confidence: 'none',
      reasons: ['Could not extract model metadata'],
      warnings: [],
      recommendations: [],
    }
  }

  // TEAM-463: Use WASM compatibility checking with proper type conversion
  const wasm = await getWasmModule()
  
  // Convert our ModelMetadata to WASM ModelMetadata format
  // WASM expects enum values, not strings
  const wasmMetadata = {
    architecture: metadata.architecture as any, // WASM ModelArchitecture enum
    format: metadata.format as any, // WASM ModelFormat enum
    quantization: metadata.quantization as any, // WASM Quantization enum or null
    parameters: metadata.parameters,
    sizeBytes: metadata.sizeBytes,
    maxContextLength: metadata.maxContextLength,
  }
  
  return wasm.is_model_compatible_wasm(wasmMetadata)
}

/**
 * Filter models to only include compatible ones
 *
 * @param models - Array of HuggingFace models
 * @returns Array of compatible models
 *
 * @example
 * ```typescript
 * const allModels = await listHuggingFaceModels({ limit: 100 })
 * const compatible = filterCompatibleModels(allModels)
 * console.log(`${compatible.length} compatible models found`)
 * ```
 */
export async function filterCompatibleModels(models: HFModel[]): Promise<HFModel[]> {
  const results = await Promise.all(
    models.map(async (model) => {
      const result = await checkModelCompatibility(model)
      return { model, compatible: result.compatible }
    }),
  )
  return results.filter((r) => r.compatible).map((r) => r.model)
}

/**
 * Search HuggingFace models (compatible only)
 *
 * @param query - Search query string
 * @param options - Search options (limit, sort, onlyCompatible)
 * @returns Array of compatible models matching the query
 *
 * @example
 * ```typescript
 * const models = await searchCompatibleModels('llama', { limit: 10 })
 * // All returned models are guaranteed to be compatible
 * ```
 */
export async function searchCompatibleModels(
  query: string,
  options: SearchOptions & { onlyCompatible?: boolean } = {},
): Promise<Model[]> {
  const { limit = 50, onlyCompatible = true } = options

  // Fetch more models than requested to account for filtering
  const fetchLimit = onlyCompatible ? limit * 3 : limit

  const hfModels = await fetchHFModels(query, { limit: fetchLimit, sort: 'downloads' })

  // Filter compatible models
  const compatible = onlyCompatible ? await filterCompatibleModels(hfModels) : hfModels

  // Limit to requested amount
  return compatible.slice(0, limit).map(convertHFModel)
}

/**
 * List compatible HuggingFace models
 *
 * @param options - Search options (limit, sort)
 * @returns Array of compatible models
 *
 * @example
 * ```typescript
 * const models = await listCompatibleModels({ limit: 50 })
 * // All returned models are guaranteed to be compatible
 * ```
 */
export async function listCompatibleModels(options: SearchOptions = {}): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort

  // Fetch more models to account for filtering
  const fetchLimit = limit * 3

  const hfModels = await fetchHFModels(undefined, { limit: fetchLimit, sort: sortParam })
  const compatible = await filterCompatibleModels(hfModels)

  return compatible.slice(0, limit).map(convertHFModel)
}

/**
 * Convert CivitAI model to our Model type
 */
function convertCivitAIModel(civitai: CivitAIModel): Model {
  // TEAM-422: Defensive programming - handle missing/undefined fields
  // Get the latest version
  const latestVersion = civitai.modelVersions?.[0]

  // Calculate total size from all files
  const totalBytes = latestVersion?.files?.reduce((sum, file) => sum + file.sizeKb * 1024, 0) || 0

  // TEAM-422: Get first non-NSFW image from latest version
  const imageUrl = latestVersion?.images?.find((img) => !img.nsfw)?.url

  // Safe fallbacks for all fields
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

/**
 * Create default CivitAI filters
 * TEAM-429: Helper function for creating filter objects
 */
export function createDefaultCivitaiFilters(): CivitaiFilters {
  return {
    time_period: 'AllTime',
    model_type: 'All',
    base_model: 'All',
    sort: 'Most Downloaded',
    nsfw: {
      max_level: 'None',
      blur_mature: true,
    },
    page: null,
    limit: 100,
  }
}

/**
 * Get compatible CivitAI models
 *
 * TEAM-429: Now uses CivitaiFilters for type-safe filtering
 * @param filters - Optional filter configuration (uses defaults if not provided)
 * @returns Array of compatible CivitAI models
 *
 * @example
 * ```typescript
 * // Use defaults
 * const models = await getCompatibleCivitaiModels()
 * 
 * // Custom filters
 * const filtered = await getCompatibleCivitaiModels({
 *   ...createDefaultCivitaiFilters(),
 *   timePeriod: 'Month',
 *   baseModel: 'SDXL 1.0',
 *   limit: 50,
 * })
 * ```
 */
export async function getCompatibleCivitaiModels(
  filters?: Partial<CivitaiFilters>,
): Promise<Model[]> {
  // Merge with defaults
  const mergedFilters: CivitaiFilters = {
    ...createDefaultCivitaiFilters(),
    ...filters,
  }

  try {
    const civitaiModels = await fetchCivitAIModels(mergedFilters)
    return civitaiModels.map(convertCivitAIModel)
  } catch (error) {
    console.error('[marketplace-node] Failed to fetch CivitAI models:', error)
    return []
  }
}

/**
 * Get a specific CivitAI model by ID
 *
 * @param modelId - CivitAI model ID
 * @returns Raw CivitAI model data
 *
 * @example
 * ```typescript
 * const model = await getCivitaiModel(257749)
 * console.log(model.name, model.creator.username)
 * ```
 */
export async function getCivitaiModel(modelId: number): Promise<CivitAIModel> {
  return fetchCivitAIModel(modelId)
}

/**
 * TEAM-464: Fetch README.md from HuggingFace model repository
 * 
 * @param modelId - Model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
 * @param revision - Git revision (default: "main")
 * @returns README content as markdown string, or null if not found
 */
export async function getHuggingFaceModelReadme(
  modelId: string,
  revision?: string,
): Promise<string | null> {
  return fetchHFModelReadme(modelId, revision)
}

// TODO: Implement Worker Catalog in future phases
