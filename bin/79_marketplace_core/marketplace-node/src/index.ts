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

import { fetchHFModels, fetchHFModel, type HFModel } from './huggingface'
import type { Model, SearchOptions, ModelMetadata, CompatibilityResult } from './types'

// TEAM-413: Lazy-load WASM to avoid build-time issues in Next.js
// The WASM module is only loaded when compatibility checking is actually used
let wasmModule: typeof import('../wasm/marketplace_sdk') | null = null

async function getWasmModule() {
  if (!wasmModule) {
    wasmModule = await import('../wasm/marketplace_sdk')
  }
  return wasmModule
}

// Re-export types
export type { Model, SearchOptions, Worker, ModelFile, ModelMetadata, CompatibilityResult } from './types'

/**
 * Convert HuggingFace model to our Model type
 */
function convertHFModel(hf: HFModel): Model {
  const parts = hf.id.split('/')
  const name = parts.length >= 2 ? parts[1] : hf.id
  const author = parts.length >= 2 ? parts[0] : hf.author || null
  
  // Calculate total size
  const totalBytes = hf.siblings?.reduce((sum, file) => sum + (file.size || 0), 0) || 0
  
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
    siblings: hf.siblings || [],
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
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
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
export async function searchHuggingFaceModels(
  query: string,
  options: SearchOptions = {}
): Promise<Model[]> {
  const { limit = 50 } = options
  
  const hfModels = await fetchHFModels(query, 'downloads', limit)
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
export async function listHuggingFaceModels(
  options: SearchOptions = {}
): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort
  
  const hfModels = await fetchHFModels(undefined, sortParam, limit)
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
  const architectureTags = model.tags?.filter(t => 
    ['llama', 'mistral', 'phi', 'qwen', 'gemma'].some(arch => t.toLowerCase().includes(arch))
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
  
  // Detect format from files
  const files = model.siblings || []
  const hasSafetensors = files.some(f => f.rfilename.endsWith('.safetensors'))
  const hasGguf = files.some(f => f.rfilename.endsWith('.gguf'))
  
  let format = 'unknown'
  if (hasSafetensors) format = 'safetensors'
  else if (hasGguf) format = 'gguf'
  
  // Get context length from config
  const maxContextLength = model.config?.max_position_embeddings || 2048
  
  // Calculate size
  const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0)
  
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
  
  // Lazy-load WASM and call compatibility function
  const wasm = await getWasmModule()
  return wasm.is_model_compatible_wasm(metadata)
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
    })
  )
  return results.filter(r => r.compatible).map(r => r.model)
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
  options: SearchOptions & { onlyCompatible?: boolean } = {}
): Promise<Model[]> {
  const { limit = 50, onlyCompatible = true } = options
  
  // Fetch more models than requested to account for filtering
  const fetchLimit = onlyCompatible ? limit * 3 : limit
  
  const hfModels = await fetchHFModels(query, 'downloads', fetchLimit)
  
  // Filter compatible models
  const compatible = onlyCompatible 
    ? await filterCompatibleModels(hfModels)
    : hfModels
  
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
export async function listCompatibleModels(
  options: SearchOptions = {}
): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort
  
  // Fetch more models to account for filtering
  const fetchLimit = limit * 3
  
  const hfModels = await fetchHFModels(undefined, sortParam, fetchLimit)
  const compatible = await filterCompatibleModels(hfModels)
  
  return compatible.slice(0, limit).map(convertHFModel)
}

// TODO: Implement CivitAI and Worker Catalog in future phases
