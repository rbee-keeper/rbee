// TEAM-415: Node.js wrapper for marketplace-sdk
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
import type { Model, SearchOptions } from './types'

// Re-export types
export type { Model, SearchOptions, Worker, ModelFile } from './types'

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

// TODO: Implement CivitAI and Worker Catalog in future phases
