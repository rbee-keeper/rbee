// TEAM-405: Node.js wrapper for marketplace-sdk WASM
// Provides a clean API for Next.js and other Node.js apps

/**
 * Marketplace Node.js SDK
 * 
 * This package wraps the marketplace-sdk WASM module for use in Node.js environments
 * (Next.js, Express, etc.). It provides a clean API for searching HuggingFace models,
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

import { createSDKLoader } from '@rbee/sdk-loader'
import type { Model, Worker } from '@rbee/marketplace-sdk'

// Re-export types from marketplace-sdk
export type {
  Model,
  ModelSource,
  Worker,
  WorkerType,
  ModelFilters,
  SortOrder,
} from '@rbee/marketplace-sdk'

/**
 * Search options for marketplace queries
 */
export interface SearchOptions {
  /** Maximum number of results */
  limit?: number
  /** Sort order */
  sort?: 'popular' | 'recent' | 'trending'
}

// TEAM-405: SDK loader for marketplace WASM module
// Uses singleflight pattern to ensure only one load happens
const marketplaceLoader = createSDKLoader<any>({
  packageName: '@rbee/marketplace-sdk',
  requiredExports: ['init'], // WASM modules have init function
  timeout: 15000,
  maxAttempts: 3,
})

/**
 * Load marketplace SDK (internal helper)
 */
async function getSDK() {
  const { sdk } = await marketplaceLoader.loadOnce()
  return sdk
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
  const sdk = await getSDK()
  const { limit = 50 } = options
  
  // TODO: Call WASM search function when implemented
  // For now, return empty array until WASM functions are exposed
  return []
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
  const sdk = await getSDK()
  const { limit = 50 } = options
  
  // TODO: Call WASM list function when implemented
  // For now, return empty array until WASM functions are exposed
  return []
}

/**
 * Search CivitAI models
 * 
 * @param query - Search query string
 * @param options - Search options (limit, sort)
 * @returns Array of models matching the query
 * 
 * @example
 * ```typescript
 * const models = await searchCivitAIModels('anime', { limit: 10 })
 * ```
 */
export async function searchCivitAIModels(
  query: string,
  options: SearchOptions = {}
): Promise<Model[]> {
  const sdk = await getSDK()
  
  // TODO: Implement CivitAI search
  // For now, return empty array until CivitAI client is implemented
  return []
}

/**
 * List available worker binaries
 * 
 * @param options - Search options (limit, sort)
 * @returns Array of worker binaries
 * 
 * @example
 * ```typescript
 * const workers = await listWorkerBinaries({ limit: 20 })
 * ```
 */
export async function listWorkerBinaries(
  options: SearchOptions = {}
): Promise<Worker[]> {
  const sdk = await getSDK()
  
  // TODO: Implement worker catalog listing
  // For now, return empty array until worker catalog is implemented
  return []
}
